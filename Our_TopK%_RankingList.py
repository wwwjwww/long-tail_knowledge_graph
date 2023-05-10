import numpy as np
# import args
from dataset import Reader
# import utils
import torch
from tranE import TransE, transE_loader, distanceL1
import os
import logging
import math
import argparse
import random
import json

def main():
    parser = argparse.ArgumentParser(add_help=True)
    # args, _ = parser.parse_known_args()
    parser.add_argument('--model', default='transE', help='model name')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], help='run training or evaluation')
    parser.add_argument('-ds', '--dataset', default='WN18RR', help='dataset')
    args, _ = parser.parse_known_args()
    parser.add_argument('--save_dir', default=f'./checkpoints/{args.dataset}/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=f'./checkpoints/{args.dataset}')
    parser.add_argument('--log_folder', default=f'./checkpoints/{args.dataset}/', help='model output directory')


    # data
    parser.add_argument('--data_path', default=f'./data/{args.dataset}/', help='path to the dataset')
    #parser.add_argument('--dir_emb_ent', default="entity2vec.txt", help='pretrain entity embeddings')
    #parser.add_argument('--dir_emb_rel', default="relation2vec.txt", help='pretrain entity embeddings')
    parser.add_argument('--entity2id_path', default=f'./data/{args.dataset}/entity2id.txt', help='path to entity2id')
    parser.add_argument('--relation2id_path', default=f'./data/{args.dataset}/relation2id.txt', help='path to relation2id')
    parser.add_argument('--relationVector_path', default=f'./data/{args.dataset}/relationVector.txt', help='path to relationVector')
    parser.add_argument('--entityVector_path', default=f'./data/{args.dataset}/entityVector.txt', help='path to entityVector')
    parser.add_argument('--num_batch', default=2740, type=int, help='number of batch')
    parser.add_argument('--num_train', default=0, type=int, help='number of triples')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--total_ent', default=0, type=int, help='number of entities')
    parser.add_argument('--total_rel', default=0, type=int, help='number of relations')


    # optimization
    parser.add_argument('--max_epoch', default=6, help='max epochs')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate')
    parser.add_argument('--gama', default=0.5, type=float, help="margin parameter")
    parser.add_argument('--lam', default=0.1, type=float, help="trade-off parameter")
    parser.add_argument('--mu', default=0.001, type=float, help="gated attention parameter")
    parser.add_argument('--anomaly_ratio', default=0.05, type=float, help="anomaly ratio")
    parser.add_argument('--num_anomaly_num', default=300, type=int, help="number of anomalies")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    dataset = Reader(args, args.data_path)
    if args.model == 'transE':
        train(args, dataset, device)
        test(args, dataset, device)
    else:
        if args.mode == 'train':
            train(args, dataset, device)
        elif args.mode == 'test':
        # raise NotImplementedError
             test(args, dataset, device)
        else:
             raise ValueError('Invalid mode')

def train(args, dataset, device):
    # Dataset parameters
    # data_name = args.dataset
    model_name = args.model
    all_triples = dataset.train_data
    # labels = dataset.labels
    total_num_anomalies = dataset.num_anomalies
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.log_folder, model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation

    if model_name == "transE":
        entity_set = dataset.ent2id.values()
        relation_set = dataset.rel2id.values()
        triple_list = list(dataset.train_data)

        transE = TransE(entity_set, relation_set, triple_list, args.relationVector_path, args.entityVector_path, embedding_dim=50, learning_rate=0.01, margin=4, L1=True )
        transE.emb_initialize()
        transE.train(epochs=1001)

    print("The training ends!")

def test(args, dataset, device):
    # Dataset parameters
    # data_name = args.dataset
    device = torch.device('cpu')
    data_path = args.data_path
    model_name = args.model

    total_num_anomalies = dataset.num_anomalies
    print(dataset.num_long_tail_triple)
    total_num_long_tail_anomalies = 3483
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.log_folder, model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio)  + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation

    entityId2vec, relationId2vec = transE_loader(data_path)
    all_triples = dataset.triples_with_anomalies
    anomaly_label = dataset.triples_with_anomalies_labels
    long_tail_label = dataset.triples_with_long_tail_labels

    dlist = []
    dlist_long_tail = []

    for idx in range(len(all_triples)):
        h = all_triples[idx][0]
        r = all_triples[idx][1]
        t = all_triples[idx][2]
        dlist.append((idx, distanceL1(np.array(entityId2vec[h]), np.array(relationId2vec[r]), np.array(entityId2vec[t]))))
        dlist = sorted(dlist, key=lambda val: val[1], reverse=True)
    with open("rank_triple", 'w') as f:
        f.write(str(dlist))
        f.close()

    for idx in range(len(all_triples)):
        if long_tail_label[idx] == 1:
            h = all_triples[idx][0]
            r = all_triples[idx][1]
            t = all_triples[idx][2]
            dlist_long_tail.append((idx, distanceL1(np.array(entityId2vec[h]), np.array(relationId2vec[r]), np.array(entityId2vec[t]))))
            dlist_long_tail = sorted(dlist_long_tail, key=lambda val: val[1], reverse=True)
    with open("rank_triple_long_tail", 'w') as f:
        f.write(str(dlist_long_tail))
        f.close()
    '''
    with open("rank_triple") as f:
        dlist = f.read()
        dlist = eval(dlist)
    
    with open("rank_triple_long_tail") as f1:
        dlist_long_tail = f1.read()
        dlist_long_tail = eval(dlist_long_tail)     
    '''


    ratios = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
              0.20, 0.30, 0.45]
    for i in range(len(ratios)):
        num_k = int(ratios[i] * dataset.num_original_triples)
        num_k_long_tail = int(ratios[i] * dataset.num_long_tail_triple)
        anomaly_detected = 0
        anomaly_detected_long_tail = 0
        
        for j in range(num_k):
            if anomaly_label[int(dlist[j][0])] == 1:
                anomaly_detected += 1
        precision_k = anomaly_detected * 1.0 / num_k
        recall = anomaly_detected * 1.0 / total_num_anomalies
        logging.info(
            '[Test][%s][%s] Precision %f -- %f : %f' % (
            args.dataset, model_name, args.anomaly_ratio, ratios[i], precision_k))
        logging.info(
            '[Test][%s][%s] Recall  %f-- %f : %f' % (args.dataset, model_name, args.anomaly_ratio, ratios[i], recall))
        logging.info('[Test][%s][%s] anomalies in total: %d -- discovered:%d -- K : %d' % (
            args.dataset, model_name, total_num_anomalies, anomaly_detected, num_k))

        for m in range(num_k_long_tail):
            if anomaly_label[int(dlist_long_tail[m][0])] == 1:
                anomaly_detected_long_tail += 1
        precision_k_long_tail = anomaly_detected_long_tail * 1.0 / num_k_long_tail
        recall_long_tail = anomaly_detected_long_tail * 1.0 / total_num_long_tail_anomalies
        logging.info(
            '[Test][%s][%s] Long-tail triple Precision %f -- %f : %f' % (
                args.dataset, model_name, args.anomaly_ratio, ratios[i], precision_k_long_tail))
        logging.info(
            '[Test][%s][%s] Long-tail triple Recall  %f-- %f : %f' % (args.dataset, model_name, args.anomaly_ratio, ratios[i], recall_long_tail))
        logging.info('[Test][%s][%s] Long-tail triple anomalies in total: %d -- discovered:%d -- K : %d' % (
            args.dataset, model_name, total_num_anomalies, anomaly_detected_long_tail, num_k_long_tail))


if __name__ == '__main__':
    main()