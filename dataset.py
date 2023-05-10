import os.path
import json
import numpy as np
import random
import torch
import math
import argparse
from random import shuffle


class Reader:
    def __init__(self, args, path):

        self.ent2id = dict()
        self.rel2id = dict()
        self.id2ent = dict()
        self.id2rel = dict()
        self.h2t = {}
        self.t2h = {}

        self.num_anomalies = 0
        self.triples = []
        self.start_batch = 0
        self.path = path

        self.A = {}
        self.read_triples()

        self.triple_ori_set = set(self.triples)
        self.num_original_triples = len(self.triples)

        self.num_entity = self.num_ent()
        self.num_relation = self.num_rel()
        print('entity&relation: ', self.num_entity, self.num_relation)

        if not os.path.exists(str(self.path) + "/long_tail_node.txt"):
            self.preprocess_long_tail_triples()

        long_tail_node = []
        long_tail_relation = []
        node_path = str(self.path) + "/long_tail_node.txt"
        relation_path = str(self.path) + "/long_tail_relation.txt"

        if os.path.exists(node_path):
            with open(node_path) as f:
                long_tail_node = json.loads(str(f.read()))
        if os.path.exists(relation_path):
            with open(relation_path) as f1:
                long_tail_node = json.loads(str(f1.read()))

        self.num_long_tail_triple = len(long_tail_node) + len(long_tail_relation)
        self.long_tail_label = self.load_long_tail_labels()
        self.long_tail_triple_label = [(self.triples[i], self.long_tail_label[i]) for i in range(len(self.triples))]

        self.bp_triples_label = self.inject_anomaly(args)

        self.num_triples_with_anomalies = len(self.bp_triples_label)
        self.train_data, self.labels = self.get_data()
        self.triples_with_anomalies, self.triples_with_anomalies_labels, self.triples_with_long_tail_labels = self.get_data_test()

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def get_add_ent_id(self, ent):
        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        else:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent

        return ent_id

    def get_add_rel_id(self, rel):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
        return rel_id

    def init_embeddings(self, entity_file, relation_file):
        entity_emb, relation_emb = [], []

        with open(entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open(relation_file) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])

        return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

    def read_triples(self):
        print('Read begin!')
        for file in ["train", "valid", "test"]:
            with open(self.path + '/' + file + ".txt", "r") as f:
                for line in f.readlines():
                    try:
                        head, rel, tail = line.strip().split("\t")
                    except:
                        print(line)
                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)

                    self.triples.append((head_id, rel_id, tail_id))

                    self.A[(head_id, tail_id)] = rel_id
                    # self.A[head_id][tail_id] = rel_id

                    # generate h2t
                    if not head_id in self.h2t.keys():
                        self.h2t[head_id] = set()
                    temp = self.h2t[head_id]
                    temp.add(tail_id)
                    self.h2t[head_id] = temp

                    # generate t2h
                    if not tail_id in self.t2h.keys():
                        self.t2h[tail_id] = set()
                    temp = self.t2h[tail_id]
                    temp.add(head_id)
                    self.t2h[tail_id] = temp

        print("Read end!")
        return self.triples


    def preprocess_long_tail_triples(self):
        print("preprocess long tail triples")

        original_triples = self.triples
        if True:
            head = [original_triples[i][0] for i in range(len(original_triples))]
            relation = [original_triples[i][1] for i in range(len(original_triples))]
            tail = [original_triples[i][2] for i in range(len(original_triples))]
            all_node = head + tail

            long_tail_node = list(set([node for node in all_node if all_node.count(node)<=5]))
            print(len(long_tail_node))
            node_path = str(self.path) + "/long_tail_node.txt"
            if os.path.exists(node_path):
                with open(node_path, "a") as f:
                    f.write(str(long_tail_node))
            else:
                with open(node_path, "w") as f:
                    f.write(str(long_tail_node))

            long_tail_relation = list(set([re for re in relation if relation.count(re)<=5]))
            print(len(long_tail_relation))
            relation_path = str(self.path) + "/long_tail_relation.txt"
            if os.path.exists(relation_path):
                with open(node_path, "a") as f:
                    f.write(str(long_tail_node))
            else:
                with open(node_path, "w") as f:
                    f.write(str(long_tail_node))

    def load_long_tail_labels(self):
        original_triples = self.triples

        long_tail_triple_label = []
        long_tail_node = []
        long_tail_relation = []
        node_path = str(self.path) + "/long_tail_node.txt"
        relation_path = str(self.path) + "/long_tail_relation.txt"
        triple_label_path = str(self.path) + "/long_tail_label_triple.txt"

        if os.path.exists(node_path):
            with open(node_path) as f:
                long_tail_node = json.loads(str(f.read()))

        if os.path.exists(relation_path):
            with open(relation_path) as f1:
                long_tail_relation = json.loads(str(f1.read()))

        for triple in original_triples:
            label = 0
            if triple[0] in long_tail_node or triple[1] in long_tail_relation or triple[2] in long_tail_node:
                label = 1
            long_tail_triple_label.append(label)

        return long_tail_triple_label



    def rand_ent_except(self, ent):
        rand_ent = random.randint(1, self.num_ent() - 1)
        while rand_ent == ent:
            rand_ent = random.randint(1, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:
                new_head = self.rand_ent_except(head)
                neg_triples.append((new_head, rel, tail))
            else:
                new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def generate_anomalous_triples(self, pos_triples):
        neg_triples = []
        #print(pos_triples)
        #iter_triples = [pos_triples[i][0] for i in range(len(pos_triples))]
        for triple_label in pos_triples:
            head = triple_label[0][0]
            rel = triple_label[0][1]
            tail = triple_label[0][2]
            label = triple_label[1]

            head_or_tail = random.randint(0, 2)
            if head_or_tail == 0:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = rel
                new_tail = tail
                # neg_triples.append((new_head, rel, tail))
            elif head_or_tail == 1:
                new_head = head
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = tail
            else:
                # new_tail = self.rand_ent_except(tail)
                # neg_triples.append((head, rel, new_tail))
                new_head = head
                new_relation = rel
                new_tail = random.randint(0, self.num_entity - 1)
            anomaly = (new_head, new_relation, new_tail)
            while anomaly in self.triple_ori_set:
                if head_or_tail == 0:
                    new_head = random.randint(0, self.num_entity - 1)
                    new_relation = rel
                    new_tail = tail
                    # neg_triples.append((new_head, rel, tail))
                elif head_or_tail == 1:
                    new_head = head
                    new_relation = random.randint(0, self.num_relation - 1)
                    new_tail = tail
                else:
                    # new_tail = self.rand_ent_except(tail)
                    # neg_triples.append((head, rel, new_tail))
                    new_head = head
                    new_relation = rel
                    new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            triples_anomaly = [anomaly, label]
            neg_triples.append(triples_anomaly)
        return neg_triples

    def generate_anomalous_triples_2(self, num_anomaly):
        neg_triples = []
        for i in range(num_anomaly):
            new_head = random.randint(0, self.num_entity - 1)
            new_relation = random.randint(0, self.num_relation - 1)
            new_tail = random.randint(0, self.num_entity - 1)

            anomaly = (new_head, new_relation, new_tail)

            while anomaly in self.triple_ori_set:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)

            neg_triples.append(anomaly)
        return neg_triples

    def shred_triples(self, triples):
        h_dix = [triples[i][0] for i in range(len(triples))]
        r_idx = [triples[i][1] for i in range(len(triples))]
        t_idx = [triples[i][2] for i in range(len(triples))]
        return h_dix, r_idx, t_idx

    def shred_triples_and_labels(self, triples_and_labels):
        heads = [triples_and_labels[i][0][0] for i in range(len(triples_and_labels))]
        rels = [triples_and_labels[i][0][1] for i in range(len(triples_and_labels))]
        tails = [triples_and_labels[i][0][2] for i in range(len(triples_and_labels))]
        labels = [triples_and_labels[i][1] for i in range(len(triples_and_labels))]
        return heads, rels, tails, labels

    def all_triplets(self):
        ph_all, pr_all, pt_all = self.shred_triples(self.triples)
        nh_all, nr_all, nt_all = self.shred_triples(self.generate_neg_triples(self.triples))
        return ph_all, pt_all, nh_all, nt_all, pr_all

    def get_data(self):
        # bp_triples_label = self.inject_anomaly()
        bp_triples_label = self.bp_triples_label
        anomaly_labels = [bp_triples_label[i][2] for i in range(len(bp_triples_label))]
        bp_triples = [(bp_triples_label[i][0], bp_triples_label[i][1]) for i in range(len(bp_triples_label))]
        bn_triples = self.generate_anomalous_triples(bp_triples_label)
        all_triples = bp_triples + bn_triples
        all_triples_with_no_labels = [all_triples[i][0] for i in range(len(all_triples))]

        return all_triples_with_no_labels, self.toarray(anomaly_labels)

    def get_data_test(self):
        bp_triples_label = self.bp_triples_label
        anomaly_labels = [bp_triples_label[i][2] for i in range(len(bp_triples_label))]
        long_tail_labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]

        return bp_triples, anomaly_labels, long_tail_labels

    def toarray(self, x):
        return torch.from_numpy(np.array(list(x)).astype(np.int32))

    def inject_anomaly(self, args):
        print("Inject anomalies!")
        #original_triples = self.triples
        original_triples = self.long_tail_triple_label

        triple_size = len(original_triples)

        self.num_anomalies = int(args.anomaly_ratio * self.num_original_triples)
        args.num_anomaly_num = self.num_anomalies
        print("###########Inject TOP@K% Anomalies##########")
        # if self.isInjectTopK:
        #     self.num_anomalies = args.num_anomaly_num
        #     print("###########Inject TOP@K Anomalies##########")
        # else:
        #
        # idx = random.sample(range(0, self.num_original_triples - 1), num_anomalies)

        idx = random.sample(range(0, self.num_original_triples - 1), self.num_anomalies)
        selected_triples = [original_triples[idx[i]] for i in range(len(idx))]
        #anomalies = self.generate_anomalous_triples(selected_triples) + self.generate_anomalous_triples_2(self.num_anomalies // 2)
        anomalies = self.generate_anomalous_triples(selected_triples)

        triple_label = [(original_triples[i][0], original_triples[i][1], 0) for i in range(len(original_triples))]
        anomaly_label = [(anomalies[i][0], anomalies[i][1], 1) for i in range(len(anomalies))]

        triple_anomaly_label = triple_label + anomaly_label
        shuffle(triple_anomaly_label)
        return triple_anomaly_label

'''
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-ds', '--dataset', default='WN18RR', help='dataset')
args, _ = parser.parse_known_args()

parser.add_argument('--anomaly_ratio', default=0.05, type=float, help="anomaly ratio")
parser.add_argument('--num_anomaly_num', default=300, type=int, help="number of anomalies")
args = parser.parse_args()

#dataset = Reader(args, "data/WN18RR")
dataset = Reader(args, "data/FB15K")
print(dataset.train_data)
'''

