import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
import pdb

class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """
    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        # Load the file, add lines to self.data
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        # fact->(e1_id, e2_id, relaiton_str)
        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32  )  # id2relation_weight: one relation corresponding to a weight
            self.bag_scope = [  ]  # id2instance_idx: each ele is a list, contain all instance idx for a fact
            self.name2id = {  }  # fact_name->fact_id
            self.bag_name = [  ]  # fact_name list
            self.facts = {}  # if the fact has appeared
            # go through each instance
            for idx, item in enumerate(self.data):
                # Annotated test set
                if 'anno_relation_list' in item:
                    for r in item['anno_relation_list']:
                        fact = (item['h']['id'], item['t']['id'], r)
                        if r != 'NA':
                            self.facts[fact] = 1
                    assert entpair_as_bag
                    name = (item['h']['id'], item['t']['id'])
                else:
                    fact = (item['h']['id'], item['t']['id'], item['relation']  )  # ('Q1331049', 'Q3056359', 'place served by transport hub')
                    if item['relation'] != 'NA':
                        self.facts[fact] = 1
                    if entpair_as_bag:
                        name = (item['h']['id'], item['t']['id'])
                    else:
                        name = fact
                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0  # the frequency of the relation
            self.weight = 1.0 / (self.weight ** 0.05)  # for a relation, more freq->less weight
            self.weight = torch.from_numpy(self.weight)
        else:
            pass

    def __len__(self):
        return len(self.bag_scope)

    # get all instance infomation for a fact(bag), index->fact index
    def __getitem__(self, index):
        bag = self.bag_scope[index]  # all instance idx for a fact
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)  # random sample bag_size instances
            else:
                resize_bag = bag + list \
                    (np.random.choice(bag, self.bag_size - len(bag)))  # random choice bag_size - len(bag) instances
            bag = resize_bag

        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']  ]  # rel_idx for the bag
        # go through each instance in bag
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item)  )# encode item
            if seqs is None:
                seqs = []  # contain three item, word_idx_list, distoe1_list, distoe2_list for all instance in the bag
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (n, L), n is the size of bag
        return [rel, self.bag_name[index], len(bag), bag] + seqs

    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (sumn, L)
            seqs[i] = seqs[i].expand \
                ((torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, ) + seqs[i].size())
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]  # count: bag_size, label:rel_idx
        seqs = data[4:]
        bag_idx = data[3]

        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0) # (batch, bag, L)
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope, bag_idx] + seqs


    def eval(self, pred_result, threshold=0.5):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True  )  # sort by score
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)  # all facts number

        entpair = {}

        for i, item in enumerate(sorted_pred_result):
            # Save entpair label and result for later calculating F1
            idtf = item['entpair'][0] + '#' + item['entpair'][1]  # e1_id#e2_id
            if idtf not in entpair:
                entpair[idtf] =  {  # init entpair[idtf]
                    'label': np.zeros((len(self.rel2id)), dtype=np.int),  # right label -> 1
                    'pred': np.zeros((len(self.rel2id)), dtype=np.int),  # labels whose score >= threshold -> 1
                    'score': np.zeros((len(self.rel2id)), dtype=np.float)  # score for each relation label
                }
            if (item['entpair'][0], item['entpair'][1],
                item['relation']) in self.facts:  # the prediction is right(因为self.facts里面是所有正确的Fact）
                correct += 1
                entpair[idtf]['label'][self.rel2id[item['relation']]] = 1
            if item['score'] >= threshold:
                entpair[idtf]['pred'][self.rel2id[item['relation']]] = 1
            entpair[idtf]['score'][self.rel2id[item['relation']]] = item['score']

            prec.append(float(correct) / float(i + 1))  # current currect rate
            rec.append(float(correct) / float(total))  # total currect rate

        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        best_threshold = sorted_pred_result[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()]['score']  # the fact which has max f1, its score
        mean_prec = np_prec.mean()

        label_vec = []
        pred_result_vec = []
        score_vec = []
        for ep in entpair:
            label_vec.append(entpair[ep]['label'])  # right label -> 1
            pred_result_vec.append(entpair[ep]['pred'])  # labels whose score >= threshold -> 1
            score_vec.append(entpair[ep]['score'])  # score for each relation label
        label_vec = np.stack(label_vec, 0)
        pred_result_vec = np.stack(pred_result_vec, 0)
        score_vec = np.stack(score_vec, 0)

        micro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))),
                                                  average='micro')
        micro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))),
                                               average='micro')
        micro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))),
                                            average='micro')

        macro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))),
                                                  average='macro')
        macro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))),
                                               average='macro')
        macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))),
                                            average='macro')

        pred_result_vec = score_vec >= best_threshold  # [instance_num, class_num], label score >= threshold->True
        max_macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))),
                                                average='macro')
        max_micro_f1_each_relation = {}
        for rel in self.rel2id:
            if rel != 'NA' and rel != 'no relation':  # f1_score for each relation
                max_micro_f1_each_relation[rel] = sklearn.metrics.f1_score(label_vec, pred_result_vec,
                                                                           labels=[self.rel2id[rel]], average='micro')
        # p@200:in top 200 facts, how many facts are predicted exactly
        return {'max_micro_f1': max_micro_f1, 'max_macro_f1': max_macro_f1,
                'auc': auc, 'p@100': np_prec[99], 'p@200': np_prec[199], 'p@300': np_prec[299],
                'avg_p300': (np_prec[99] + np_prec[199] + np_prec[299]) / 3, 'micro_f1': micro_f1, 'macro_f1': macro_f1,
                'max_micro_f1_each_relation': max_micro_f1_each_relation}


def BagRELoader(path, rel2id, tokenizer, batch_size,
        shuffle, entpair_as_bag=False, bag_size=0, num_workers=0,
        collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

