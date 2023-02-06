import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader, BagRELoader
from .utils import AverageMeter, ContrastiveLoss
from tqdm import tqdm
import sklearn.metrics as cls_metric
from .focal_loss import FocalLoss

import numpy as np
import os
import copy
import pdb
from torch.optim.lr_scheduler import StepLR

class BagRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt,
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd',
                 bag_size=0,
                 loss_weight=False,
                 old_state_dict=None,
                 args=None,
                 is_focal_loss=False,
                 size_average=True,
                 gamma=2):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        self.args = args
        # Load data
        if train_path != None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                bag_size=bag_size,
                entpair_as_bag=False)

        if val_path != None:
            self.val_loader = BagRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=True)
        
        if test_path != None:
            self.test_loader = BagRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=True
            )
        # Model
        # self.model = nn.DataParallel(model)
        self.model = model
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.load_state_dict(torch.load(old_state_dict)['state_dict'])
        self.model.load_state_dict(torch.load(old_state_dict)['state_dict'])
        self.device = args.device
        # Criterion
        if is_focal_loss:
            alpha = self.train_loader.dataset.weight
            self.criterion = FocalLoss(len(model.rel2id), alpha=alpha, gamma=gamma, size_average=size_average)
        elif loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")

        # self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.8)

        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='auc', soft=True):
        best_metric = 0

        self_training_loss = nn.KLDivLoss(reduction = 'none') if soft else nn.CrossEntropyLoss(reduction = 'none')

        for epoch in range(self.max_epoch):
            # Train
            self.model.train()#######
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():  # 关掉老师模型的梯度
                p.requires_grad = False
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            # avg_pos_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(label, scope, *args, bag_size=self.bag_size)
                pusedo_logits = self.teacher_model(label, scope, *args, bag_size=self.bag_size)
                loss = self.criterion(logits, label)

                if self.args.calc_loss_weight > 0:
                    loss += self.args.calc_loss_weight * self.calc_loss(input=torch.log(torch.softmax(logits, -1)),
                                        target=pusedo_logits,
                                        loss=self_training_loss,
                                        thresh=self.args.self_training_eps,
                                        soft=True,
                                        conf='entropy',
                                        confreg=self.args.self_training_confreg)


                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                # pos_total = (label != 0).long().sum()
                # pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                # if pos_total > 0:
                #     pos_acc = float(pos_correct) / float(pos_total)
                # else:
                #     pos_acc = 0

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                # avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                
                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Val 
            print("=== Epoch %d val ===" % epoch)
            ######
            result = self.wrench_eval_model(self.val_loader)
            print("wrench_Micro F1: %.4f" % (result['mic_f1']))
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                # torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_metric = result[metric]
        print("Best %s on val set: %f" % (metric, best_metric))


    def calc_loss(self, input, target, loss, thresh=0.95, soft=True, conf='max', confreg=0.1):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)  # [batch size, class num]

        if conf == 'max':
            weight = torch.max(target, axis=1).values
            w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(self.device)
        elif conf == 'entropy':  # 选出来这个batch里熵值大于阈值的实例（w=1)否则w=0
            weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)  # [batch size]
            weight = 1 - weight / np.log(weight.size(-1)+1)
            w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(self.device)
        target = self.soft_frequency(target, probs=True, soft=soft)  # 做了一些变换

        loss_batch = loss(input, target)  # 计算KL散度, [batch_size, class_num]

        l = torch.sum(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))  # 只考虑在teacher模型预测中熵值大于阈值的数据，且用熵值作为权重

        n_classes_ = input.shape[-1]
        l -= confreg * (torch.sum(input * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_)  # l减去了什么东西
        return l

    def soft_frequency(self, logits,  probs=False, soft = True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = self.args.self_training_power
        if not probs:
            softmax = nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=0)
        t = y**power / f
        #print('t', t)
        t = t + 1e-10
        p = t/torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)

    def contrastive_loss(self, input, feat, target, conf='none', thresh=0.1, distmetric='l2'):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        if conf == 'max':
            weight = torch.max(target, axis=1).values
            w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(self.device)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)
            weight = 1 - weight / np.log(weight.size(-1))
            w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(self.device)
        input_x = input[w]

        feat_x = feat[w]
        batch_size = input_x.size()[0]
        if batch_size == 0:
            return 0
        index = torch.randperm(batch_size).to(self.device)  # 随机打乱数据
        input_y = input_x[index, :]  # 找到这个数据在input 和 feat中的vector
        feat_y = feat_x[index, :]
        argmax_x = torch.argmax(input_x, dim=1)  # 没打乱的数据在input中最可能的类别
        argmax_y = torch.argmax(input_y, dim=1)  # 打乱的数据在input中最可能的类别
        agreement = torch.FloatTensor([1 if x == True else 0 for x in argmax_x == argmax_y]).to(self.device)

        criterion = ContrastiveLoss(margin=1.0, metric=distmetric)
        loss, dist_sq, dist = criterion(feat_x, feat_y, agreement)  # 让具有同个标签的不同数据具有相似的表达

        return loss


    def wrench_eval_model(self, eval_loader):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            t = tqdm(eval_loader)

            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(None, scope, *args, train=False, bag_size=self.bag_size) # results after softmax
                logits = logits.cpu().numpy()

                for i in range(len(logits)):
                    max_score, max_rel_id = torch.max(torch.FloatTensor(logits[i]), 0)
                    # if self.model.id2rel[int(max_rel_id)] != 'no relation':
                        # ent_pair = bag_name[i][:2]
                    y_pred.append(int(max_rel_id))
                    y_true.append(int(label[i]))


            #####
            acc = cls_metric.accuracy_score(y_true, y_pred)
            mic_prec = cls_metric.precision_score(y_true, y_pred, average='micro')
            mic_f1 = cls_metric.f1_score(y_true, y_pred, average='micro')
            mic_rec = cls_metric.recall_score(y_true, y_pred, average='micro')

            mac_f1 = cls_metric.f1_score(y_true, y_pred, average='macro')
            mac_prec = cls_metric.precision_score(y_true, y_pred, average='macro')
            mac_rec = cls_metric.recall_score(y_true, y_pred, average='macro')
            result = {'acc':acc, 'mic_prec':mic_prec, 'mic_f1':mic_f1, 'mic_rec': mic_rec,
                      'mac_f1': mac_f1, 'mac_prec': mac_prec, 'mac_rec': mac_rec}

        return result

    def get_nero_val2res(self, eval_loader):
        self.model.eval()
        logits_list = []
        label_list = []
        val_list = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]
        with torch.no_grad():
            t = tqdm(eval_loader)

            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(None, scope, *args, train=False, bag_size=self.bag_size)  # results after softmax
                logits = logits.cpu().numpy()
                label = label.cpu()
                logits_list += logits.tolist()
                label_list += label.tolist()

        val2res = {}
        for val_threshold in val_list:
            correct_by_relation = 0.0
            guessed_by_relation = 0.0
            gold_by_relation = 0.0
            for i in range(len(logits_list)):
                logit = torch.FloatTensor(logits_list[i])
                max_score, max_rel_id = torch.max(torch.FloatTensor(logit), 0)
                guess = max_rel_id
                gold = label_list[i]
                val = torch.sum(logit * -torch.log(logit))
                if val >= val_threshold:
                    guess = self.model.rel2id['no relation']

                if guess == self.model.rel2id['no relation'] and gold == self.model.rel2id['no relation']:
                    continue
                elif guess == self.model.rel2id['no relation'] and gold != self.model.rel2id['no relation']:
                    gold_by_relation += 1
                elif guess != self.model.rel2id['no relation'] and gold == self.model.rel2id['no relation']:
                    guessed_by_relation += 1
                else:
                    guessed_by_relation += 1
                    gold_by_relation += 1
                    if guess == gold:
                        correct_by_relation += 1

            prec_micro = 0.0
            if guessed_by_relation > 0:
                prec_micro = float(correct_by_relation) / float(guessed_by_relation)
            recall_micro = 0.0
            if gold_by_relation > 0:
                recall_micro = float(correct_by_relation) / float(gold_by_relation)
            f1_micro = 0.0
            if prec_micro + recall_micro > 0.0:
                f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
            result = {'prec_micro': prec_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro}
            val2res[val_threshold] = result
            print("{} - {}".format(val_threshold, result))
        return val2res



    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        # self.model.module.load_state_dict(state_dict)

