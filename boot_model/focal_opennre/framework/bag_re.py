import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader, BagRELoader
from .utils import AverageMeter
from tqdm import tqdm
import sklearn.metrics as cls_metric
import os
from .focal_loss import FocalLoss

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
                 is_focal_loss=False,
                 size_average=True,
                 gamma=2):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
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
        ######
        # self.model = nn.DataParallel(model)
        self.model = model
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
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='auc'):
        best_metric = 0
        for epoch in range(self.max_epoch):
            # Train
            self.train()
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
                loss = self.criterion(logits, label)
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
            print("Micro F1: %.4f" % (result['mic_f1']))
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                # torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_metric = result[metric]
        print("Best %s on val set: %f" % (metric, best_metric))


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
            result = {'acc':acc, 'mic_prec':mic_prec, 'mic_f1': mic_f1, 'mic_rec': mic_rec,
                      'mac_f1': mac_f1, 'mac_prec': mac_prec, 'mac_rec': mac_rec}

        return result



    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        # self.model.module.load_state_dict(state_dict)

