import sys, json
import torch
import os
import numpy as np
import pred_opennre
import pred_opennre.framework
import argparse
import logging
import random
import time
import pdb
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', help='Checkpoint name')
parser.add_argument('--rel2id_path', default='', type=str, help='Relation to ID file')
parser.add_argument('--max_length', default=128, type=int, help='Maximum sentence length')
parser.add_argument('--mask_entity', action='store_true', help='Mask entity mentions')
parser.add_argument('--pretrain_path', default='bert-base-uncased', help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--pred_test_file', default='', type=str, help='unlabeled data file')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--max_epoch', default=3, type=int, help='Max number of training epochs')
parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate')
parser.add_argument('--bag_size', type=int, default=1, help='Fixed bag size. If set to 0, use original bag sizes')
parser.add_argument('--topk_per_relation', default=50, type=int, help='topk high confidence inst per relation')
parser.add_argument('--result_path', default='', help='topk prediction result path')
parser.add_argument('--rest_path', default='', help='rest data path')



args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_prediction(framework, bag_size, batch_size):
    dataset = framework.test_loader.dataset.data
    rel2dataset = {}
    rel2idx = {}
    total_num = 0
    framework.model.eval()
    with torch.no_grad():
        t = tqdm(framework.test_loader)
        for iter, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            label = data[0]
            indexs = data[1]
            args = data[2:]
            logits = model(*args)
            score, pred = logits.max(-1)

            for idx in range(len(indexs)):

                rel = model.id2rel[int(pred[idx])]
                confidence = float(score[idx])
                instance_idx = int(indexs[idx])


                #save the real relation of data obj
                if 'real_rel' not in dataset[instance_idx]:
                    dataset[instance_idx]['real_rel'] = dataset[instance_idx]['relation']
                    # print("there is a inst with no real rel!")

                # save predicted relation and confidence to data
                dataset[instance_idx]['relation'] = rel
                dataset[instance_idx]['confidence'] = confidence

                #save data obj to rel2dataset
                if rel not in rel2dataset:
                    rel2dataset[rel] = []
                    rel2idx[rel] = []

                #filter out repeat instance by rel2idx
                if instance_idx not in rel2idx[rel]:
                    rel2dataset[rel].append(dataset[instance_idx])
                    rel2idx[rel].append(instance_idx)
                    total_num += 1

    assert total_num == len(dataset)

    return rel2dataset

#for each relation, save topk prediction results with highest confidence
#return topk results and rest results
def get_topk_prediction(rel2data, topk):
    data_list = []
    rest_list = []

    for rel in rel2data:
        for i in range(len(rel2data[rel])):
            if 'confidence' not in rel2data[rel][i]:
                rel2data[rel][i]['confidence'] = -1

        rel2data[rel].sort(key=lambda item: item['confidence'], reverse=True)

        if topk < len(rel2data[rel]):
            data_list.extend(rel2data[rel][:topk])
            rest_list.extend(rel2data[rel][topk:])
        else:
            data_list.extend(rel2data[rel])
        print(len(rel2data[rel]))

    return data_list, rest_list


if __name__ == '__main__':
    rel2id = json.load(open(args.rel2id_path))
    #load framework
    sentence_encoder = pred_opennre.encoder.BERTEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
    ######
    model = pred_opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    framework = pred_opennre.framework.SentenceRE(
        train_path=None,
        val_path=None,
        test_path=args.pred_test_file,
        model=model,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        opt="adamw"
    )
    framework.load_state_dict(torch.load(args.ckpt)['state_dict'])


    #prediction
    rel2data = get_prediction(framework, args.bag_size, args.batch_size)
    data_list, rest_list = get_topk_prediction(rel2data, args.topk_per_relation)

    # write results
    iw = open(args.result_path, 'w', encoding='utf-8')
    for obj in data_list:
        iw.write(json.dumps(obj))
        iw.write('\n')

    rw = open(args.rest_path, 'w', encoding='utf-8')
    for obj in rest_list:
        # give correct relation back to no use instances in this step
        obj['relation'] = obj['real_rel']
        rw.write(json.dumps(obj))
        rw.write('\n')







