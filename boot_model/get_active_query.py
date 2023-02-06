import pdb
import json
import pred_opennre
from pred_opennre import framework
import argparse
import torch
from tqdm import tqdm
from act_dataloader import *
import os
sys.path.append('..')
from data.data_utils import *
import scipy.stats

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', help='Checkpoint name')
parser.add_argument('--rel2id_path', default='', type=str, help='Relation to ID file')
parser.add_argument('--max_length', default=128, type=int, help='Maximum sentence length')
parser.add_argument('--mask_entity', action='store_true', help='Mask entity mentions')
parser.add_argument('--pretrain_path', default='bert-base-uncased', help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--unlabel_rule_path', default='', type=str, help='unlabel rules path')
parser.add_argument('--boot_rule_path', default='', type=str, help='current rules set file')
parser.add_argument('--conf_rule_path', default='', type=str, help='confuse rules path')
parser.add_argument('--mask_num', default=3, type=int, help='number of masks')
parser.add_argument('--query_path', default='', help='path for most confused querys')
parser.add_argument('--query_boot_path', default='', help='path for wrong predicted bootstrap rules')
parser.add_argument('--rest_path', default='', help='rest data path')
parser.add_argument('--topk', default=200, type=int, help='topk most confused data')
parser.add_argument('--threshold', default=0.97, type=float, help='similar threshold for confuse rules')
parser.add_argument('--mask_weight', default=[0.3, 0.4, 0.3], type=list, help='mask weights')
parser.add_argument('--isSelfFix', action='store_true', help='if get fix relation from real rel')
parser.add_argument('--query_rule_num', type=int, default=40, help='max number of wrong labeled boot rules to label')

parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--max_epoch', default=3, type=int, help='Max number of training epochs')
parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate')
parser.add_argument('--bag_size', type=int, default=1, help='Fixed bag size. If set to 0, use original bag sizes')
parser.add_argument('--cpu_num', type=int, default=3, help='cpu number for parallel')


args = parser.parse_args()

def get_prediction(my_framework, bag_size):

    rules_logits, conf_logits, boot_preds = [], [], []

    my_framework.model.eval()
    with torch.no_grad():
        t = tqdm(my_framework.val_loader)
        for iter, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            args = data[2:]

            logits = my_framework.model(*args)
            logits = logits.cpu()

            rules_logits.extend(logits)

    with torch.no_grad():
        t = tqdm(my_framework.test_loader)
        for iter, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            args = data[2:]

            logits = my_framework.model(*args)
            logits = logits.cpu()

            conf_logits.extend(logits)

    with torch.no_grad():
        t = tqdm(my_framework.train_loader)
        for iter, data in enumerate(t):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            args = data[2:]

            logits = my_framework.model(*args)

            score, pred = logits.max(-1)
            boot_preds.extend(pred)

    conf_logits = torch.stack(conf_logits).squeeze(1)
    rules_logits = torch.stack(rules_logits).squeeze(1)

    conf_logits = torch.softmax(conf_logits, dim=1)
    rules_logits = torch.softmax(rules_logits, dim=1)

    return conf_logits, rules_logits, boot_preds

 # scipy.stats.entropy(torch.cat(conf_logits.cpu(), 0), torch.cat(rules_logits.cpu(), 0), axis=1)

def get_framework():
    sentence_encoder = pred_opennre.encoder.BERTEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
    model = pred_opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    my_framework = framework.SentenceRE(
        train_path=args.boot_rule_path,
        val_path=args.unlabel_rule_path,
        test_path=args.conf_rule_path,
        model=model,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        opt="adamw"
    )
    my_framework.load_state_dict(torch.load(args.ckpt)['state_dict'])
    return my_framework


def write_unlabel_query(data_list, sorted_data_idx, query_path, rest_path, topk, isSelfFix):
    qf = open(query_path, 'w', encoding='utf-8')
    ref = open(rest_path, 'w', encoding='utf-8')

    for i in range(len(sorted_data_idx)):
        data_id = sorted_data_idx[i]
        obj = data_list[data_id]
        # obj = simple_rule_obj(obj)
        if i <= topk:
            if isSelfFix:
                if 'real_rel' in obj:
                    obj['relation'] = obj['real_rel']
            qf.write(json.dumps(obj))
            qf.write('\n')
        else:
            ref.write(json.dumps(obj))
            ref.write('\n')
    qf.close()
    ref.close()

def write_boot_query(boot_rules, boot_preds, query_boot_path, rel2id, isSelfFix, query_rule_num):
    qbf = open(query_boot_path, 'w', encoding='utf-8')
    wrong_num = 0
    for i in range(len(boot_preds)):

        pred_id = boot_preds[i]
        rule_rel_id = rel2id[boot_rules[i]['relation']]
        if pred_id != rule_rel_id:
            if wrong_num <= query_rule_num and 'is_act' not in boot_rules[i]:
                obj = boot_rules[i]
                # obj = simple_rule_obj(boot_rules[i])
                if isSelfFix:
                    if 'real_rel' in obj:
                        obj['relation'] = obj['real_rel']
                qbf.write(json.dumps(obj))
                qbf.write('\n')
            wrong_num += 1

    qbf.close()



if __name__ == '__main__':
    rel2id = json.load(open(args.rel2id_path))
    # get confuse rules
    unlabel_rules_data = get_data_list(args.unlabel_rule_path)
    get_confuse_data_parallel(unlabel_rules_data, args.conf_rule_path, args.mask_num, args.mask_weight, args.threshold, args.cpu_num)
    # get boot rules
    boot_rules = get_data_list(args.boot_rule_path)

    #get framework
    my_framework = get_framework()

    #get logits and predictions

    conf_logits, rules_logits, boot_preds = get_prediction(my_framework, args.bag_size)

    KL_res = scipy.stats.entropy(rules_logits, conf_logits, axis=1)

    #sorted data as KL
    _, sorted_data_idxs = torch.sort(torch.FloatTensor(KL_res), descending=True)


    # write results
    write_unlabel_query(unlabel_rules_data, sorted_data_idxs, args.query_path, args.rest_path, args.topk, args.isSelfFix)
    write_boot_query(boot_rules, boot_preds, args.query_boot_path, rel2id, args.isSelfFix, args.query_rule_num)





