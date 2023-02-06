import json
import random
from random import sample
import os
import pdb

def trans_obj(org_obj, target_rel):
    isPos = False

    if org_obj["relation"] == target_rel:
        org_obj["label"] = True
        isPos = True
    else:
        org_obj["label"] = False
    org_obj["target_rel"] = target_rel

    return org_obj, isPos

def trans_file(org_file, new_file, target_rel, sample_num):
    orf = open(org_file, 'r', encoding='utf-8')
    nef = open(new_file, 'w', encoding='utf-8')
    pos_list = []
    neg_list = []
    pos_samp_num = sample_num
    neg_samp_num = sample_num
    for line in orf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        obj, isPos = trans_obj(obj, target_rel)
        if isPos:
            pos_list.append(obj)
        else:
            neg_list.append(obj)

    if len(pos_list) < pos_samp_num:
        pos_samp_num = len(pos_list)
    ran_pos_list = sample(pos_list, pos_samp_num)

    if len(neg_list) < neg_samp_num:
        neg_samp_num = len(neg_list)
    ran_neg_list = sample(neg_list, neg_samp_num)

    print("Output {} pos and {} neg for file {} in relation {}.".format(len(ran_pos_list), len(ran_neg_list), new_file, target_rel))

    for obj in ran_pos_list:
        nef.write(json.dumps(obj))
        nef.write("\n")

    for obj in ran_neg_list:
        nef.write(json.dumps(obj))
        nef.write("\n")

    nef.close()
    return sample_num

def get_rel2id(train_file, rel2id_file):
    trf = open(train_file, 'r', encoding='utf-8')
    rel2id = {}
    idx = 0
    for line in trf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        rel_str = obj['relation']
        if rel_str not in rel2id:
            rel2id[rel_str] = idx
            idx += 1
    trf.close()
    ref = open(rel2id_file, 'w', encoding='utf-8')
    json.dump(rel2id, ref)
    return rel2id

if __name__ == '__main__':
    home = ""
    dataset="tacred"
    org_train_file = home + "step0/RE_model_data/train.json"
    org_test_file = home + dataset + "/my_test.json"
    org_val_file = home + "step0/RE_model_data/dev.json"
    rel2id_file = home + dataset + "/my_rel2id.json"

    relf = open(rel2id_file,'r', encoding='utf-8')
    rel2id = json.load(relf)

    train_num = 86
    test_num = 30000
    val_num = 86

    for rel in rel2id:
        if rel != "no relation":
            continue
        target_rel = rel
        target_dir = home + "step0/ptuning/bool_cls"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        tar_train_file = os.path.join(target_dir, "train.txt")
        tar_test_file = os.path.join(target_dir, "test.txt")
        tar_val_file = os.path.join(target_dir, "dev.txt")


        samp_num_train = trans_file(org_train_file, tar_train_file, target_rel, train_num)
        samp_num_test = trans_file(org_test_file, tar_test_file, target_rel, test_num)
        samp_num_val = trans_file(org_val_file, tar_val_file, target_rel, val_num)

        print("relation: {} sample num is {}-{}-{}".format(rel2id[rel], samp_num_train, samp_num_test, samp_num_val))