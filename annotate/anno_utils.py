import json
import torch

def compute_pr(anno_list, rel2idx, topk):
    rel2pr = {}
    rel2pr_val = {}
    total_count = 0
    correct_count = 0
    for rel_str in rel2idx:
        rel2pr[rel_str] = []
        rel2pr_val[rel_str] = []
        for i in range(4):
            rel2pr[rel_str].append(0.0)  # TP, FP, TN, FN

    length = topk if topk < len(anno_list) else len(anno_list)
    for i in range(length):
        if "real_relation" in anno_list[i]:
            rel_str = anno_list[i]['real_relation']
        else:
            rel_str = anno_list[i]['real_rel']
        if anno_list[i]['relation'] == rel_str:
            rel2pr[rel_str][0] += 1
            correct_count += 1
        else:
            rel2pr[rel_str][3] += 1
        for key in rel2pr:
            if key == rel_str: continue
            if anno_list[i]['relation'] != key:
                rel2pr[key][2] += 1
            else:
                rel2pr[key][1] += 1
        total_count += 1

    total_pre, total_recall, total_f1, valid_num = 0.0, 0.0, 0.0, 0.0
    for key in rel2pr:
        if (rel2pr[key][0] == 0 and rel2pr[key][3] == 0) or (rel2pr[key][0] == 0 and rel2pr[key][1] == 0):
            continue
        precision = float(rel2pr[key][0]) / float(rel2pr[key][0] + rel2pr[key][1])
        recall = float(rel2pr[key][0]) / float(rel2pr[key][0] + rel2pr[key][3])
        # print('relation:{} pre:{} recall:{}'.format(key, precision, recall))
        if precision == 0 and recall == 0:
            continue
        f1 = 2*(precision * recall) / (precision + recall)

        total_pre += precision
        total_recall += recall
        total_f1 += f1
        valid_num += 1

    total_pre = total_pre / valid_num
    total_recall = total_recall / valid_num
    total_f1 = total_f1 / valid_num

    average_pre = float(correct_count) / float(total_count)
    return total_pre, total_recall, total_f1, average_pre

def eval_anno_list(anno_list, rel2idx, pr_path):
    pw = open(pr_path, 'w', encoding='utf-8')
    for i in range(1, int(len(anno_list) / 100) + 1):
        topk = i * 100
        pre, recall, f1, avg_pre = compute_pr(anno_list, rel2idx, topk)
        pr_str = "topk={}, pre:{}, recall:{}, f1:{}, avg_pre:{}".format(topk, pre, recall, f1, avg_pre)
        pw.write(pr_str)
        pw.write('\n')
        print(pr_str)

def get_rel2id(rel2id_file):
    relf = open(rel2id_file, 'r', encoding='utf-8')
    rel2id = json.load(relf)
    return rel2id

def get_rule_list(rule_path):
    ruf = open(rule_path, 'r', encoding='utf-8')
    rule_list = []
    for line in ruf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        rule_list.append(obj)
    return rule_list

def cal_cos(emb1, emb2, sim_function):
    emb1 = torch.FloatTensor(emb1)
    emb2 = torch.FloatTensor(emb2)
    cos_score = sim_function(emb1.unsqueeze(0), emb2.unsqueeze(0))
    return cos_score

def get_rel2weight(sort_rel2votes, top_rel_num, rel2id):
    id2weight = []
    for i in range(len(rel2id)):
        id2weight.append(0.0)
    for i in range(len(sort_rel2votes)):
        if i < top_rel_num:
            rel = sort_rel2votes[i][0]
            id2weight[rel2id[rel]] = sort_rel2votes[i][1]
    id2weight = torch.FloatTensor(id2weight)
    id2weight = torch.softmax(id2weight, dim=0)
    id2weight = id2weight.tolist()
    return id2weight