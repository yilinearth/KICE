import json

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
    if valid_num == 0:
        valid_num = 0.0001
    total_pre = total_pre / valid_num
    total_recall = total_recall / valid_num
    total_f1 = total_f1 / valid_num

    average_pre = float(correct_count) / float(total_count)
    return total_pre, total_recall, total_f1, average_pre



def eval_anno_list(anno_list, rel2idx, pr_file):
    pf = open(pr_file, 'w', encoding='utf-8')
    for i in range(1, int(len(anno_list) / 50) + 1):
        topk = i * 50
        pre, recall, f1, avg_pre = compute_pr(anno_list, rel2idx, topk)
        pr_str = "topk={}, pre:{}, recall:{}, f1:{}, avg_pre:{}".format(topk, pre, recall, f1, avg_pre)
        pf.write(pr_str)
        pf.write('\n')
        print(pr_str)

def get_unlabel_data(unlabel_file):
    unf = open(unlabel_file, 'r', encoding='utf-8')
    un_list = []
    for line in unf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        un_list.append(obj)
    return un_list

def get_id2rel(rel2id):
    id2rel = {}
    for rel in rel2id:
        id2rel[rel2id[rel]] = rel
    return id2rel

def get_rel2id(rel2id_file):
    relf = open(rel2id_file, 'r', encoding='utf-8')
    rel2id = json.load(relf)
    return rel2id

def dump_data(data_list, out_file):
    outf = open(out_file, 'w')
    for data in data_list:
        outf.write(json.dumps(data))
        outf.write('\n')
    outf.close()