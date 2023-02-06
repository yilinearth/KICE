import json
from gen_logits import *
import torch
from label_utils import *
import numpy as np
import pdb
INF = 100

def load_logits(rel2id, logits_file, inst_num):
    lof = open(logits_file, 'r', encoding='utf-8')

    logits_list = torch.zeros(len(rel2id), inst_num, 2)
    logits_list[:, :] = torch.tensor([INF, -INF])


    for line in lof:
        line = line.strip()
        if len(line) == 0:
            continue
        logit_obj = json.loads(line)
        rel_str = logit_obj["rel"]
        rel_id = rel2id[rel_str]
        logits_list[rel_id] = torch.FloatTensor(logit_obj["logits"])

    return logits_list



def get_label(logits_list, id2rel):
    # logits_list = torch.Tensor(logits_list)
    logits_list = torch.softmax(logits_list, dim=2)
    logits_res = logits_list[:,:,1]#[rel_num, inst_num]
    logits_res = torch.squeeze(logits_res)

    logits_res = torch.max(logits_res, dim=0)
    label_list = []
    conf_list = []

    for i in range(len(logits_res[0])):
        label_id = int(logits_res[1][i])
        conf = float(logits_res[0][i])

        label_list.append(id2rel[label_id])
        conf_list.append(conf)

    return label_list, conf_list

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

def get_anno_list(label_list, conf_list, un_list):
    idx = 0
    anno_list = []

    for obj in un_list:
        if 'relation' in obj:
            real_rel = obj['relation']
            obj['real_rel'] = real_rel

        obj['relation'] = label_list[idx]
        obj['confidence'] = conf_list[idx]
        anno_list.append(obj)

        idx += 1

    anno_list.sort(key=lambda item: item['confidence'], reverse=True)

    return anno_list

def dump_anno(anno_file, rest_file, anno_list, save_num):
    anf = open(anno_file, 'w', encoding='utf-8')
    ref = open(rest_file, 'w', encoding='utf-8')
    idx = 0
    for obj in anno_list:
        if idx >= save_num:
            ref.write(json.dumps(obj))
            ref.write('\n')
        else:
            anf.write(json.dumps(obj))
            anf.write('\n')
    ref.close()
    anf.close()



# if __name__ == "__main__":
#     # logits_file = "FewGLUE_32dev/BoolRE/step0/unlabel_logits.txt"
#     logits_file = "FewGLUE_32dev/BoolRE/tacred4_step0/test_logits.txt"
#     # unlabel_file = "FewGLUE_32dev/BoolRE/step0/unlabel.txt"
#     unlabel_file = "FewGLUE_32dev/BoolRE/tacred/test.json"
#     pr_file = "FewGLUE_32dev/BoolRE/tacred4_step0/pr_result.txt"
#     anno_file = "FewGLUE_32dev/BoolRE/tacred4_step0/anno_data.txt"
#     rest_file = "FewGLUE_32dev/BoolRE/tacred4_step0/rest_data.txt"
#     save_num = 3000
#
#     pdb.set_trace()
#     rel2id_file = "FewGLUE_32dev/BoolRE/tacred/rel2id.json"
#     relf = open(rel2id_file, 'r', encoding='utf-8')
#     rel2id = json.load(relf)
#     id2rel = get_id2rel(rel2id)
#
#     un_list = get_unlabel_data(unlabel_file)
#
#     logits_list = load_logits(rel2id, logits_file, len(un_list))
#     label_list, conf_list = get_label(logits_list, id2rel)
#     anno_list = get_anno_list(label_list, conf_list, un_list)
#
#     eval_anno_list(anno_list, rel2id, pr_file)
#
#     dump_anno(anno_file, rest_file, anno_list, save_num)








