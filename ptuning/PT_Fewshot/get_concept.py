import numpy as np
import argparse
from gen_label import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='', help='ptuned model directory')
parser.add_argument('--unlabeled_path', default='', help='path for unlabeled dataset')
parser.add_argument('--task_name', default='boolre', help='boolre or multicon or sinre')
parser.add_argument('--logits_file', default='', help='save path for all logits result')
parser.add_argument('--con2id_file', default='', help='concept2id file')
parser.add_argument('--batch_size', default=8, help='')
parser.add_argument('--device', default='cuda', help='')
parser.add_argument('--gpu_num', default=1, help='')
parser.add_argument('--save_path', default='', help='path for save instances')

args = parser.parse_args()


#get the prediction result for each binary classifier
def get_merge_logits(args):

    con2id = get_rel2id(args.con2id_file)
    id2con = get_id2rel(con2id)

    metric = ["acc"]
    eval_config = EvalConfig(device=args.device,
                             n_gpu=args.gpu_num,
                             metrics=metric,
                             per_gpu_eval_batch_size=args.batch_size)

    logits_list = get_logits(args.model_dir, eval_config, con2id, args.task_name, args.unlabeled_path, args.logits_file)

    return logits_list

#get topk high quality label results, and save in file
def get_anno_res(logits_list, data_file, save_file, con2id):
    data_list = get_unlabel_data(data_file)
    outf = open(save_file, 'w', encoding='utf-8')
    id2con = get_id2rel(con2id)

    con_idx = 0
    for inst_id in range(len(data_list)):
        obj = data_list[inst_id]

        logit_list = torch.FloatTensor(logits_list[con_idx])
        _, con_id = torch.max(logit_list, 0)
        h_con = id2con[int(con_id)]
        con_idx += 1

        logit_list = torch.FloatTensor(logits_list[con_idx])
        _, con_id = torch.max(logit_list, 0)
        t_con = id2con[int(con_id)]
        con_idx += 1

        obj['h']['concept'] = h_con
        obj['t']['concept'] = t_con

        obj['text'] = " ".join(obj['token'])
        outf.write(json.dumps(obj))
        outf.write('\n')

def load_logits(logits_file):
    lof = open(logits_file, 'r', encoding='utf-8')
    logits_list = json.load(lof)
    return logits_list


if __name__ == "__main__":
    con2id = get_rel2id(args.con2id_file)
    logits_list = get_merge_logits(args)
    # logits_list = load_logits(args.logits_file)
    get_anno_res(logits_list, args.unlabeled_path, args.save_path, con2id)