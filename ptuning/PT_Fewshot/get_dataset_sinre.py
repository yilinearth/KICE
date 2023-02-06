#merge all binery classifier and generate high quality dataset and confuse dataset
import argparse
from label_utils import *
from gen_logits_multire import *
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='', help='ptuned model directory')
parser.add_argument('--unlabeled_path', default='', help='path for unlabeled dataset')
parser.add_argument('--task_name', default='boolre', help='boolre or multicon or sinre')
parser.add_argument('--words_file', default='', help='save path for all predicted result')
parser.add_argument('--rel2id_file', default='', help='rel2id file')
parser.add_argument('--batch_size', default=8, help='')
parser.add_argument('--device', default='cuda', help='')
parser.add_argument('--gpu_num', default=1, help='')
parser.add_argument('--save_path', default='', help='path for save instances')


args = parser.parse_args()

#get the prediction result for each binary classifier
def get_merge_logits(args):

    rel2id = get_rel2id(args.rel2id_file)

    metric = ["acc", "f1-macro"]
    eval_config = EvalConfig(device=args.device,
                             n_gpu=args.gpu_num,
                             metrics=metric,
                             per_gpu_eval_batch_size=args.batch_size)

    words_list = get_logits(args.model_dir, eval_config, rel2id, args.task_name, args.unlabeled_path, args.words_file)
    words_list = words_list
    return words_list


#get topk high quality label results, and save in file
def get_anno_res(words_list, data_file, save_file):
    data_list = get_unlabel_data(data_file)
    outf = open(save_file, 'w', encoding='utf-8')
    words_list = words_list[0]
    for inst_id in range(len(data_list)):
        word_list = words_list[inst_id]
        obj = data_list[inst_id]

        obj['mask_1'] = word_list
        obj['mask_0'] = [obj['h']['concept']]
        obj['mask_2'] = [obj['t']['concept']]

        obj['text'] = " ".join(obj['token'])
        outf.write(json.dumps(obj))
        outf.write('\n')

def load_logits(words_file):
    words_list = json.load(words_file)
    return words_list


if __name__ == "__main__":
    words_list = get_merge_logits(args)
    get_anno_res(words_list, args.unlabeled_path, args.save_path)






