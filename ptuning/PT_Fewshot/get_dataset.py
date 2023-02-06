#merge all binery classifier and generate high quality dataset and confuse dataset
import argparse
from label_utils import *
from gen_logits import *
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='', help='ptuned model directory')
parser.add_argument('--unlabeled_path', default='', help='path for unlabeled dataset')
parser.add_argument('--task_name', default='boolre', help='boolre or multicon or sinre')
parser.add_argument('--logits_file', default='', help='save path for logits result')
parser.add_argument('--rel2id_file', default='', help='rel2id file')
parser.add_argument('--batch_size', default=8, help='')
parser.add_argument('--device', default='cpu', help='')
parser.add_argument('--gpu_num', default=1, help='')
parser.add_argument('--save_num', default=10, type=int, help='the number of saving no relation data')
parser.add_argument('--save_path', default='', help='path for saving instances')

parser.add_argument('--type', default='save_fix', help='save fix data (save_fix) or save only valid data (save_valid)')
parser.add_argument('--target_rel', default='no relation', help='the target relation for binary classification')


args = parser.parse_args()

def get_single_logits(args, target_rel):
    metric = ["acc", "f1"]
    rel2id = get_rel2id(args.rel2id_file)
    target_relid = rel2id[target_rel]
    eval_config = EvalConfig(device=args.device,
                             n_gpu=args.gpu_num,
                             metrics=metric,
                             per_gpu_eval_batch_size=args.batch_size)
    logit_list = get_sin_logit(args.model_dir, eval_config, target_relid, target_rel,  args.task_name, args.unlabeled_path, args.logits_file)
    logits_list = torch.FloatTensor(logit_list)
    return logits_list

#do softmax on logits,
def deal_logits(logits_list):
    logits_list = torch.softmax(logits_list, dim=1)
    logits_res = logits_list[:, 1] #the probability that inst belongs to target rel (no relation)
    logits_res = torch.squeeze(logits_res)

    return logits_res


#compute label confidence, the difference between max and second max prob for each instance
def distin_logits(logits_res):
    inst_num = len(logits_res[0])
    label_list = []
    for i in range(inst_num):
        logit_list = logits_res[:,i].tolist()

        #sort logits
        sort_idx = np.argsort(logit_list)

        #get max logit and relation
        label_id = sort_idx[-1]
        diff = logit_list[sort_idx[-1]] - logit_list[sort_idx[-2]]

        label_obj = {}
        label_obj['inst_id'] = i
        label_obj['label'] = label_id
        label_obj['diff'] = diff
        label_obj['confidence'] = logit_list[sort_idx[-1]] + diff
        label_list.append(label_obj)
    return label_list

def get_valid_list(logits_res):
    inst_num = len(logits_res)
    label_list = []
    for i in range(inst_num):
        label_obj = {}
        label_obj['inst_id'] = i
        label_obj['valid_conf'] = 1 - logits_res[i] #the probability that the data not belong to na
        label_list.append(label_obj)

    return label_list


#get the data list for invalid data
def get_invalid_data_list(valid_list, data_file, save_num, target_rel):
    valid_list.sort(key=lambda item: item['valid_conf'], reverse=True)

    data_list = get_unlabel_data(data_file)
    correct = 0.0
    total_valid = 0.0
    invalid_list = []
    invalid_id_list = []
    save_valid_list = []
    for i in range(len(valid_list)):
        inst_id = valid_list[i]['inst_id']
        obj = data_list[inst_id]
        if 'real_rel' in obj:
            real_rel = obj['real_rel']
        else:
            real_rel = obj['relation']

        if i <= save_num:
            if real_rel != target_rel:#real relation is not 'no relation'
                correct += 1
                total_valid += 1
            save_valid_list.append(obj)
        else:
            if real_rel != target_rel:
                total_valid += 1

            invalid_list.append(obj)
            invalid_id_list.append(inst_id)


    print("correct : {} - total_valid : {}".format(correct, total_valid))
    print("correct rate is {}".format(correct / total_valid))
    return invalid_list, save_valid_list, invalid_id_list

def save_fixed_file(invalid_id_list, data_file, save_file, target_rel):
    data_list = get_unlabel_data(data_file)
    saf = open(save_file, 'w', encoding='utf-8')
    correct = 0
    for data_id in range(len(data_list)):
        data = data_list[data_id]
        if data_id in invalid_id_list:
            if 'real_rel' not in data:
                data['real_rel'] = data['relation']
            data['relation'] = target_rel
        if data['relation'] == data['real_rel']:
            correct += 1
        saf.write(json.dumps(data))
        saf.write('\n')
    print("correct: {}".format(correct))

def write_valid_file(invalid_list, save_file):
    saf = open(save_file, 'w', encoding='utf-8')
    for data in invalid_list:
        saf.write(json.dumps(data))
        saf.write('\n')


def load_sin_logit(logits_file):
    lof = open(logits_file, 'r', encoding='utf-8')
    for line in lof:
        line = line.strip()
        if len(line) == 0:
            continue
        logit_obj = json.loads(line)
        logits_list = torch.FloatTensor(logit_obj["logits"])
    return logits_list


if __name__ == "__main__":

    rel2id = get_rel2id(args.rel2id_file)
    id2rel = get_id2rel(rel2id)
    logits_res = get_single_logits(args, args.target_rel)
    # logits_res = load_sin_logit(args.logits_file)
    logits_res = deal_logits(logits_res)
    valid_list = get_valid_list(logits_res)
    invalid_list, save_valid_list, invalid_id_list = get_invalid_data_list(valid_list, args.unlabeled_path, args.save_num, args.target_rel)
    if args.type == "save_fix":
        save_fixed_file(invalid_id_list, args.unlabeled_path, args.save_path, args.target_rel)
    elif args.type == "save_valid":
        write_valid_file(save_valid_list, args.save_path)




