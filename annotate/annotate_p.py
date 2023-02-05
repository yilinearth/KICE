import json
import time
import copy
import torch
import argparse
import pdb
import multiprocessing
from anno_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='', help='data for annotating')
parser.add_argument('--rule_path', default='', help='rule path')
parser.add_argument('--tmp_path', default='', help='tmp file path for load')
parser.add_argument('--output_path', default='', help='annotate results path')
parser.add_argument('--rest_path', default='', help='rest data path')
parser.add_argument('--rel2id_path', default='', help='rel2id data path')
parser.add_argument('--pr_res_path', default='', help='evaluation result path')
parser.add_argument('--topk', default=200, type=int, help='topk number of annotation')
parser.add_argument('--threshold', default=0.95, type=float, help='similar threshold')
parser.add_argument('--load', action='store_true', help='is load tmp file?')
parser.add_argument('--is_sort', action='store_true', help='is sort anno list?')
parser.add_argument('--mask_weight', default=[0.3, 0.4, 0.3], type=list, help='weights for two concept feature and relation feature')
parser.add_argument('--type', default='simple', help='annotate manner')
parser.add_argument('--mask_num', default=3, type=int, help='number of mask words')
parser.add_argument('--save_na_num', default=600, type=int, help='number of no relation data')

args = parser.parse_args()
mask_weight = args.mask_weight


def cal_sim_score(obj1, obj2, mask_num):
    sim_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    score = 0.0
    for i in range(mask_num):
        emb_name = 'mask_' + str(i) + '_emb'
        score += mask_weight[i]*cal_cos(obj1[emb_name], obj2[emb_name], sim_function)
    return float(score)

#major voting
def deal_with_single_data_list_most_vote(sub_list: list, rule_list: list, mask_num: int, threshold: float, shared_dict, idx, rel2id) -> None:
    correct_num, no_rel_count, total_count = 0, 0, 0
    res_list = []
    for data_obj in sub_list:
       # go through all the rules and measure the similarity
        rel2votes = {}
        best_score = 0.0
        for rule in rule_list:
            score = cal_sim_score(data_obj, rule, mask_num)
            # participate the voting if similarity beyond the threshold
            if score >= threshold:
                rule_rel = rule['relation']
                if rule_rel not in rel2votes:
                    rel2votes[rule_rel] = 0
                rel2votes[rule_rel] += score

        # pick the label with most votes
        if len(rel2votes) != 0:
            sort_rel2votes = sorted(rel2votes.items(), key = lambda x: x[1], reverse=True)
            best_rel = sort_rel2votes[0][0]
            best_score = rel2votes[best_rel]
        else:
            best_rel = 'no relation' if 'no relation' in rel2id else list(rel2id.keys())[0]
            best_score = 0

        # record the ground truth label
        if 'real_rel' not in data_obj:
            data_obj['real_rel'] = data_obj['relation']

        total_count += 1

        # correct labeling number
        if best_rel == data_obj['real_rel']:
            correct_num += 1

        # the number of wrong labeled no relation data
        if data_obj['real_rel'] == "no relation" and best_rel != "no relation":
            no_rel_count += 1

        data_obj['relation'] = best_rel #rule generated label
        data_obj['confidence'] = best_score #labeling confidence
        print("{}-{}-{}".format(total_count, correct_num, no_rel_count))

        res_list.append(copy.deepcopy(data_obj))
    shared_dict[idx] = ((res_list, correct_num, no_rel_count, total_count))

def deal_with_single_data_list_max_vote(sub_list: list, rule_list: list, mask_num: int, threshold: float, shared_dict, idx, rel2id) -> None:
    correct_num, no_rel_count, total_count = 0, 0, 0
    res_list = []
    for data_obj in sub_list:
       # go through all the rules and measure the similarity
        rel2votes = {}
        rel2scores = {}
        for rule in rule_list:
            score = cal_sim_score(data_obj, rule, mask_num)
            # participate the voting if similarity beyond the threshold
            if score >= threshold:
                rule_rel = rule['relation']
                if rule_rel not in rel2votes:
                    rel2votes[rule_rel] = 0
                    rel2scores[rule_rel] = 0
                if score > rel2scores[rule_rel]:
                    rel2scores[rule_rel] = score
                    rel2votes[rule_rel] += 1

        # pick the label with most votes and largest similarity
        if len(rel2scores) != 0:
            sort_rel2score= sorted(rel2scores.items(), key = lambda x: x[1], reverse=True)
            best_rel = sort_rel2score[0][0]
            best_score = rel2scores[best_rel]

        else:
            best_rel = 'no relation' if 'no relation' in rel2id else list(rel2id.keys())[0]
            best_score = 0

        # record the ground truth label
        if 'real_rel' not in data_obj:
            data_obj['real_rel'] = data_obj['relation']

        total_count += 1


        # correct labeling number
        if best_rel == data_obj['real_rel']:
            correct_num += 1

        # the number of wrong labeled no relation data
        if data_obj['real_rel'] == "no relation" and best_rel != "no relation":
            no_rel_count += 1

        data_obj['relation'] = best_rel #rule generated label
        data_obj['confidence'] = best_score #labeling confidence

        res_list.append(copy.deepcopy(data_obj))
        print("{}-{}-{}".format(total_count, correct_num, no_rel_count))

    shared_dict[idx] = ((res_list, correct_num, no_rel_count, total_count))

#parallel annotation
def knn_annotate_best_conf_parallel(data_path, rule_list, mask_num, tmp_path, threshold, rel2id, is_sort=True):
    
    def read_data_list(data_path: str) -> list:
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    continue
                data_obj = json.loads(line)
                data_list.append(data_obj)
        return data_list
    
    def divide_and_conquer(data_list: list, rule_list: list, cpu_num: int, mask_num: int, threshold: float, shared_dict, rel2id) -> None:
        single_list_len = int(len(data_list) / cpu_num)
        if single_list_len < 5:
            cpu_num = 1
        parameter_list = []
        for i in range(cpu_num):
            start_pos = i*single_list_len
            end_pos = (i+1)*single_list_len if cpu_num != i+1 else len(data_list)
            rule_list_copy = copy.deepcopy(rule_list)
            sub_list_copy  = copy.deepcopy(data_list[start_pos: end_pos])
            mask_num_copy  = copy.deepcopy(mask_num)
            threshold_copy = copy.deepcopy(threshold)
            parameter_list.append((sub_list_copy, rule_list_copy, mask_num_copy, threshold_copy))
        process_pool = multiprocessing.Pool(cpu_num)
        for i in range(cpu_num):
            process_pool.apply_async(deal_with_single_data_list_max_vote, args=(parameter_list[i][0], parameter_list[i][1], parameter_list[i][2], parameter_list[i][3], shared_dict, i, rel2id))
            print("start process {}".format(i))
        process_pool.close()
        process_pool.join()

    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    start_time = time.time()
    anno_res_list = []
    correct_num, no_rel_count, total_count = 0, 0, 0

    data_list = read_data_list(data_path)
    parallel_cpu_num = 10
    divide_and_conquer(data_list, rule_list, parallel_cpu_num, mask_num, threshold, shared_dict, rel2id)

    end_time = time.time()
    print("time: {}s".format(end_time-start_time))
    
    for i in range(parallel_cpu_num):
        res_list, single_correct_num, single_no_rel_count, single_total_count = shared_dict[i]
        anno_res_list += res_list
        correct_num   += single_correct_num
        no_rel_count  += single_no_rel_count
        total_count   += single_total_count
        
    with open(tmp_path, 'w', encoding='utf-8') as fp:
        #save the annotated results
        for data_obj in anno_res_list:
            fp.write(json.dumps(data_obj)+'\n')


    print("total_data is {}, correct is {}".format(total_count, correct_num))

    #sort the annotated data with labeling confidence
    if is_sort:
        anno_res_list.sort(key=lambda item: item['confidence'], reverse=True)

    return anno_res_list

def get_anno_res(anno_res_list, rest_list, top_num, output_path, rest_path, save_na_num=0):
    outf = open(output_path, 'w', encoding='utf-8')
    ref = open(rest_path, 'w', encoding='utf-8')

    data_idx = 0
    rest_num = 0
    na_count = 0
    pos_correct = 0
    correct = 0
    rel2num = {}
    for data in anno_res_list:
        if data_idx > top_num or (data['relation'] == 'no relation' and na_count > save_na_num)\
                or (data['relation'] in rel2num and rel2num[data['relation']] >= int((top_num-save_na_num) / 4)):
            ref.write(json.dumps(data))
            ref.write('\n')
            rest_num += 1
        else:
            if data['relation'] == 'no relation':
                na_count += 1
            outf.write(json.dumps(data))
            outf.write('\n')
            data_idx += 1
            if data['relation'] not in rel2num:
                rel2num[data['relation']] = 0
            rel2num[data['relation']] += 1

            if data['relation'] == data['real_rel']:
                correct += 1
                if data['relation'] != 'no relation':
                    pos_correct += 1

    for data in rest_list:
        ref.write(json.dumps(data))
        ref.write('\n')
        rest_num += 1
    outf.close()
    ref.close()
    print("rest num: {}".format(rest_num))
    print("pos correct num: {}".format(pos_correct))
    print("total correct num: {}".format(correct))

def load_tmp(tmp_path, is_sort=True):
    tmf = open(tmp_path, 'r', encoding='utf-8')
    anno_res_list = []
    for line in tmf:
        line = line.strip()
        if len(line) == 0:
            continue
        anno_res_list.append(json.loads(line))
    if is_sort:
        anno_res_list.sort(key=lambda item: item['confidence'], reverse=True)
    return anno_res_list

if __name__ == '__main__':

    rel2id = get_rel2id(args.rel2id_path)
    if args.load:
        #load anno_res_list from tmp_path
        anno_res_list = load_tmp(args.tmp_path, args.is_sort)
    else:
        rule_list = get_rule_list(args.rule_path)
        anno_res_list = knn_annotate_best_conf_parallel(args.data_path, rule_list, args.mask_num, args.tmp_path, args.threshold, rel2id, args.is_sort)


    rest_list = []
    valid_anno_list = []
    for obj in anno_res_list:
        if obj['confidence'] == 0:
            rest_list.append(obj)
        else:
            valid_anno_list.append(obj)
    valid_anno_list.sort(key=lambda item: item['confidence'], reverse=True)
    #compute label precision and recall
    # if not args.load:
    #     eval_anno_list(valid_anno_list, rel2id, args.pr_res_path)

    get_anno_res(valid_anno_list, rest_list, args.topk, args.output_path, args.rest_path, args.save_na_num)
