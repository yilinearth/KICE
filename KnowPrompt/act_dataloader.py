import torch
import sys
import json
import copy
import multiprocessing

def cal_cos(emb1, emb2, sim_function):
    emb1 = torch.FloatTensor(emb1)
    emb2 = torch.FloatTensor(emb2)
    cos_score = sim_function(emb1.unsqueeze(0), emb2.unsqueeze(0))
    return cos_score

def cal_sim_score(obj1, obj2, mask_num, mask_weight):
    sim_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    score = 0.0
    for i in range(mask_num):
        emb_name = 'mask_' + str(i) + '_emb'
        score += mask_weight[i]*cal_cos(obj1[emb_name], obj2[emb_name], sim_function)
    return float(score)

#for each rule, get its confuse rules
def get_confuse_data_q(unlabel_rules, conf_path, mask_num, mask_weight, threshold):
    conf = open(conf_path, 'w', encoding='utf-8')
    no_conf_count = 0
    for i in range(len(unlabel_rules)):
        is_find = False
        for j in range(len(unlabel_rules)):
            if j != i:
                sim_score = cal_sim_score(unlabel_rules[i], unlabel_rules[j], mask_num, mask_weight)
                if sim_score >= threshold:
                    conf.write(json.dumps(unlabel_rules[j]))
                    conf.write('\n')
                    is_find = True
                    break
        if not is_find:
            conf.write(json.dumps(unlabel_rules[i]))
            conf.write('\n')
            no_conf_count += 1
            print("no_conf {}, total {}".format(no_conf_count, i))
        print("build sim score: {}".format(i))

    conf.close()

def get_confuse_data_sin(sin_rules_list, total_rule_list, mask_num, mask_weight, threshold, sin_list_len, share_dict, id):

    no_conf_count = 0
    conf_rules_list = [] #长度应该和sin_rules_list一样
    for i in range(len(sin_rules_list)):
        is_find = False
        for j in range(len(total_rule_list)):
            if j != sin_list_len*id + i:
                sim_score = cal_sim_score(sin_rules_list[i], total_rule_list[j], mask_num, mask_weight)
                if sim_score >= threshold:
                    conf_rules_list.append(total_rule_list[j])
                    is_find = True
                    break
        if not is_find:
            conf_rules_list.append(sin_rules_list[i])
            no_conf_count += 1
            print("no_conf {}, total {}".format(no_conf_count, i))
        print("build sim score: {}".format(i))

    assert len(conf_rules_list) == len(sin_rules_list)
    share_dict[id] = conf_rules_list
    return share_dict


def get_confuse_data_parallel(unlabel_rules, conf_path, mask_num, mask_weight, threshold, cpu_num):
    def divide_and_conquer(unlabel_rules, mask_num, mask_weight, threshold, cpu_num, share_dict):
        rules_len = int(len(unlabel_rules) / cpu_num)
        parameter_list = []
        for i in range(cpu_num):
            start_id = i*rules_len
            end_id = (i+1)*rules_len if (i+1) < cpu_num else len(unlabel_rules)
            sin_rule_list = copy.deepcopy(unlabel_rules[start_id: end_id])
            total_rule_list = copy.deepcopy(unlabel_rules)
            sin_mask_num = copy.deepcopy(mask_num)
            sin_mask_weight = copy.deepcopy(mask_weight)
            sin_threshold = copy.deepcopy(threshold)
            sin_list_len = copy.deepcopy(rules_len)
            parameter_list.append((sin_rule_list, total_rule_list, sin_mask_num, sin_mask_weight, sin_threshold, sin_list_len))
        # for i in range(cpu_num):
        #     get_confuse_data_sin(parameter_list[i][0], parameter_list[i][1], parameter_list[i][2], parameter_list[i][3],
        #                                                          parameter_list[i][4], parameter_list[i][5], share_dict, i)
        process_pool = multiprocessing.Pool(cpu_num)
        for i in range(cpu_num):
            process_pool.apply_async(get_confuse_data_sin, args=(parameter_list[i][0], parameter_list[i][1], parameter_list[i][2], parameter_list[i][3],
                                                                 parameter_list[i][4], parameter_list[i][5], share_dict, i))
            print("start process {}".format(i))
        process_pool.close()
        process_pool.join()

    manager = multiprocessing.Manager()
    share_dict = manager.dict()
    divide_and_conquer(unlabel_rules, mask_num, mask_weight, threshold, cpu_num, share_dict)
    total_conf_list = []
    for i in range(cpu_num):
        total_conf_list += share_dict[i]

    conf = open(conf_path, 'w')
    for conf_data in total_conf_list:
        conf.write(json.dumps(conf_data))
        conf.write('\n')

    conf.close()


#for each rule, get its confuse rules
def get_confuse_data(unlabel_rules, conf_path, mask_num, mask_weight):
    conf = open(conf_path, 'w', encoding='utf-8')
    sim_scores = torch.zeros(len(unlabel_rules), len(unlabel_rules))

    for i in range(len(unlabel_rules)-1):
        for j in range(i+1, len(unlabel_rules)):
            sim_scores[i][j] = cal_sim_score(unlabel_rules[i], unlabel_rules[j], mask_num, mask_weight)
            sim_scores[j][i] = sim_scores[i][j]
        print("build sim score: {}".format(i))

    _, rule_ids = torch.max(sim_scores, 1)

    for i in range(len(unlabel_rules)):
        confuse_id = rule_ids[i]
        conf.write(json.dumps(unlabel_rules[confuse_id]))
        conf.write('\n')

    conf.close()


