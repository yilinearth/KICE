import json
import random
import pdb
import os

def samp(total_file, samp_num):
    tof = open(total_file, 'r', encoding='utf-8')
    obj_list = []
    for line in tof:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        obj_list.append(obj)
    random.shuffle(obj_list)

    samp_obj = []
    rest_obj = []
    for id, obj in enumerate(obj_list):
        if id <= samp_num:
            samp_obj.append(obj)
        else:
            rest_obj.append(obj)
    return samp_obj, rest_obj


def write_obj(obj_list, out_file):
    outf = open(out_file, 'w', encoding='utf-8')
    for obj in obj_list:
        outf.write(json.dumps(obj))
        outf.write("\n")
    outf.close()


def get_rel2id(rel2id_file):
    relf = open(rel2id_file, 'r', encoding='utf-8')
    rel2id = json.load(relf)

    return  rel2id

def samp_test(test_file, samp_rel_list, rel2id):
    samp_list = []
    tesf = open(test_file, 'r', encoding='utf-8')
    for line in tesf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        rel_str = obj['relation']
        if rel2id[rel_str] in samp_rel_list:
            samp_list.append(obj)
    return samp_list

#id2samp_inst
def get_samp_list(samp_dir, rel2id):
    rel_dir_list = os.listdir(samp_dir)
    samp_list = []
    for i in range(len(rel2id)):
        samp_list.append([])
    for rel_dir in rel_dir_list:
        if 'rel_' not in str(rel_dir):
            continue
        sub_train_path = os.path.join(samp_dir, rel_dir, 'train.txt')
        subf = open(sub_train_path, 'r', encoding='utf-8')
        for line in subf:
            line = line.strip()
            if len(line) == 0:
                continue
            obj = json.loads(line)
            rel = obj['relation']
            rel_id = rel2id[rel]
            samp_list[rel_id].append(obj['id'])
        subf.close()
    return samp_list

def get_rest_list(total_file, rel2id, samp_list, rest_file):
    tof = open(total_file, 'r', encoding='utf-8')
    ref = open(rest_file, 'w', encoding='utf-8')
    for line in tof:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        rel = obj['relation']
        id = obj['id']
        rel_id = rel2id[rel]
        if id not in samp_list[rel_id]:
            ref.write(json.dumps(obj))
            ref.write('\n')
    ref.close()



if __name__ == "__main__":
    home = "/home/luyilin/Ptuningv1/PT-Fewshot/FewGLUE_32dev/BoolRE/"
    samp_type = "get_rest"
    if samp_type == "unlabel":
        total_file = home + "org/unlabel.txt"
        samp_file = home + "step0/unlabel.txt"
        rest_file = home + "step0/rest.txt"
        pdb.set_trace()
        samp_num = 30000
        samp_list, rest_list = samp(total_file, samp_num)
        print(len(samp_list))
        print(len(rest_list))
        write_obj(samp_list, samp_file)
        write_obj(rest_list, rest_file)

    if samp_type == "test":
        test_file = home + "org/test.txt"
        samp_file = home + "org/test_rel_1x.txt"
        rel2id_file = home + "org/rel2id.json"
        samp_rel_list = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        rel2id = get_rel2id(rel2id_file)
        samp_list = samp_test(test_file, samp_rel_list, rel2id)
        write_obj(samp_list, samp_file)

    if samp_type == "get_rest":
        pdb.set_trace()
        total_file = home + "tacred/ptune_train.json"
        samp_dir = home + "tacred4/step0"
        rel2id_file = home + "tacred/rel2id.json"
        rest_file = home + "tacred4/step0/rest_aft_step0.txt"
        rel2id = get_rel2id(rel2id_file)

        samp_list = get_samp_list(samp_dir, rel2id)
        get_rest_list(total_file, rel2id, samp_list, rest_file)

