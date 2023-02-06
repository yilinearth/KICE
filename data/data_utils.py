import json

def get_rel2data(in_file, rel2idx):
    rel2rules = []
    max_id = -1
    for rel in rel2idx:
        if rel2idx[rel] > max_id:
            max_id = rel2idx[rel]
    for i in range(max_id+1):
        rel2rules.append([])
    inf = open(in_file, "rb")
    for line in inf:
        if(len(line) != 0):
            obj = json.loads(line)
            rel_str = obj["relation"]
            idx = rel2idx[rel_str]
            rel2rules[idx].append(obj)
    return rel2rules

def get_id2rel(rel2id):
    id2rel = {}
    for rel in rel2id:
        id2rel[rel2id[rel]] = rel
    return id2rel

def get_rel2id(rel2id_file):
    relf = open(rel2id_file, 'r', encoding='utf-8')
    rel2id = json.load(relf)

    return rel2id

def simple_obj(obj):
    if 'mask_0_emb' in obj:  del (obj['mask_0_emb'])
    if 'mask_1_emb' in obj:  del (obj['mask_1_emb'])
    if 'mask_2_emb' in obj:  del (obj['mask_2_emb'])
    # if 'mask_0' in obj: del(obj['mask_0'])
    # if 'mask_1' in obj: del (obj['mask_1'])
    # if 'mask_2' in obj: del (obj['mask_2'])
    if 'is_act' in obj: obj['is_act'] = 1
    return obj

def simple_rule_obj(obj):
    if 'mask_0_emb' in obj:  del (obj['mask_0_emb'])
    if 'mask_1_emb' in obj:  del (obj['mask_1_emb'])
    if 'mask_2_emb' in obj:  del (obj['mask_2_emb'])
    return obj

def get_data_list(data_file):
    inf = open(data_file, 'r', encoding='utf-8')
    data_list = []
    for line in inf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        data_list.append(obj)
    return data_list

def isSameObj(obj1, obj2):
    sent1 = " ".join(obj1['token'])
    sent2 = " ".join(obj2['token'])
    h1 = obj1['h']['name'].lower()
    h2 = obj2['h']['name'].lower()
    t1 = obj1['t']['name'].lower()
    t2 = obj2['t']['name'].lower()
    if sent1.lower() == sent2.lower() and h1 == h2 and t1 == t2:
        return True
    else:
        return False