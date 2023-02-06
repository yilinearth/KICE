#transfer the form of rel2id file

import json
import pdb

def trans_rel2id(rel2id_file, output_file):
    relf = open(rel2id_file, 'r', encoding='utf-8')
    outf = open(output_file, 'w', encoding='utf-8')

    rel2id = json.load(relf)
    my_rel2id = {}
    for rel in rel2id:
        rel_str = rel.replace("_", " ").replace(":", " ").replace("per", "person").replace("org", "organization")
        idx = rel2id[rel]
        my_rel2id[rel_str] = idx

    json.dump(my_rel2id, outf)
    outf.close()

def convert_relation(org_rel, org_rel2id, new_id2rel):
    if org_rel == "NA":
        org_rel = "no_relation"
    org_id = org_rel2id[org_rel]
    new_rel = new_id2rel[org_id]
    return new_rel

def add_ent_id(input_file, output_file, org_rel2id, new_id2rel):
    inf = open(input_file, 'r', encoding='utf-8')
    outf = open(output_file, 'w', encoding='utf-8')

    for line in inf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        obj['h']['id'] = obj['h']['name'].lower()
        obj['t']['id'] = obj['t']['name'].lower()
        org_rel = obj['relation']
        new_rel = convert_relation(org_rel, org_rel2id, new_id2rel)
        obj['relation'] = new_rel
        outf.write(json.dumps(obj))
        outf.write('\n')
    outf.close()

def get_id2rel(rel2id):
    id2rel = {}
    for rel in rel2id:
        id2rel[rel2id[rel]] = rel
    return id2rel

def get_rel2id(rel2id_file):
    relf = open(rel2id_file, 'r', encoding='utf-8')
    rel2id = json.load(relf)

    return rel2id


if __name__ == '__main__':
    home_dir = ""
    rel2id_file = home_dir + "rel2id.json"
    new_rel2id_file = home_dir + "my_rel2id.json"
    trans_rel2id(rel2id_file, new_rel2id_file)

    org_rel2id = get_rel2id(rel2id_file)
    new_rel2id = get_rel2id(new_rel2id_file)
    new_id2rel = get_id2rel(new_rel2id)

    old_train_file = home_dir + "train.json"
    new_train_file = home_dir + "my_train.json"
    add_ent_id(old_train_file, new_train_file, org_rel2id, new_id2rel)

    old_val_file = home_dir + "dev.json"
    new_val_file = home_dir + "my_dev.json"
    add_ent_id(old_val_file, new_val_file, org_rel2id, new_id2rel)

    old_test_file = home_dir + "test.json"
    new_test_file = home_dir + "my_test.json"
    add_ent_id(old_test_file, new_test_file, org_rel2id, new_id2rel)

