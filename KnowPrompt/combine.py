import json
from data_utils import *
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--org_path', default='', help='org total data file')
parser.add_argument('--in_path1', default='', help='input file')
parser.add_argument('--in_path2', default='', help='input file')
parser.add_argument('--output_path', default='', help='output file')
parser.add_argument('--type', default='combine', help='add_data, add rules or add query rules')

parser.add_argument('--old_rule_file', default='', help='old rules set')
parser.add_argument('--query_unlabel_file', default='', help='query data file')
parser.add_argument('--query_boot_file', default='', help='query rules file')
parser.add_argument('--new_rule_file', default='', help='new rules set')
parser.add_argument('--mask_num', default=3, type=int, help='mask number')

args = parser.parse_args()

# combine data from file1 and file2 to output file
def combine(in_file1, in_file2, output_file, type):
    outf = open(output_file, 'w', encoding='utf-8')
    count = 0
    with open(in_file1, 'r', encoding='utf-8') as inf1:
        for line in inf1:
            line = line.strip()
            if len(line) == 0: continue
            obj = json.loads(line)
            if type == 'simple':
                obj = simple_obj(obj) # delete 'mask' and 'mask_emb' attributes in obj
            outf.write(json.dumps(obj))
            outf.write("\n")
            count += 1
            print(count)

    with open(in_file2, 'r', encoding='utf-8') as inf2:
        for line in inf2:
            line = line.strip()
            if len(line) == 0: continue
            obj = json.loads(line)
            if type == 'simple':
                obj = simple_obj(obj)
            outf.write(json.dumps(obj))
            outf.write("\n")
            count += 1
            print(count)

def add_query_rules(old_rule_file, query_unlabel_file, query_boot_file, new_rule_file, mask_num):
    #rules in previous steps
    old_rules = get_data_list(old_rule_file)
    #fixed rules
    query_boot_rules = get_data_list(query_boot_file)
    #update previous rules set
    for query_obj in query_boot_rules:
        # query_id = query_obj['id']
        for i in range(len(old_rules)):
            if isSameObj(old_rules[i], query_obj):
            # if old_rules[i]['id'] == query_id:
                old_rules[i]['relation'] = query_obj['relation']
                old_rules[i]['is_act'] = 1
                for mask_id in range(mask_num):
                    key_name = 'mask_' + str(mask_id)
                    old_rules[i][key_name] = query_obj[key_name]

    #write rules
    nrf = open(new_rule_file, 'w', encoding='utf-8')
    quf = open(query_unlabel_file, 'r', encoding='utf-8')
    #write new human labeled rules
    for line in quf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        obj['is_act'] = 1
        nrf.write(json.dumps(obj))
        nrf.write('\n')
    #write updated previous rules set
    for obj in old_rules:
        nrf.write(json.dumps(obj))
        nrf.write('\n')
    nrf.close()

def fix_relation(org_file, fix_file, out_file):
    orf = open(org_file, 'r')
    id2rel = {}
    count = 0
    for line in orf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        id = obj["id"]
        if id in id2rel:
            pdb.set_trace()
        id2rel[id] = obj['relation']
    fif = open(fix_file, 'r')
    outf = open(out_file, 'w')
    for line in fif:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        id = obj["id"]
        obj['relation'] = id2rel[id]
        outf.write(json.dumps(obj))
        outf.write('\n')
        count += 1
    print(count)
    orf.close()
    fif.close()
    outf.close()



def get_rest_file(sample_file, in_file, rest_file):
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
    sample_list = []
    #get sample list
    sf = open(sample_file, "r")
    inf = open(in_file, "r")
    rf = open(rest_file, "w", encoding='utf-8')
    for line in sf:
        if len(line) <= 0: continue
        line = line.strip()
        obj = json.loads(line)
        sample_list.append(obj)
    count = 0
    for line in inf:
        isSample = False
        if len(line) <= 0: continue
        line = line.strip()
        obj = json.loads(line)
        for sample_obj in sample_list:
            if isSameObj(obj, sample_obj):
                isSample = True
                break
        if not isSample:
            rf.write(line)
            rf.write('\n')
            count += 1
            print(count)
def get_real_data(in_file, out_file):
    inf = open(in_file, 'r')
    outf = open(out_file, 'w')
    for line in inf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)
        if 'real_rel' in obj:
            obj['relation'] = obj['real_rel']
        outf.write(json.dumps(obj))
        outf.write('\n')
    outf.close()

if __name__ == '__main__':
    # used in active learning, add new human labeled rules and fixed rules in previous steps to rules set
    if args.type == 'add_query_rules':
        add_query_rules(args.old_rule_file, args.query_unlabel_file, args.query_boot_file, args.new_rule_file,
                        args.mask_num)
    # add bootstrap rules to rules set or training data to training set
    elif args.type == 'fix':
        fix_relation(args.org_path, args.in_path1, args.output_path)
    elif args.type == 'get_rest':
        get_rest_file(args.in_path1, args.org_path, args.output_path)
    elif args.type == 'get_real':
        get_real_data(args.in_path1, args.output_path)
    else:
        combine(args.in_path1, args.in_path2, args.output_path, args.type)