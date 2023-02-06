import json
import argparse
import random

from data_utils import *
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--rel2id_path', default='', help='relation2id file')
parser.add_argument('--total_data_path', default='', help='original training set')
parser.add_argument('--samp_data_path', default='', help='sample data file')
parser.add_argument('--rest_path', default='', help='rest data file after sample')
parser.add_argument('--samp_num', default=86, type=int, help='sample num per relation')

args = parser.parse_args()


def sample_data_num(rel2data, samp_path, rest_path, samp_num):
    sampf = open(samp_path, 'w', encoding='utf-8')
    restf = open(rest_path, 'w', encoding='utf-8')

    for rel_id in range(len(rel2data)):
        count = 0
        print(len(rel2data[rel_id]))
        random.shuffle(rel2data[rel_id])
        for item in rel2data[rel_id]:
            if count < samp_num:
                sampf.write(json.dumps(item))
                sampf.write('\n')
                count += 1
            else:
                restf.write(json.dumps(item))
                restf.write('\n')
                count += 1
    sampf.close()
    restf.close()



if __name__ == '__main__':
    rel2id = get_rel2id(args.rel2id_path)
    rel2data = get_rel2data(args.total_data_path, rel2id)
    sample_data_num(rel2data, args.samp_data_path, args.rest_path, args.samp_num)

