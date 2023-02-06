from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json
import pdb
import argparse

def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="tacred")
    return parser


parser = _setup_parser()
args = parser.parse_args()
pdb.set_trace()
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
def split_label_words(tokenizer, label_list):#从label list中获得每个label的token ids
    label_word_list = []
    for label in label_list:
        if label == 'no_relation' or label == "NA" or label == 'no relation':
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            tmps = label
            label = label.lower()
            # label = label.split("(")[0]
            # label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list

with open(f"../data/{args.dataset_name}/my_rel2id.json", "r") as file:
    t = json.load(file)
    label_list = list(t)

t = split_label_words(tokenizer, label_list)

with open(f"../data/{args.dataset_name}/{args.model_name_or_path}_{args.dataset_name}.pt", "wb") as file:
    torch.save(t, file)