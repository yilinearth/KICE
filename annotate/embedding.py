#get mask word embedding for data and rules
from bert_serving.client import BertClient
import requests
import json
import torch
import argparse
import pdb

parser = argparse.ArgumentParser()
#input your bert client
bc = BertClient(ip='10.214.223.249', check_length=False)


parser.add_argument('--data_path', default='', help='data for embedding')
parser.add_argument('--output_path', default='', help='output path')
parser.add_argument('--mask_num', default=3, type=int, help='number of mask words')
args = parser.parse_args()

#get word embedding
def get_word_emb(word):
    res = bc.encode([word],show_tokens=True)
    bert_tokens = res[1][0]
    emb = torch.sum(torch.FloatTensor(res[0][0][1:len(bert_tokens)-1]), dim=0) / (len(bert_tokens)-2)
    return emb

#get embedding for each type of mask words (the embedding for word list is average of words embeddings)
def get_word_list_emb(word_list):
    emb = None
    word_num = 0
    for word in word_list:
        word = word.replace("Ġ", "").replace(".", "")
        word = word.strip()
        if len(word) == 0:
            continue
        if emb is None:
            emb = get_word_emb(word)
        else:
            emb += get_word_emb(word)
        word_num += 1
    if emb is None:
        emb = torch.zeros(512)
    if word_num != 0:
        emb = emb / word_num
    return emb

#for each data, get embeddings for three types of mask words
def embedding_words(data_path, mask_num, output_path):
    daf = open(data_path, 'r', encoding='utf-8')
    outf = open(output_path, 'w', encoding='utf-8')
    idx = 0
    for line in daf:
        line = line.strip()
        if len(line) == 0:
            continue
        obj = json.loads(line)

        for i in range(mask_num):
            mask_name = 'mask_' + str(i)
            mask_emb = get_word_list_emb(obj[mask_name])
            key_name = 'mask_' + str(i) + '_emb'
            obj[key_name] = mask_emb.tolist()

        outf.write(json.dumps(obj))
        outf.write('\n')
        idx += 1
        print(idx)

    outf.close()


if __name__ == '__main__':
    embedding_words(args.data_path, args.mask_num, args.output_path)