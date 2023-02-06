# coding:utf-8
import sys, json
import torch
import os
import numpy as np
import cos_opennre
from cos_opennre import framework
import argparse
import logging
import random
import time
import pdb
import pickle

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='',
        help='Checkpoint name')
parser.add_argument('--result', default='',
        help='Result name')
parser.add_argument('--pooler', default='cls', choices=['cls', 'entity'],
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true',
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='mic_f1', choices=['mic_f1'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'wiki_distant', 'nyt10', 'nyt10m', 'wiki20m'],
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

parser.add_argument('--res_file', default='',
        help='eval result files')
parser.add_argument('--pr_res_file', default='',
        help='pr result files')

# Bag related
parser.add_argument('--bag_size', type=int, default=4,
        help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=16, type=int,
        help='Batch size')
######4e-6
parser.add_argument('--lr', default=4e-6, type=float,
        help='Learning rate')
# parser.add_argument('--lr', default=8e-7, type=float,
#         help='Learning rate')

parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')
#self
parser.add_argument('--self_training_eps', type=float,
        default=0.5, help='threshold for confidence')
parser.add_argument('--self_training_confreg', type=float, default=0.05,
        help='confidence smooth power')
parser.add_argument('--distmetric', type=str, default="cos",
        help='distance type. Choices = [cos, l2]')
parser.add_argument('--self_training_contrastive_weight', type=float, default=0,
        help='contrastive learning weight')
parser.add_argument('--old_ckpt', default='',
        help='Old Checkpoint name')
parser.add_argument('--device', default='cuda',
        help='Device')
parser.add_argument('--self_training_power', type=float, default=2,
                    help='power of pred score')
parser.add_argument('--calc_loss_weight', type=float, default=0.1,
                    help='weight of calc loss')
parser.add_argument('--loss_weight', action='store_true',
        help='Only run test')
parser.add_argument('--is_focal_loss', action='store_true',
        help='is used focal loss')
# Exp
parser.add_argument('--aggr', default='att', choices=['one', 'att', 'avg'])

# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)
args = parser.parse_args()

# Some basic settings
root_path = '..'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

time_str = time.strftime("%Y%m%d-%H%M%S")

ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)
old_ckpt = 'ckpt/{}.pth.tar'.format(args.old_ckpt)

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))


# Define the sentence encoder
if args.pooler == 'entity':
    sentence_encoder = cos_opennre.encoder.BERTEntityEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
elif args.pooler == 'cls':
    sentence_encoder = cos_opennre.encoder.BERTEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
else:
    raise NotImplementedError


# Define the model
if args.aggr == 'att':
    model = cos_opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)
elif args.aggr == 'avg':
    model = cos_opennre.model.BagAverage(sentence_encoder, len(rel2id), rel2id)
elif args.aggr == 'one':
    model = cos_opennre.model.BagOne(sentence_encoder, len(rel2id), rel2id)
else:
    raise NotImplementedError

framework = framework.BagRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt="adamw",
    bag_size=args.bag_size,
    old_state_dict=old_ckpt,
    loss_weight=args.loss_weight,
    args=args,
    is_focal_loss=args.is_focal_loss
)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.wrench_eval_model(framework.test_loader)
# Print the result
resf = open(args.res_file, 'w', encoding='utf-8')

logging.info('Test set results:')

resf.write('wrench eval result:\n')
resf.write(json.dumps(result))
resf.write('\n')
print(result)
