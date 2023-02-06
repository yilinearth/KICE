"""Experiment-running framework."""
import argparse
import importlib
from logging import debug

import numpy as np
from pytorch_lightning.trainer import training_tricks
import torch
import pytorch_lightning as pl
import lit_models
from lit_models import transformer
import yaml
import time
from transformers import AutoConfig, AutoModel
from pytorch_lightning.plugins import DDPPlugin
import os
import pdb
import json
from act_dataloader import *
import scipy.stats
from torch.utils.data import DataLoader


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_class", type=str, default="WIKI80_pred")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--rel2id_file", type=str, default=None, help='relation2id file')
    parser.add_argument("--unlabel_file", type=str, default=None, help='unlabeled set')
    parser.add_argument("--confuse_file", type=str, default=None, help='distribution file')
    parser.add_argument("--query_file", type=str, default=None, help='path for most confused querys')
    parser.add_argument("--query_boot_file", type=str, default=None, help='path for wrong predicted bootstrap rules')
    parser.add_argument("--boot_rule_file", type=str, default=None, help='current rules set file')
    parser.add_argument("--rest_file", type=str, default=None, help='rest data path')
    parser.add_argument("--ckpt_file", type=str, default=None, help='checkpoint file')

    parser.add_argument('--threshold', default=0.97, type=float, help='similar threshold for confuse rules')
    parser.add_argument('--mask_weight', default=[0.3, 0.4, 0.3], type=list, help='mask weights')
    parser.add_argument('--mask_num', default=3, type=int, help='number of masks')
    parser.add_argument('--cpu_num', type=int, default=3, help='cpu number for parallel')

    parser.add_argument('--topk', default=20, type=int, help='number of query data')
    parser.add_argument('--query_rule_num', default=20, type=int, help='number of query data')
    parser.add_argument('--isSelfFix', action='store_true', help='if get fix relation from real rel')
    parser.add_argument("--dataset", type=str, default='tacred', help='dataset name')
    parser.add_argument("--new_data_dir", type=str, default=None, help='next step dictionary')

    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.transformer.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

device = "cuda"
from tqdm import tqdm

def write_unlabel_query(data_list, sorted_data_idx, query_path, rest_path, topk, isSelfFix):
    qf = open(query_path, 'w', encoding='utf-8')
    ref = open(rest_path, 'w', encoding='utf-8')

    for i in range(len(sorted_data_idx)):
        data_id = sorted_data_idx[i]
        obj = data_list[data_id]
        if i <= topk:
            if isSelfFix:
                if 'real_rel' in obj:
                    obj['relation'] = obj['real_rel']
            qf.write(json.dumps(obj))
            qf.write('\n')
        else:
            ref.write(json.dumps(obj))
            ref.write('\n')
    qf.close()
    ref.close()

def write_boot_query(boot_rules, boot_logits, query_boot_path, rel2id, isSelfFix, query_rule_num):
    qbf = open(query_boot_path, 'w', encoding='utf-8')
    wrong_num = 0
    for i in range(len(boot_logits)):
        _, pred_id = torch.max(torch.FloatTensor(boot_logits[i]), 0)
        rule_rel_id = rel2id[boot_rules[i]['relation']]
        if pred_id != rule_rel_id:
            if wrong_num <= query_rule_num and 'is_act' not in boot_rules[i]:
                obj = boot_rules[i]
                # obj = simple_rule_obj(boot_rules[i])
                if isSelfFix:
                    if 'real_rel' in obj:
                        obj['relation'] = obj['real_rel']
                qbf.write(json.dumps(obj))
                qbf.write('\n')
            wrong_num += 1

    qbf.close()

def dump_data_list(data_list, output_file):
    outf = open(output_file, 'w')
    for data in data_list:
        outf.write(json.dumps(data))
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

def load_data_list(in_file):
    inf = open(in_file, 'r')
    data_list = []
    for line in inf:
        line = line.strip()
        if len(line) == 0:
            continue
        data = json.loads(line)
        data_list.append(data)
    return data_list

def get_model_pred_logits(lit_model, data_loader):
    pred_result = lit_model.get_pred_logits(data_loader)
    return pred_result['test_logits']

def get_data_loader(data, dataset):
    dataloader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=data.num_workers, pin_memory=True)
    return dataloader

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    pdb.set_trace()

    # get confuse data
    unlabel_rules_data = load_data_list(args.unlabel_file)
    get_confuse_data_parallel(unlabel_rules_data, args.confuse_file, args.mask_num, args.mask_weight, args.threshold,
                              args.cpu_num)

    #get model
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.transformer.{args.litmodel_class}")

    #get model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model, type='act')
    data.setup()
    model.resize_token_embeddings(len(data.tokenizer))

    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    model_state = torch.load(args.ckpt_file)["state_dict"]
    for key in list(model_state.keys()):
        if 'old_model' in key:
            del (model_state[key])

    lit_model.load_state_dict(model_state)

    lit_model = lit_model.cuda()

    #get prediction
    data_confuse = get_data_loader(data, data.data_confuse)
    data_unlabel = get_data_loader(data, data.data_unlabel)
    data_boot = get_data_loader(data, data.data_boot)

    conf_logits = get_model_pred_logits(lit_model, data_confuse)
    unlabel_logits = get_model_pred_logits(lit_model, data_unlabel)
    boot_logits = get_model_pred_logits(lit_model, data_boot)

    #get unlabel query data
    KL_res = scipy.stats.entropy(unlabel_logits, conf_logits, axis=1)
    _, sorted_data_idxs = torch.sort(torch.FloatTensor(KL_res), descending=True)

    # write results
    write_unlabel_query(unlabel_rules_data, sorted_data_idxs, args.query_file, args.rest_file, args.topk,
                        args.isSelfFix)
    boot_rules = load_data_list(args.boot_rule_file)
    rel2id = get_rel2id(args.rel2id_file)
    write_boot_query(boot_rules, boot_logits, args.query_boot_file, rel2id, args.isSelfFix, args.query_rule_num)



if __name__ == "__main__":

    main()
