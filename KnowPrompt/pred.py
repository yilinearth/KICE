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
    parser.add_argument("--topk", type=int, default=None, help='choose topk data with highest model confidence as new rules')
    parser.add_argument("--candidate_file", type=str, default=None, help='new rules file')
    parser.add_argument("--rest_file", type=str, default=None, help='rest data file')
    parser.add_argument("--ckpt_file", type=str, default=None, help='checkpoint file')
    parser.add_argument("--dataset", type=str, default=None, help='dataset name (tacred or retacred)')



    
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


def merge_logit_to_data(pred_result, data_list, id2rel):
    pred_logits = pred_result['test_logits']
    rel2data = {}
    for i in range(len(pred_logits)):
        max_score, max_rel_id = torch.max(torch.FloatTensor(pred_logits[i]), 0)
        data = data_list[i]
        data['confidence'] = float(max_score)
        if 'real_rel' not in data:
            data['real_rel'] = data['relation']
        data['relation'] = id2rel[int(max_rel_id)]
        pred_rel_str = id2rel[int(max_rel_id)]
        if pred_rel_str not in rel2data:
            rel2data[pred_rel_str] = []
        rel2data[pred_rel_str].append(data)

    return rel2data

def get_topk_prediction(rel2data, topk):
    data_list = []
    rest_list = []

    for rel in rel2data:
        for i in range(len(rel2data[rel])):
            if 'confidence' not in rel2data[rel][i]:
                rel2data[rel][i]['confidence'] = -1

        rel2data[rel].sort(key=lambda item: item['confidence'], reverse=True)

        if topk < len(rel2data[rel]):
            data_list.extend(rel2data[rel][:topk])
            rest_list.extend(rel2data[rel][topk:])
        else:
            data_list.extend(rel2data[rel])
        print(len(rel2data[rel]))

    return data_list, rest_list

def dump_data_list(data_list, output_file):
    outf = open(output_file, 'w')
    correct = 0
    for data in data_list:
        if data['relation'] == data['real_rel']:
            correct += 1
        outf.write(json.dumps(data))
        outf.write('\n')
    outf.close()
    print("total {} correct {}".format(len(data_list), correct))

def get_id2rel(rel2id):
    id2rel = {}
    for rel in rel2id:
        id2rel[rel2id[rel]] = rel
    return id2rel

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

def get_data_loader(data, dataset):
    dataloader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=data.num_workers, pin_memory=True)
    return dataloader


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.transformer.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model, type='can')
    data.setup()
    model.resize_token_embeddings(len(data.tokenizer))


    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    model_state = torch.load(args.ckpt_file)["state_dict"]
    for key in list(model_state.keys()):
        if 'old_model' in key:
            del (model_state[key])

    lit_model.load_state_dict(model_state)

    lit_model = lit_model.cuda()

    data_pred = get_data_loader(data, data.data_pred)
    pred_result = lit_model.get_pred_logits(data_pred)
    if args.data_dir[-1] == '0':
        unlabel_data_list = load_data_list(os.path.join(args.data_dir, "rt_aft_samp_valid_emb.json"))
    else:
        unlabel_data_list = load_data_list(os.path.join(args.data_dir, "rt_aft_anno.json"))
    relf = open(args.rel2id_file, 'r')
    rel2id = json.load(relf)
    id2rel = get_id2rel(rel2id)

    rel2data = merge_logit_to_data(pred_result, unlabel_data_list, id2rel)
    can_list, rest_list = get_topk_prediction(rel2data, args.topk)

    dump_data_list(can_list, args.candidate_file)
    dump_data_list(rest_list, args.rest_file)



if __name__ == "__main__":

    main()
