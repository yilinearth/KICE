import json
import os
from pet.wrapper import TransformerModelWrapper
from data_utils.task_processors import PROCESSORS, load_examples, UNLABELED_SET
from pet.modeling import *
from pet.config import EvalConfig
import torch
import pdb

def get_pred_result(model_path, unlabel_data, eval_config, rel2id):

    wrapper = TransformerModelWrapper.from_pretrained(model_path, rel2id)
    label_result = evaluate_mask_word(wrapper, unlabel_data, eval_config)
    return label_result




def get_logits(model_dir, eval_config, rel2id, task_name, data_dir, logits_file):

    lof = open(logits_file, 'w', encoding='utf-8')

    unlabel_data = load_examples(
        task_name, data_dir, UNLABELED_SET, num_examples=-1, num_examples_per_label=None, target_rel=None, rel2id=rel2id)

    # load model and get prediction
    model_path = model_dir
    words_res = get_pred_result(model_path, unlabel_data, eval_config, rel2id)#[inst_num, class_num]
    lof.write(json.dumps(words_res))
    lof.write('\n')

    return words_res









