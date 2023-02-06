import json
import os
from pet.wrapper import TransformerModelWrapper
from data_utils.task_processors import PROCESSORS, load_examples, UNLABELED_SET
from pet.modeling import evaluate
from pet.config import EvalConfig
import torch
import pdb



def get_pred(model_path, unlabel_data, eval_config, rel2id=None):
    wrapper = TransformerModelWrapper.from_pretrained(model_path, rel2id=rel2id)
    label_result = evaluate(wrapper, unlabel_data, eval_config)
    return label_result['logits']


def get_sin_logit(models_dir, eval_config, target_relid, target_rel, task_name, data_dir, logits_file):

    lof = open(logits_file, 'w', encoding='utf-8')

    count = 0
    unlabel_data = load_examples(
        task_name, data_dir, UNLABELED_SET, num_examples=-1, num_examples_per_label=None, target_rel=target_rel)

    # load model and get prediction
    rel_dir = 'rel_' + str(target_relid)
    model_path = os.path.join(models_dir, 'p1-i0')

    if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
        print("No model file here!")

    logit_res = get_pred(model_path, unlabel_data, eval_config)

    write_obj = {}
    write_obj['relid'] = target_relid
    write_obj['logits'] = logit_res.tolist()
    lof.write(json.dumps(write_obj))
    lof.write('\n')
    count += 1
    print(count)

    return logit_res


def get_logits(model_dir, eval_config, rel2id, task_name, data_dir, logits_file):

    lof = open(logits_file, 'w', encoding='utf-8')

    unlabel_data = load_examples(
        task_name, data_dir, UNLABELED_SET, num_examples=-1, num_examples_per_label=None, target_rel=None, rel2id=rel2id)

    # load model and get prediction
    model_path = model_dir
    words_res = get_pred(model_path, unlabel_data, eval_config, rel2id)#[inst_num, class_num]
    lof.write(json.dumps(words_res.tolist()))
    lof.write('\n')

    return words_res



