# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
import jsonpickle
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, RobertaForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig
from transformers.data.metrics import simple_accuracy

import log
from pet import preprocessor
from data_utils.task_processors import TASK_HELPERS
from pet.config import WrapperConfig, EvalConfig
from pet.utils import InputFeatures, DictDataset, distillation_loss, exact_match
import pdb
logger = log.get_logger('root')

CONFIG_NAME = 'wrapper_config.json'
MLM_WRAPPER = "mlm"

WRAPPER_TYPES = [MLM_WRAPPER]

PREPROCESSORS = {
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        MLM_WRAPPER: BertForMaskedLM
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        MLM_WRAPPER: RobertaForMaskedLM
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        MLM_WRAPPER: AlbertForMaskedLM
    },
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step
}






class ContinuousPrompt(torch.nn.Module):
    def __init__(self, config:WrapperConfig, tokenizer):
        super(ContinuousPrompt, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_size = config.embed_size
        self.hidden_size = self.embed_size#768
        self.prompt_length = self.config.pattern_id # The pattern_id is supposed to indicate the number of continuous prompt tokens.
        #prompt length: 有多少个连续prompt token

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False)


        model_class = MODEL_CLASSES[self.config.model_type][MLM_WRAPPER]
        self.model = model_class.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None)

        #prompt_size, embed_size
        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.embed_size)
        if config.prompt_encoder_type == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size))
        elif config.prompt_encoder_type == "mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size))
        else:
            raise ValueError('unknown prompt_encoder_type.')


    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):

        return self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          labels=labels,
                          token_type_ids=token_type_ids)





class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig, rel2id=None):
        self.config = config

        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)

        self.model = ContinuousPrompt(config, self.tokenizer)

        self.preprocessor = PREPROCESSORS[MLM_WRAPPER](self,
                                                       self.config.task_name,
                                                       self.config.pattern_id,
                                                       rel2id)
        #wsc有
        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None
        self.rel2id = rel2id
        #######
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()######


    def save(self, path: str) -> None:
        logger.info("Saving models.")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

        if self.config.prompt_encoder_type == "lstm":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "lstm_head": model_to_save.lstm_head.state_dict(),
                "mlp_head": model_to_save.mlp_head.state_dict()
            }
        elif self.config.prompt_encoder_type == "mlp":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "mlp": model_to_save.mlp.state_dict()
            }
        else:
            raise ValueError("unknown prompt_encoder_type.")

        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)


    @classmethod
    def from_pretrained(cls, path: str, rel2id=None) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)

        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)

        wrapper.model = ContinuousPrompt(wrapper.config, wrapper.tokenizer)
        model_class = MODEL_CLASSES[wrapper.config.model_type][MLM_WRAPPER]
        wrapper.model.model = model_class.from_pretrained(path)

        save_path_file = os.path.join(path, "embeddings.pth")
        ######
        # data = torch.load(save_path_file, map_location='cpu')
        data = torch.load(save_path_file)
        wrapper.model.prompt_embeddings.load_state_dict(data["prompt_embeddings"])
        if "lstm_head" in data:
            assert ("mlp_head" in data)
            wrapper.model.lstm_head.load_state_dict(data["lstm_head"])
            wrapper.model.mlp_head.load_state_dict(data["mlp_head"])
        if "mlp" in data:
            wrapper.model.mlp_head.load_state_dict(data["mlp"])

        wrapper.preprocessor = PREPROCESSORS[MLM_WRAPPER](wrapper, wrapper.config.task_name, wrapper.config.pattern_id, rel2id)
        wrapper.rel2id = rel2id
        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
            if wrapper.config.task_name in TASK_HELPERS else None
        ######
        # if torch.cuda.device_count() > 1:
        #     wrapper.model = torch.nn.DataParallel(wrapper.model)
        wrapper.model.cuda()######

        return wrapper


    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))


    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self,
              train_data:List[InputExample],
              eval_data:List[InputExample],
              dev_data:List[InputExample],
              eval_config:EvalConfig,
              pattern_iter_output_dir,
              per_gpu_train_batch_size: int = 8,
              n_gpu: int = 1,
              num_train_epochs: int = 3,
              gradient_accumulation_steps: int = 1,
              weight_decay: float = 0.0,
              learning_rate: float = 5e-5,
              adam_epsilon: float = 1e-8,
              warmup_steps=0,
              max_grad_norm: float = 1,
              logging_steps: int = 50,
              max_steps=-1, **_):
        """
        Train the underlying language model.

        :param train_data: the training examples to use
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        print("\n")
        print("num_steps_per_dataset:")
        print(len(train_dataloader) // gradient_accumulation_steps)
        print("total_steps:")
        print(t_total)
        print("num_train_epochs:")
        print(num_train_epochs)
        print("\n")


        cur_model = self.model.module if hasattr(self.model, 'module') else self.model#=self.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [#预训练模型的所有参数
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
        ]

        if self.config.prompt_encoder_type == "lstm":
            embedding_parameters = [
                {'params': [p for p in cur_model.lstm_head.parameters()]},
                {'params': [p for p in cur_model.mlp_head.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
            ]#记录一些参数,以便定义优化器
        elif self.config.prompt_encoder_type == "mlp":
            embedding_parameters = [
                {'params': [p for p in cur_model.mlp.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        embedding_optimizer = AdamW(embedding_parameters, lr=learning_rate, eps=adam_epsilon)
        embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, "writer_logs"))

        ### TODO
        prev_loss = 0.0
        best_dev_acc = 0.0
        best_dev_f1 = 0.0
        best_global_step = 0
        best_loss = 0.0
        early_stop_epoch = 0

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        logger.info("dev_data performance before training.")

        dev_scores = self.eval_dev(dev_data, eval_config, n_gpu)
        logger.info(dev_scores)

        logger.info("eval_data performance before training.")

        dev_scores = self.eval_dev(eval_data, eval_config, n_gpu)
        logger.info(dev_scores)

        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = {k: t.cuda() for k, t in batch.items()}######
                # batch = {k: t for k, t in batch.items()}

                loss = self.task_helper.train_step(batch) if self.task_helper else None
                if loss is None:
                    loss = TRAIN_STEP_FUNCTIONS[MLM_WRAPPER](self)(batch)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    ## TODO
                    writer.add_scalar("train_loss", (tr_loss - prev_loss), global_step=global_step)
                    prev_loss = tr_loss

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    embedding_optimizer.step()
                    embedding_scheduler.step()

                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        print(json.dumps({**logs, **{'step': global_step}}))

                    ## TODO
                    if global_step % self.config.eval_every_step == 0:
                        dev_scores = self.eval_dev(dev_data, eval_config, n_gpu)

                        if self.config.task_name in ["cb", "record", "multirc", "multire", "multicon", "sinre"]:
                            f1_str = "f1" if (self.config.task_name != "cb" and self.config.task_name != "multire" and
                                              self.config.task_name != "multicon" and self.config.task_name != "sinre") else "f1-macro"
                            ######
                            # f1_str = 'nero_f1'
                            if dev_scores["acc"] >= best_dev_acc and dev_scores[f1_str] >= best_dev_f1:

                                if dev_scores["acc"] > best_dev_acc and dev_scores[f1_str] > best_dev_f1:
                                    early_stop_epoch = 0
                                else:
                                    early_stop_epoch += 1

                                best_dev_acc = dev_scores["acc"]
                                best_dev_f1 = dev_scores[f1_str]
                                best_global_step = global_step
                                best_loss = tr_loss

                                logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                                logger.info("best_dev_acc: %.4f | best_dev_f1: %.4f | best_global_step: %d" % \
                                            (best_dev_acc, best_dev_f1, best_global_step))
                                logger.info(dev_scores)

                                self.save(pattern_iter_output_dir)
                                ######
                                # logger.info("eval_data performance:")
                                # eval_scores = self.eval_dev(eval_data, eval_config, n_gpu)
                                # logger.info(eval_scores)
                            else:
                                early_stop_epoch += 1
                                logger.info(dev_scores)
                                logger.info(early_stop_epoch)


                        elif self.config.task_name in ["rte", "wic", "boolq", "wsc", "copa", "boolre"]:
                            if dev_scores["acc"] >= best_dev_acc:
                                if dev_scores["acc"] > best_dev_acc:
                                    early_stop_epoch = 0
                                else:
                                    early_stop_epoch += 1

                                best_dev_acc = dev_scores["acc"]
                                best_global_step = global_step
                                best_loss = tr_loss

                                logger.info("Saving trained model at {}...".format(pattern_iter_output_dir))
                                logger.info("best_dev_acc: %.4f | best_global_step: %d" % \
                                            (best_dev_acc, best_global_step))

                                self.save(pattern_iter_output_dir)
                                ######
                                # logger.info("eval_data performance:")
                                # eval_scores = self.eval_dev(eval_data, eval_config, n_gpu)
                                # logger.info(eval_scores)
                            else:
                                early_stop_epoch += 1
                                logger.info(dev_scores)
                                logger.info(early_stop_epoch)

                if 0 < max_steps < global_step or early_stop_epoch >= 10:
                    epoch_iterator.close()
                    break

            if 0 < max_steps < global_step or early_stop_epoch >= 10:
                train_iterator.close()
                break

        return best_global_step, (best_loss / best_global_step if best_global_step > 0 else -1)


    def get_nero_eval(self, gold_label, pred_label, rel2id):
        correct_by_relation = 0.0
        guessed_by_relation = 0.0
        gold_by_relation = 0.0
        for i in range(len(pred_label)):

            if pred_label[i] == rel2id['no relation'] and gold_label[i] == rel2id['no relation']:
                continue
            elif pred_label[i] == rel2id['no relation'] and gold_label[i] != rel2id['no relation']:
                gold_by_relation += 1
            elif pred_label[i] != rel2id['no relation'] and gold_label[i] == rel2id['no relation']:
                guessed_by_relation += 1
            else:
                guessed_by_relation += 1
                gold_by_relation += 1
                if pred_label[i] == gold_label[i]:
                    correct_by_relation += 1

        if guessed_by_relation > 0:
            prec_micro = float(correct_by_relation) / float(guessed_by_relation)
        recall_micro = 0.0
        if gold_by_relation > 0:
            recall_micro = float(correct_by_relation) / float(gold_by_relation)
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        result = {'prec_micro': prec_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro}
        return result

    def eval_dev(self, dev_data, eval_config, n_gpu):
        self.model.eval()
        results = self.eval(dev_data,
                            per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
                            n_gpu=n_gpu)
        predictions = np.argmax(results['logits'], axis=1)
        scores = {}
        metrics = eval_config.metrics if eval_config.metrics else ['acc']
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            elif metric == 'em':
                scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
            elif metric == 'nero':
                eval_result = self.get_nero_eval(results['labels'], predictions, self.rel2id)
                scores['nero_prec'] = eval_result['prec_micro']
                scores['nero_rec'] = eval_result['recall_micro']
                scores['nero_f1'] = eval_result['f1_micro']
            else:
                raise ValueError(f"Metric '{metric}' not implemented")
        return scores



    def eval(self,
             eval_data: List[InputExample],
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1) -> Dict:

        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        eval_losses = [0.0]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.cuda() for k, t in batch.items()} ######
            # batch = {k: t for k, t in batch.items()}
            labels = batch['labels']#1->True
            indices = batch['idx']
            with torch.no_grad():

                logits = self.task_helper.eval_step(batch) if self.task_helper else None
                if logits is None:
                    logits = EVALUATION_STEP_FUNCTIONS[MLM_WRAPPER](self)(batch)#[batch_size, class_num]

                prediction_scores = logits.float().cuda()######
                # prediction_scores = logits.float()
                eval_loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
                eval_losses.append(eval_loss.item())

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)


        return {
            "eval_loss": np.mean(eval_losses),
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

    def eval_words(self, eval_data: List[InputExample],
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1) -> Dict:
        eval_dataset = self._generate_dataset(eval_data, type='eval')
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        eval_losses = [0.0]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.cuda() for k, t in batch.items()} ######
            # batch = {k: t for k, t in batch.items()}
            labels = batch['labels']  # 1->True
            indices = batch['idx']
            with torch.no_grad():
                words_list = self.mlm_eval_step_words(batch)

            if preds is None:
                preds = words_list

            else:
                for i in range(len(preds)):
                    preds[i].extend(words_list[i])

        return preds



    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, type=None):
        features = self._convert_examples_to_features(data, labelled=labelled, type=type)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
            'block_flag': torch.tensor([f.block_flag for f in features], dtype=torch.long)
        }

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)
        return DictDataset(**feature_dict)


    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True, type=None) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled, type=type)
            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
            """
        return features


    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        input_ids = batch['input_ids']
        bz = batch['input_ids'].shape[0]
        block_flag = batch["block_flag"]
        model = self.model.module if hasattr(self.model, 'module') else self.model

        if self.config.model_type == "albert":
            raw_embeds = model.model.albert.embeddings.word_embeddings(input_ids)
        elif self.config.model_type == "bert":
            raw_embeds = model.model.bert.embeddings.word_embeddings(input_ids)
        elif self.config.model_type == "roberta":
            raw_embeds = model.model.roberta.embeddings.word_embeddings(input_ids)
        #[2, 128(emb_size)]
        replace_embeds = model.prompt_embeddings(
            torch.LongTensor(list(range(model.prompt_length))).cuda())#prompt_length=2, torch.LongTensor(list(range(model.prompt_length)))=[0,1]
            # torch.LongTensor(list(range(model.prompt_length))))######
        replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]

        if self.config.prompt_encoder_type == "lstm":
            replace_embeds = model.lstm_head(replace_embeds)[0]  # [batch_size, prompt_len, 2 * hidden_dim]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head(replace_embeds)
            else:
                replace_embeds = model.mlp_head(replace_embeds).squeeze()#[2, 128(emb_size)]

        elif self.config.prompt_encoder_type == "mlp":
            replace_embeds = model.mlp(replace_embeds)
        else:
            raise ValueError("unknown prompt_encoder_type.")
        #分句子的the的下标【[ 1, 35]】
        blocked_indices = (block_flag == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]
        #遍历所有input，raw_embeds是原句子embed
        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):#对每一个句子，把the位置的嵌入改成连续prompt嵌入（有几句话就有几个连续prompt）
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]#把prompt embed插进去

        inputs = {'inputs_embeds': raw_embeds, 'attention_mask': batch['attention_mask']}

        if self.config.model_type in ['bert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs


    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM training step."""
        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        return loss

    def mlm_eval_step_words(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""

        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)#[batch_size, max_seq_length, vocab_size]
        words_id_list = self.preprocessor.pvp.convert_mlm_logits_to_words_id(batch['mlm_labels'], outputs[0])
        words_list = self.preprocessor.pvp.convert_word_ids_to_words(words_id_list, self.tokenizer)
        return words_list#[mask_num, batch_size, topn]

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""

        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

