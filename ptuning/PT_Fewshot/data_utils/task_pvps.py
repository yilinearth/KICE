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
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
from transformers import PreTrainedTokenizer, GPT2Tokenizer
from pet.utils import InputExample, get_verbalization_ids

import log
from pet import wrapper as wrp
import pdb
import numpy as np

logger = log.get_logger('root')

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self, wrapper, pattern_id: int = 0, seed: int = 42, rel2id=None):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)
        self.auto_verbalizer = {}
        self.mask2topn= {
            0 : 3,
            1 : 5,
            2 : 3
        }

        if rel2id is not None:
            self.auto_verbalizer = {}
            for key in rel2id:
                id = str(rel2id[key])
                key = key.replace("of", "").replace("by", "").replace("top", "").replace("/", " ")
                key = key.strip()
                word_list = key.split(" ")
                my_word_list = []
                for i in range(len(word_list)):
                    if len(word_list[i]) != 0:
                        my_word_list.append(word_list[i])
                self.auto_verbalizer[id] = my_word_list

        """
        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)
        """

        ## if self.wrapper.config.wrapper_type in [wrp.MLM_WRAPPER, wrp.PLM_WRAPPER]:

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False, type=None) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer

        parts_a, parts_b, block_flag_a, block_flag_b = self.get_parts(example, type=type)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]#只有句子是True，其他是False
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]#turn word to id

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        # self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)
        num_special = self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length - num_special)#去掉超过最大字符数量的字符

        tokens_a = [token_id for part, _ in parts_a for token_id in part]#part a里的所有token id
        # tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else []

        ### add
        assert len(parts_a) == len(block_flag_a)
        assert len(parts_b) == len(block_flag_b)

        block_flag_a = [flag for (part, _), flag in zip(parts_a, block_flag_a) for _ in part]
        block_flag_b = [flag for (part, _), flag in zip(parts_b, block_flag_b) for _ in part]

        assert len(tokens_a) == len(block_flag_a)
        assert len(tokens_b) == len(block_flag_b)

        if tokens_b:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
            block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a, block_flag_b)
        else:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)
            block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a)#前后加上0和2


        block_flag = [item if item in [0, 1] else 0 for item in block_flag]#把2改成0
        assert len(input_ids) == len(block_flag)

        ### return input_ids, token_type_ids
        return input_ids, token_type_ids, block_flag


    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)


    @abstractmethod
    def get_parts(self, example: InputExample, type=None) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        mask_id = self.wrapper.tokenizer.mask_token_id
        input_list = enumerate(input_ids)
        label_idx_list = [i for i, x in input_list if x == mask_id]
        # label_idx = input_ids.index(mask_id)
        labels = [-1] * len(input_ids)
        for i in range(len(label_idx_list)):
            labels[label_idx_list[i]] = 1

        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:

        masked_logits = logits[mlm_labels >= 0] #[batch_size, vocab_size]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits #[batch_size, label_size]

    def convert_single_mlm_logits_to_words_id(self, logits: torch.Tensor, topn: int) -> torch.Tensor:
        logits = logits.detach().cpu().numpy()
        sort_idx = np.argsort(logits)
        topn_word_idx = sort_idx.tolist()[-1: len(sort_idx)-topn-1: -1]
        return topn_word_idx

    def convert_word_ids_to_words(self, words_id_list, tokenizer):
        words_list = []
        for i in range(len(words_id_list)):
            words_list.append([])
        for mask_id in range(len(words_id_list)):
            for batch_id in range(len(words_id_list[mask_id])):
                topn_list = []
                for word_id in words_id_list[mask_id][batch_id]:
                    word = tokenizer._convert_id_to_token(word_id)
                    topn_list.append(word)
                words_list[mask_id].append(topn_list)
        return words_list

    def convert_mlm_logits_to_words_id(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0] #[batch_size*mask_num, vocab_size]
        labels_list = enumerate(mlm_labels[0])
        mask_num = len([i for i, x in labels_list if x >= 0])
        words_id_list = []
        for i in range(mask_num):
            words_id_list.append([])

        for i, ml in enumerate(masked_logits): # ml:[vocab size]
            mask_idx = i % mask_num
            if mask_num == 1:
                topn = self.mask2topn[1]
            else:
                topn = self.mask2topn[mask_idx]
            word_id_list = self.convert_single_mlm_logits_to_words_id(ml, topn)
            words_id_list[mask_idx].append(word_id_list)

        return words_id_list #[batch_size, mask_size]


    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def convert_plm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == 1
        logits = torch.squeeze(logits, 1)  # remove second dimension as we always have exactly one <mask> per example
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(lgt) for lgt in logits])
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize



class BoolRePVP(PVP):
    VERBALIZER = {
        "False": ["No"],
        "True": ["Yes"]
    }

    def get_parts(self, example: InputExample, type=None) -> FilledPattern:
        example.text_a = example.text_a.strip('.')
        sentence = self.shortenable(example.text_a)
        triple = example.text_b.split("######")
        h = self.shortenable(triple[0])
        t = self.shortenable(triple[1])
        target_rel = self.shortenable(triple[2])

        # few-shot
        if self.pattern_id == 1:

            string_list_a = [sentence, '.', 'the', 'Question: ', 'is', h, target_rel, t, '? Answer: ', self.mask, '.']
            string_list_b = []
            block_flag_a = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            block_flag_b = []
            assert len(string_list_a) == len(block_flag_a)
            assert len(string_list_b) == len(block_flag_b)
            return string_list_a, string_list_b, block_flag_a, block_flag_b

        else:
            raise ValueError("unknown pattern_id.")

    def verbalize(self, label) -> List[str]:
        return BoolRePVP.VERBALIZER[label]

class MultiRePVP(PVP):

    def get_parts(self, example: InputExample, type=None) -> FilledPattern:
        example.text_a = example.text_a.strip('.')
        sentence = self.shortenable(example.text_a)
        triple = example.text_b.split("######")
        h = self.shortenable(triple[0])
        t = self.shortenable(triple[1])

        # few-shot
        if self.pattern_id == 4:
            if type is None:
                string_list_a = [sentence, '.', 'the', h, 'the', self.mask, 'the', t, 'the']
                block_flag_a = [0, 0, 1, 0, 1, 0, 1, 0, 1]
            elif type == 'eval':
                string_list_a = [sentence, '.', 'the', self.mask, h, 'the', self.mask, 'the', self.mask, t, 'the']
                block_flag_a = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

        elif self.pattern_id == 3:
            if type is None:
                string_list_a = [sentence, '.', 'the', h, 'the', self.mask, 'the', t]
                block_flag_a = [0, 0, 1, 0, 1, 0, 1, 0]
            elif type == 'eval':
                string_list_a = [sentence, '.', 'the', self.mask, h, 'the', self.mask, 'the', self.mask, t]
                block_flag_a = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

        elif self.pattern_id == 1:
            if type is None:
                string_list_a = [sentence, '.', 'the', h, 'is', self.mask, 'of', t, '.']
                block_flag_a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif type == 'eval':
                string_list_a = [sentence, '.', 'the', self.mask, h, 'is', self.mask, 'of', self.mask, t, '.']
                block_flag_a = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raise ValueError("unknown pattern_id.")

        string_list_b = []
        block_flag_b = []
        assert len(string_list_a) == len(block_flag_a)
        assert len(string_list_b) == len(block_flag_b)
        return string_list_a, string_list_b, block_flag_a, block_flag_b

    def verbalize(self, label) -> List[str]:
        return self.auto_verbalizer[label]


class MultiConPVP(PVP):

    def get_parts(self, example: InputExample, type=None) -> FilledPattern:
        example.text_a = example.text_a.strip('.')
        ent_name = self.shortenable(example.text_a)
        sentence = example.text_b.strip('.')
        sentence = self.shortenable(sentence)

        # few-shot
        if self.pattern_id == 1:
            string_list_a = [sentence, '.', 'the', ent_name, 'is', 'a', self.mask, '.']
            block_flag_a = [0, 0, 1, 0, 0, 0, 0, 0]

        else:
            raise ValueError("unknown pattern_id.")

        string_list_b = []
        block_flag_b = []
        assert len(string_list_a) == len(block_flag_a)
        assert len(string_list_b) == len(block_flag_b)
        return string_list_a, string_list_b, block_flag_a, block_flag_b

    def verbalize(self, label) -> List[str]:
        return self.auto_verbalizer[label]


class SinRePVP(PVP):

    def get_parts(self, example: InputExample, type=None) -> FilledPattern:
        example.text_a = example.text_a.strip('.')
        sentence = self.shortenable(example.text_a)
        triple = example.text_b.split("######")
        h = self.shortenable(triple[0])
        t = self.shortenable(triple[1])
        h_con = self.shortenable(triple[2])
        t_con = self.shortenable(triple[3])
        if len(h_con[0]) == 0:
            h_con = 'the'
        if len(t_con[0]) == 0:
            t_con = 'the'

        # few-shot
        if self.pattern_id == 4:
                string_list_a = [sentence, '.', 'the', h_con, h, 'the', self.mask, 'the', t_con, t, 'the']
                block_flag_a = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]


        elif self.pattern_id == 3:
                string_list_a = [sentence, '.', 'the', h_con, h, 'the', self.mask, 'the', t_con, t]
                block_flag_a = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]


        elif self.pattern_id == 1:
                string_list_a = [sentence, '.', 'the', h_con, h, 'is', self.mask, 'of', t_con, t, '.']
                block_flag_a = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raise ValueError("unknown pattern_id.")

        string_list_b = []
        block_flag_b = []
        assert len(string_list_a) == len(block_flag_a)
        assert len(string_list_b) == len(block_flag_b)
        return string_list_a, string_list_b, block_flag_a, block_flag_b

    def verbalize(self, label) -> List[str]:
        return self.auto_verbalizer[label]


PVPS = {
    'boolre': BoolRePVP,
    'multire': MultiRePVP,
    'sinre': SinRePVP,
    'multicon': MultiConPVP,
}
