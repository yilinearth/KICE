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
This file contains the logic for loading data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

import log
from pet import task_helpers
from pet.utils import InputExample


logger = log.get_logger('root')

def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading train/dev32/dev/test/unlabeled examples for a given task.
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_dev32_examples(self, data_dir) -> List[InputExample]:
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir, target_rel=None) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass



class MultiReProcessor(DataProcessor):

    def __init__(self, rel2id):
        self.label_list = []
        for key in rel2id:
            self.label_list.append(str(rel2id[key]))
        self.rel2id = rel2id


    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.json"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.json"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_unlabeled_examples(self, data_dir, target_rel=None):
        return self._create_examples(data_dir, "unlabeled")

    def get_labels(self):
        return self.label_list

    def _create_examples(self, path: str, set_type: str, target_rel=None) -> List[InputExample]:
        examples = []
        idx = 0
        with open(path, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                rel_obj = json.loads(line)
                relation = rel_obj['relation']
                label = str(self.rel2id[relation])

                guid = "%s-%s" % (set_type, idx)

                text_a = " ".join(rel_obj['token'])

                text_b = rel_obj["h"]["name"] + "######" + rel_obj["t"]["name"]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
                idx += 1

        return examples

class SinReProcessor(DataProcessor):

    def __init__(self, rel2id):
        self.label_list = []
        for key in rel2id:
            self.label_list.append(str(rel2id[key]))
        self.rel2id = rel2id


    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.json"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.json"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_unlabeled_examples(self, data_dir, target_rel=None):
        return self._create_examples(data_dir, "unlabeled")

    def get_labels(self):
        return self.label_list

    def _create_examples(self, path: str, set_type: str, target_rel=None) -> List[InputExample]:
        examples = []
        idx = 0
        with open(path, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                rel_obj = json.loads(line)
                relation = rel_obj['relation']
                label = str(self.rel2id[relation])

                guid = "%s-%s" % (set_type, idx)

                text_a = " ".join(rel_obj['token'])
                if 'concept' not in rel_obj["h"] and 'mask_0' in rel_obj:
                    rel_obj['h']['concept'] = rel_obj['mask_0'][0]
                    rel_obj['t']['concept'] = rel_obj['mask_2'][0]
                text_b = rel_obj["h"]["name"] + "######" + rel_obj["t"]["name"] + "######" + rel_obj["h"]["concept"] + "######" + rel_obj["t"]["concept"]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
                idx += 1

        return examples

class MultiConProcessor(DataProcessor):

    def __init__(self, con2id):
        self.label_list = []
        for key in con2id:
            self.label_list.append(str(con2id[key]))
        self.con2id = con2id


    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.json"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.json"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.json"), "dev")

    def get_unlabeled_examples(self, data_dir, target_rel=None):
        return self._create_examples(data_dir, "unlabeled")

    def get_labels(self):
        return self.label_list

    def _create_examples(self, path: str, set_type: str, target_rel=None) -> List[InputExample]:
        examples = []
        idx = 0
        with open(path, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                con_obj = json.loads(line)

                guid = "%s-%s" % (set_type, idx)

                text_h = con_obj['h']['name']
                text_t = con_obj['t']['name']
                sent = " ".join(con_obj['token'])


                if set_type == 'unlabeled' or con_obj['h']['concept'] in self.con2id:
                    h_con_label = None if set_type == 'unlabeled' else str(self.con2id[con_obj['h']['concept']])
                    example = InputExample(guid=guid, text_a=text_h, text_b=sent, label=h_con_label, idx=idx)
                    examples.append(example)
                    idx += 1
                if set_type == 'unlabeled' or con_obj['t']['concept'] in self.con2id:
                    t_con_label = None if set_type == 'unlabeled' else str(self.con2id[con_obj['t']['concept']])
                    example = InputExample(guid=guid, text_a=text_t, text_b=sent, label=t_con_label, idx=idx)
                    examples.append(example)
                    idx += 1

        return examples



class BoolReProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.txt"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.txt"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.txt"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.txt"), "dev")

    def get_unlabeled_examples(self, data_dir, target_rel=None):
        return self._create_examples(data_dir, "unlabeled", target_rel=target_rel)

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str, target_rel=None) -> List[InputExample]:
        examples = []
        idx = 0
        with open(path, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                rel_obj = json.loads(line)
                label = str(rel_obj['label']) if 'label' in rel_obj else None

                guid = "%s-%s" % (set_type, idx)

                text_a = " ".join(rel_obj['token'])
                if set_type != 'unlabeled':
                    text_b = rel_obj["h"]["name"] + "######" + rel_obj["t"]["name"] + "######" + rel_obj["target_rel"]
                else:
                    text_b = rel_obj["h"]["name"] + "######" + rel_obj["t"]["name"] + "######" + target_rel

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)
                idx += 1

        return examples


PROCESSORS = {
    "multire": MultiReProcessor,
    "sinre": SinReProcessor,
    "boolre": BoolReProcessor,
    "multicon": MultiConProcessor,
}  # type: Dict[str,Callable[[],DataProcessor]]


TASK_HELPERS = {}

METRICS = {
    "boolre": ["acc", "f1"],
    "multire": ["acc", "f1-macro"],
    "sinre": ["acc", "f1-macro", "nero"],
    "multicon": ["acc", "f1-macro"],
}


DEFAULT_METRICS = ["acc"]


TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"
DEV32_SET = "dev32"


SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET, DEV32_SET]


def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42, target_rel=None, rel2id=None) -> List[InputExample]:
    """Load examples for a given task."""

    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"

    if rel2id is not None:
        processor = PROCESSORS[task](rel2id)
    else:
        processor = PROCESSORS[task]()

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == DEV32_SET: ### TODO
        examples = processor.get_dev32_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir, target_rel=target_rel)
        for example in examples:
            example.label = processor.get_labels()[0]
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")

    if num_examples is not None and set_type == TRAIN_SET:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples
