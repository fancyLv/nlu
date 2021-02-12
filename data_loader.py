# -*- coding: utf-8 -*-
# @File  : data_loader.py
# @Author: LVFANGFANG
# @Date  : 2021/2/11 10:09 下午
# @Desc  :

import copy
import json
import logging
import os

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, guid, tokens, slot_id, intent_id, golden, context):
        self.guid = guid
        self.tokens = tokens
        self.slot_id = slot_id
        self.intent_id = intent_id
        self.golden = golden
        self.context = context

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens_ids, tokens_mask, slot_id, slot_mask, intent_id, context_seq, context_mask):
        self.tokens_ids = tokens_ids
        self.tokens_mask = tokens_mask
        self.slot_id = slot_id
        self.slot_mask = slot_mask
        self.intent_id = intent_id
        self.context_seq = context_seq
        self.context_mask = context_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    def get_examples(self, mode):
        """
        Args:
            mode: train, val, test
        """
        data_path = os.path.join(self.args.data_dir, mode + '_data.json')
        logger.info("LOOKING AT {}".format(data_path))
        data = self._read_file(data_path)
        examples = []
        for i, item in enumerate(data):
            tokens, slot, intent, golden, context = item
            guid = "%s-%s" % (mode, i)
            intent_id = [1 if i in intent else 0 for i in self.intent_labels]
            slot_id = [self.slot_labels.index(s) for s in slot]
            assert len(tokens) == len(slot_id)
            examples.append(InputExample(guid=guid, tokens=tokens, slot_id=slot_id, intent_id=intent_id, golden=golden,
                                         context=context))
        return examples


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    context_max_seq_len = 3 * max_seq_len

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = example.tokens
        slot_id = example.slot_id

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_id = slot_id[:(max_seq_len - special_tokens_count)]
        slot_mask = [1 if mask_padding_with_zero else 0] * len(slot_id)

        # Add [SEP] token
        tokens += [sep_token]
        slot_id += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_id = [pad_token_label_id] + slot_id

        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        tokens_mask = [1 if mask_padding_with_zero else 0] * len(tokens_ids)
        slot_mask = [0] + slot_mask + [0]

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(tokens_ids)
        tokens_ids = tokens_ids + ([pad_token_id] * padding_length)
        tokens_mask = tokens_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        slot_id = slot_id + ([pad_token_label_id] * padding_length)
        slot_mask = slot_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        intent_id = example.intent_id

        context = [" ".join(s[:max_seq_len]) for s in example.context]
        context = tokenizer.encode(cls_token + ' ' + (' ' + sep_token + ' ').join(context)) if context else []
        context_len = len(context)
        context_padding_length = (context_max_seq_len - context_len) if (context_max_seq_len - context_len) > 0 else 0
        context_seq = context[:context_max_seq_len] + [pad_token_id] * context_padding_length
        context_mask = [1 if mask_padding_with_zero else 0] * len(context[:context_max_seq_len])
        context_mask = context_mask + ([0 if mask_padding_with_zero else 1] * context_padding_length)

        assert len(tokens_ids) == max_seq_len, "Error with input length {} vs {}".format(len(tokens_ids), max_seq_len)
        assert len(tokens_mask) == max_seq_len, "Error with tokens mask length {} vs {}".format(len(tokens_mask),
                                                                                                max_seq_len)
        assert len(slot_id) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_id), max_seq_len)
        assert len(slot_mask) == max_seq_len, "Error with slot mask length {} vs {}".format(len(slot_mask), max_seq_len)
        assert len(context_seq) == len(context_mask), "Error with context length {} vs {}".format(len(context_seq),
                                                                                                  len(context_mask))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("tokens_ids: %s" % " ".join([str(x) for x in tokens_ids]))
            logger.info("tokens_mask: %s" % " ".join([str(x) for x in tokens_mask]))
            logger.info("slot_id: %s" % " ".join([str(x) for x in slot_id]))
            logger.info("slot_mask: %s" % " ".join([str(x) for x in slot_mask]))
            logger.info("intent_id: %s" % " ".join([str(x) for x in intent_id]))

        features.append(
            InputFeatures(tokens_ids=tokens_ids,
                          tokens_mask=tokens_mask,
                          slot_id=slot_id,
                          slot_mask=slot_mask,
                          intent_id=intent_id,
                          context_seq=context_seq,
                          context_mask=context_mask
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = JointProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}'.format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "val":
            examples = processor.get_examples("val")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, val, test is available")

        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    tokens_ids = torch.tensor([f.tokens_ids for f in features], dtype=torch.long)
    tokens_mask = torch.tensor([f.tokens_mask for f in features], dtype=torch.long)
    slot_id = torch.tensor([f.slot_id for f in features], dtype=torch.long)
    slot_mask = torch.tensor([f.slot_mask for f in features], dtype=torch.long)
    intent_id = torch.tensor([f.intent_id for f in features], dtype=torch.float)
    context_seq = torch.tensor([f.context_seq for f in features], dtype=torch.long)
    context_mask = torch.tensor([f.context_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(tokens_ids, tokens_mask, slot_id, slot_mask, intent_id, context_seq, context_mask)
    return dataset
