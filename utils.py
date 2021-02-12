# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: LVFANGFANG
# @Date  : 2021/2/11 10:08 下午
# @Desc  :

import json
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer


def get_intent_labels(args):
    with open(os.path.join(args.data_dir, args.intent_label_file)) as f:
        intent_labels = json.load(f)
    return intent_labels


def get_slot_labels(args):
    with open(os.path.join(args.data_dir, args.slot_label_file)) as f:
        slot_labels = json.load(f)
    return slot_labels


def load_tokenizer(args):
    return BertTokenizer.from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    results = {}
    intent_result = get_intent_metrics(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "intent_precision": precision_score(labels, preds),
        "intent_recall": recall_score(labels, preds),
        "intent_f1": f1_score(labels, preds)
    }


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    intent_result = (np.array(intent_preds) == np.array(intent_labels)).all(axis=1)

    slot_result = (np.array(slot_preds) == np.array(slot_labels)).all(axis=1)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }
