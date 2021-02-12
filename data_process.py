# -*- coding: utf-8 -*-
# @File  : data_process.py
# @Author: LVFANGFANG
# @Date  : 2021/2/12 7:14 下午
# @Desc  :

import json
import os
import zipfile
from collections import Counter

from transformers import BertTokenizer


def read_zipped_json(filepath):
    with zipfile.ZipFile(filepath) as myzip:
        for filename in myzip.namelist():
            with myzip.open(filename) as myfile:
                return json.load(myfile)


def preprocess():
    processed_data_dir = './data/joint'
    os.makedirs(processed_data_dir, exist_ok=True)
    all_intent = []
    all_tag = []
    context_size = 3
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    for key in ('train', 'val', 'test'):
        data = read_zipped_json(f'./data/raw/{key}.json.zip')
        processed_data = []

        count = 0
        for item in data.values():
            count += 1
            context = []
            for i, turn in enumerate(item['messages']):
                utterance = turn["content"]
                tokens = tokenizer.tokenize(utterance)
                utterance = ''.join(tokens).replace('##', '')
                tags = ['O'] * len(tokens)
                intents = []
                golden = []
                for intent, domain, slot, value in turn['dialog_act']:
                    if intent in ['Inform', 'Recommend'] and '酒店设施' not in slot:
                        if value in utterance:
                            idx = utterance.index(value)
                            start = len(tokenizer.tokenize(utterance[:idx]))
                            end = start + len(tokenizer.tokenize(value))
                            tag = '+'.join((intent, domain, slot))
                            tags[start: end] = ['B+' + tag] + \
                                               ['I+' + tag] * (end - start - 1)
                            token_v = ''.join(
                                tokens[start: end]).replace('##', '')
                            golden.append([intent, domain, slot, token_v])
                        else:
                            golden.append([intent, domain, slot, value])
                    else:
                        intents.append('+'.join([intent, domain, slot, value]))
                        golden.append([intent, domain, slot, value])
                processed_data.append(
                    [tokens, tags, intents, golden, context[-context_size:]])
                all_intent.extend(intents)
                all_tag.extend(tags)
                context.append(turn['content'])

        all_intent = [x[0] for x in Counter(all_intent).items()]
        all_tag = [x[0] for x in Counter(all_tag).items()]
        print("loaded {}, size {}".format(key, len(processed_data)))
        with open(os.path.join(processed_data_dir, f'{key}_data.json'), 'w') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("sentence label num:", len(all_intent))
    print("tag num:", len(all_tag))
    print(all_intent)
    with open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w') as f1:
        json.dump(all_intent, f1, ensure_ascii=False, indent=2)
    with open(os.path.join(processed_data_dir, 'tag_vocab.json'), 'w') as f2:
        json.dump(all_tag, f2, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    preprocess()
