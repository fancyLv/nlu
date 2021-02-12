# -*- coding: utf-8 -*-
# @File  : modeling_jointbert.py
# @Author: LVFANGFANG
# @Date  : 2021/2/11 10:05 下午
# @Desc  :

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(2 * config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(2 * config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, tokens_ids, tokens_mask, slot_id, slot_mask, intent_id, context_seq, context_mask):
        outputs = self.bert(input_ids=tokens_ids,
                            attention_mask=tokens_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        context_output = self.bert(input_ids=context_seq, attention_mask=context_mask)[1]
        sequence_output = torch.cat([context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1)], dim=-1)
        pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        if intent_id is not None:
            intent_loss_fct = torch.nn.BCEWithLogitsLoss()
            intent_loss = intent_loss_fct(intent_logits, intent_id)
            total_loss += intent_loss

        if slot_id is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_id, mask=slot_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if slot_mask is not None:
                    active_loss = slot_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_id.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_id.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
