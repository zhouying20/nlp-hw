import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification
import torch


class BertQA(nn.Module):

    def __init__(self, bert_config, target_size):
        super(BertQA, self).__init__()
        self.target_size = target_size

        self.word_embeds = BertForSequenceClassification.from_pretrained(bert_config, num_labels=target_size)

    def forward(self, sentence, attention_mask=None, token_type_ids=None):
        feats = self.word_embeds(sentence, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return feats

    def loss(self, feats, tags):
        loss_value = torch.nn.CrossEntropyLoss()(feats, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
