from bert_trigger.config import Config
from typing import DefaultDict
import torch
import numpy as np
import random
import json
import os
import collections
import pickle

from torch.utils.data import DataLoader, TensorDataset
from bert_args.models import BERT_BiLSTM_CRF
from pytorch_pretrained_bert import BertTokenizer
from bert_trigger.config import Config
from bert_trigger import BertLstmCrf, BertQA
from utils.tri_id import predict as tri_id_pre
from utils.tri_cls import predict as tri_cls_pre
from utils.tri_utils import load_model, load_vocab

script_path = os.path.abspath(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

bert_vocab = os.path.join(script_path, 'model/bert_vocab.txt')
tokenizer = BertTokenizer.from_pretrained(bert_vocab)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens, token_to_orig_map, input_ids, input_mask, segment_ids,
                 event_type, fea_trigger_offset):

        self.example_id = example_id
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.event_type = event_type
        self.fea_trigger_offset = fea_trigger_offset

class BertLstmCrfArgs(object):
    def __init__(self):
        super().__init__()

        self.max_sent_length = 502
        self.batch_size = 8
        self.rnn_dim = 256
        self.need_birnn = True

        self.model_dir = os.path.join(script_path, "model/bert-args")

        with open(os.path.join(self.model_dir, 'label2id.pkl'), 'rb') as f:
            label2id = pickle.load(f)
            self.id2label = {value:key for key, value in label2id.items()} 

        # self.args = torch.load(os.path.join(self.model_dir, 'training_args.bin'))
        self.model = BERT_BiLSTM_CRF.from_pretrained(self.model_dir, need_birnn=self.need_birnn, rnn_dim=self.rnn_dim)
        self.model.to(device)
        self.model.eval()


    def preprocess_sentence(self, sentence, events):
        features = []

        for event in events:
            trigger_offset = event[0]
            event_type = event[1]
            trigger_token = sentence[trigger_offset]

            query = ' '.join(event_type.split('.'))

            tokens = []
            token_to_orig_map = {}

            # add [CLS]
            tokens.append("[CLS]")

            # add query
            query_tokens = tokenizer.tokenize(query)
            for token in query_tokens:
                tokens.append(token)

            # add sentence
            for (i, token) in enumerate(sentence):
                token_to_orig_map[len(tokens)] = i
                sub_tokens = tokenizer.tokenize(token)
                tokens.append(sub_tokens[0])

            # only keep head tokens if sentence length \gt max
            if (len(tokens) > self.max_sent_length):
                tokens = tokens[:self.max_sent_length]

            # add [SEP]
            tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # need to padding (query length are diff)
            while len(input_ids) < self.max_sent_length:
                input_ids.append(0)
                input_mask.append(0)

            segment_ids = [0] * len(input_ids)

            sentence_offset = len(query_tokens) + 2
            fea_trigger_offset = trigger_offset + sentence_offset


            # print(len(input_ids), category_vocab.max_sent_length)
            assert len(input_ids) == len(input_mask) == len(segment_ids)

            features.append(InputFeatures(example_id=1, tokens=tokens, token_to_orig_map=token_to_orig_map, 
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
            event_type=event_type, fea_trigger_offset=fea_trigger_offset))

            # print("preprocess sentence done:\n\t{}\n\t{}\n\t{}\n\t{}\n".format(tokens, input_ids, segment_ids, input_mask))
        return features
    
    def convert_to_label_pos(self, labels, feature):
        ans = []
        for (idx, label) in enumerate(labels):
            if label.startswith('B-'):
                label_B = label
                label_I = 'I' + label_B[1:]
                start_of_label = idx
                end_of_label = idx
                if end_of_label + 1 < len(labels):
                    next_label = labels[end_of_label+1]
                    while next_label == label_I:
                        end_of_label += 1
                        if end_of_label + 1 < len(labels):
                            next_label = labels[end_of_label+1]
                        else: break

                event_type_argument_type = "_".join([feature.event_type, label_B[2:]])
                ans.append((event_type_argument_type, (feature.token_to_orig_map[start_of_label], feature.token_to_orig_map[end_of_label])))
        return ans

    def infer(self, eval_dataloader, eval_features):
        pred_labels = []
        # get predictions
        for idx, (input_ids, input_mask, segment_ids) in enumerate(eval_dataloader):

            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)

            with torch.no_grad():
                logits = self.model.predict(input_ids, segment_ids, input_mask)

            for l in logits:
                pred_label = []
                for idx in l:
                    pred_label.append(self.id2label[idx])
                pred_labels.append(pred_label)

        sentence_args = dict()
        for (i, feature) in enumerate(eval_features):
            pred_label = pred_labels[i]
            sentence_args[feature.event_type] = self.convert_to_label_pos(pred_label, feature)
        return sentence_args

    def predict(self, sentence, events):
        """
        input
            sentence: list of tokens (preprocessing by spaCy)
            events: list of envents [[trigger_offset, event_type], ......]
        """
        eval_features = self.preprocess_sentence(sentence, events)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=self.batch_size)

        pred = self.infer(eval_dataloader, eval_features)

        return pred


class BertLstmCrfTriggers(object):
    def __init__(self):
        self.config = Config()
        label_dic = load_vocab(self.config.tri_id_label_file)
        tagset_size = len(label_dic)
        self.model_id = BertLstmCrf(self.config.bert_path, tagset_size, self.config.bert_embedding, self.config.rnn_hidden, self.config.rnn_layer,dropout_ratio=self.config.dropout_ratio, dropout1=self.config.dropout1, use_cuda=self.config.use_cuda)
        self.model_id = load_model(self.model_id, path='model/bert-trigger/', name=self.config.load_path)
        self.model_cls = BertQA(self.config.bert_path, 2)
        self.model_cls = load_model(self.model_cls, path='model/bert-trigger/', name=self.config.load_tri_cls_path)
    
    def predict(self, sentence):
        pred_label = tri_id_pre(self.config, self.model_id, sentence)
        # trigger classification
        # sentence = ' '.join(tokens)
        input_cls = sentence + '|||' + ' '.join(pred_label)
        triggers = tri_cls_pre(self.config, self.model_cls, input_cls)[0]
        ans = []
        for trigger in triggers:
            ans.append((trigger[1], trigger[0].replace(':', '.')))
        return ans
