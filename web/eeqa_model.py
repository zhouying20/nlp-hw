from typing import DefaultDict
import torch
import numpy as np
import random
import json
import os
import collections


from torch.utils.data import DataLoader, TensorDataset
from eeqa.modeling import BertForTriggerClassification, BertForQuestionAnswering
from eeqa.tokenization import BertTokenizer
# from utils.preprocessing import seg_sentence

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

class BertQATrigger(object):
    def __init__(self):
        super().__init__()
        self.model_dir = os.path.join(script_path, "model/qa/trigger5-2139/")
        self.query_instance = ['verb']
        self.max_sent_length = 502  # [CLS] AND [SEP]
        with open(os.path.join(self.model_dir, 'trigger_index_to_category.json'), 'r') as f:
            self.index_to_category = json.load(f)
        self.model = BertForTriggerClassification.from_pretrained(self.model_dir, num_labels=len(self.index_to_category))
        self.model.to(device)
        self.model.eval()

    def preprocess_sentence(self, sentence):
        # sentence = seg_sentence(sentence)
        tokens = []
        segment_ids = []
        in_sentence = []

        # add [CLS]
        tokens.append("[CLS]")
        segment_ids.append(0)
        in_sentence.append(0)

        # add query
        query = self.query_instance
        for (i, token) in enumerate(query):
            sub_tokens = tokenizer.tokenize(token)
            tokens.append(sub_tokens[0])
            segment_ids.append(0)
            in_sentence.append(0)

        # add [SEP]
        tokens.append("[SEP]")
        segment_ids.append(0)
        in_sentence.append(0)

        # add sentence
        for (i, token) in enumerate(sentence):
            sub_tokens = tokenizer.tokenize(token)
            tokens.append(sub_tokens[0])
            segment_ids.append(1)
            in_sentence.append(1)

        # only keep head tokens if sentence length \gt max
        if (len(tokens) > self.max_sent_length):
            tokens = tokens[:self.max_sent_length]
            segment_ids = segment_ids[:self.max_sent_length]
            in_sentence = in_sentence[:self.max_sent_length]

        # add [SEP]
        tokens.append("[SEP]")
        segment_ids.append(1)
        in_sentence.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # print(len(input_ids), category_vocab.max_sent_length)
        assert len(input_ids) == len(segment_ids) == len(in_sentence) == len(input_mask)

        # print("preprocess sentence done:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n".format(tokens, input_ids, segment_ids, in_sentence, input_mask))
        return tokens, input_ids, segment_ids, in_sentence, input_mask

    def infer(self, eval_example):
        # get predictions
        sentence_id, input_ids, segment_ids, in_sentence, input_mask = eval_example

        sentence_id = sentence_id.tolist()
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)
        with torch.no_grad():
            logits = self.model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

        sentence_triggers = []
        for i, in_sent in enumerate(in_sentence):
            logits_i = logits[i].detach().cpu()
            _, tag_seq = torch.max(logits_i, 1)
            tag_seq = tag_seq.tolist()

            decoded_tag_seg = []
            for idj, j in enumerate(in_sent):
                if j:
                    decoded_tag_seg.append(self.index_to_category[str(tag_seq[idj])])
            for offset, tag in enumerate(decoded_tag_seg):
                if tag != "None":
                    sentence_triggers.append([offset, tag])

        return sentence_triggers

    def predict(self, sentence):
        tokens, input_ids, segment_ids, in_sentence, input_mask = self.preprocess_sentence(sentence)

        all_sentence_id = torch.tensor([[1]], dtype=torch.long)
        all_input_ids = torch.tensor([input_ids], dtype=torch.long)
        all_segment_ids = torch.tensor([segment_ids], dtype=torch.long)
        all_in_sentence = torch.tensor([in_sentence], dtype=torch.long)
        all_input_mask = torch.tensor([input_mask], dtype=torch.long)

        eval_example = (all_sentence_id, all_input_ids, all_segment_ids, all_in_sentence, all_input_mask)

        pred = self.infer(eval_example)

        return pred


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens, token_to_orig_map, input_ids, input_mask, segment_ids, if_trigger_ids,
                 #
                 event_type, argument_type, fea_trigger_offset,
                 #
                 start_position=None, end_position=None):

        self.example_id = example_id
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.if_trigger_ids = if_trigger_ids

        self.event_type = event_type
        self.argument_type = argument_type
        self.fea_trigger_offset = fea_trigger_offset


RawResult = collections.namedtuple("RawResult",
                                   ["example_id", "event_type_offset_argument_type", "start_logits", "end_logits"])


class BertQAArgs(object):
    def __init__(self):
        super().__init__()

        template_dir = os.path.join(script_path, "model/qa/question_templates")
        normal_file = os.path.join(template_dir, "arg_queries.csv")
        des_file = os.path.join(template_dir, "description_queries.csv")
        self.query_templates = self.read_query_templates(normal_file, des_file)
        self.nth_query = 5
        self.batch_size = 8
        self.max_answer_length = 3
        self.larger_than_cls = True
        self.n_best_size = 20

        self.model_dir = os.path.join(script_path, "model/qa/args-5-2139/")

        self.max_sent_length = 502  # [CLS] AND [SEP]
        self.model = BertForQuestionAnswering.from_pretrained(self.model_dir)
        self.model.to(device)
        self.model.eval()

    def read_query_templates(self, normal_file, des_file):
        """Load query templates"""
        query_templates = dict()
        with open(normal_file, "r", encoding='utf-8') as f:
            for line in f:
                event_arg, query = line.strip().split(",")
                event_type, arg_name = event_arg.split("_")

                if event_type not in query_templates:
                    query_templates[event_type] = dict()
                if arg_name not in query_templates[event_type]:
                    query_templates[event_type][arg_name] = list()

                # 0 template arg_name
                query_templates[event_type][arg_name].append(arg_name)
                # 1 template arg_name + in trigger (replace [trigger] when forming the instance)
                query_templates[event_type][arg_name].append(arg_name + " in [trigger]")
                # 2 template arg_query
                query_templates[event_type][arg_name].append(query)
                # 3 arg_query + trigger (replace [trigger] when forming the instance)
                query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

        with open(des_file, "r", encoding='utf-8') as f:
            for line in f:
                event_arg, query = line.strip().split(",")
                event_type, arg_name = event_arg.split("_")
                # 4 template des_query
                query_templates[event_type][arg_name].append(query)
                # 5 template des_query + trigger (replace [trigger] when forming the instance)
                query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

        for event_type in query_templates:
            for arg_name in query_templates[event_type]:
                assert len(query_templates[event_type][arg_name]) == 6

        return query_templates

    def preprocess_sentence(self, sentence, events):
        # sentence = seg_sentence(sentence)

        # all_tokens = []
        # all_input_ids = []
        # all_segment_ids = []
        # all_if_trigger_ids = []
        # all_input_mask = []
        # all_token_to_orig_map = []
        features = []

        for event in events:
            trigger_offset = event[0]
            event_type = event[1]
            trigger_token = sentence[trigger_offset]

            for argument_type in self.query_templates[event_type]:
                query = self.query_templates[event_type][argument_type][5]
                query = query.replace("[trigger]", trigger_token)

                tokens = []
                segment_ids = []
                token_to_orig_map = {}

                # add [CLS]
                tokens.append("[CLS]")
                segment_ids.append(0)

                # add query
                query_tokens = tokenizer.tokenize(query)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)

                # add [SEP]
                tokens.append("[SEP]")
                segment_ids.append(0)

                # add sentence
                for (i, token) in enumerate(sentence):
                    token_to_orig_map[len(tokens)] = i
                    sub_tokens = tokenizer.tokenize(token)
                    tokens.append(sub_tokens[0])
                    segment_ids.append(1)

                # only keep head tokens if sentence length \gt max
                if (len(tokens) > self.max_sent_length):
                    tokens = tokens[:self.max_sent_length]
                    segment_ids = segment_ids[:self.max_sent_length]

                # add [SEP]
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # need to padding (query length are diff)
                while len(input_ids) < self.max_sent_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                sentence_offset = len(query_tokens) + 2
                fea_trigger_offset = trigger_offset + sentence_offset

                if_trigger_ids = [0] * len(segment_ids)
                if_trigger_ids[fea_trigger_offset] = 1

                # all_tokens.append(tokens)
                # all_input_ids.append(input_ids)
                # all_segment_ids.append(segment_ids)
                # all_if_trigger_ids.append(if_trigger_ids)
                # all_input_mask.append(input_mask)
                # all_token_to_orig_map.append(token_to_orig_map)
                features.append(InputFeatures(example_id=1, tokens=tokens, token_to_orig_map=token_to_orig_map, 
                input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids, 
                event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset))

                # print(len(input_ids), category_vocab.max_sent_length)
                assert len(input_ids) == len(segment_ids) == len(if_trigger_ids) == len(input_mask)

                # print("preprocess sentence done:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n".format(tokens, input_ids, segment_ids, input_mask, if_trigger_ids))
        return features

    def _get_best_indexes(self, logits, cls_logit=None):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= self.n_best_size:
                break
            if self.larger_than_cls:
                if index_and_score[i][1] < cls_logit:
                    break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def make_predictions(self, all_features, all_results):
        _PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                                ["start_index", "end_index", "start_logit", "end_logit"])
        features = all_features
        results = all_results
        final_predictions = collections.defaultdict(list)

        for (feature_index, feature) in enumerate(features):
            predictions = []
            event_type_argument_type = "_".join([feature.event_type, feature.argument_type])
            event_type_offset_argument_type = "_".join([feature.event_type, str(feature.token_to_orig_map[feature.fea_trigger_offset]), feature.argument_type])

            start_indexes, end_indexes = None, None
            prelim_predictions = []
            for result in results:
                if result.event_type_offset_argument_type == event_type_offset_argument_type:
                    start_indexes = self._get_best_indexes(result.start_logits, result.start_logits[0])
                    end_indexes = self._get_best_indexes(result.end_logits, result.end_logits[0])
                    # add span preds
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            if start_index >= len(feature.tokens) or end_index >= len(feature.tokens):
                                continue
                            if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:
                                continue
                            if end_index < start_index:
                                continue
                            length = end_index - start_index + 1
                            if length > self.max_answer_length:
                                continue
                            prelim_predictions.append(
                                _PrelimPrediction(start_index=start_index, end_index=end_index,
                                                start_logit=result.start_logits[start_index], end_logit=result.end_logits[end_index]))

                    ## add null pred
                    if not self.larger_than_cls:
                        feature_null_score = result.start_logits[0] + result.end_logits[0]
                        prelim_predictions.append(
                            _PrelimPrediction(start_index=0, end_index=0,
                                            start_logit=result.start_logits[0], end_logit=result.end_logits[0]))

                    ## sort
                    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

                    # all_predictions[example_id][event_type_offset_argument_type] = prelim_predictions

                    ## get final pred in format: [event_type_offset_argument_type, [start_offset, end_offset]]
                    max_num_pred_per_arg = 2
                    for idx, pred in enumerate(prelim_predictions):
                        if (idx + 1) > max_num_pred_per_arg: break
                        if pred.start_index == 0 and pred.end_index == 0: break
                        orig_sent_start = feature.token_to_orig_map[pred.start_index]
                        orig_sent_end = feature.token_to_orig_map[pred.end_index]
                        predictions.append([event_type_argument_type, [orig_sent_start, orig_sent_end]])
            final_predictions[feature.event_type].extend(predictions)

        return final_predictions

    def infer(self, eval_dataloader, eval_features):
        all_results = []
        # get predictions
        for idx, (input_ids, input_mask, segment_ids, if_trigger_ids, example_indices) in enumerate(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            if_trigger_ids = if_trigger_ids.to(device)
    
            with torch.no_grad():
                batch_start_logits, batch_end_logits = self.model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                example_id = eval_feature.example_id
                event_type_offset_argument_type = "_".join([eval_feature.event_type, str(eval_feature.token_to_orig_map[eval_feature.fea_trigger_offset]), eval_feature.argument_type])
                all_results.append(RawResult(
                    example_id=example_id, start_logits=start_logits, end_logits=end_logits,
                    event_type_offset_argument_type=event_type_offset_argument_type))

        preds = self.make_predictions(eval_features, all_results)

        return preds


    def predict(self, sentence, events):
        eval_features = self.preprocess_sentence(sentence, events)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_if_trigger_ids = torch.tensor([f.if_trigger_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=self.batch_size)

        pred = self.infer(eval_dataloader, eval_features)

        return pred
