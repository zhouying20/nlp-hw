# coding=utf-8
import argparse
import json
import re

import torch
from torch import optim
from torch.autograd import Variable
from config import Config
from utils import load_vocab, read_corpus_tri_cls, load_model, save_model, read_corpus_tr_id
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model import BertQA
import tqdm


def train():
    """Train Model"""
    # load config
    config = Config()
    print('settings:\n', config)
    # load corpus
    print('loading corpus')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.tri_cls_label_file)
    # load train and dev and test dataset
    train_data = read_corpus_tri_cls(config.tri_cls_train_file, max_length=config.max_length, vocab=vocab)
    train_ids = torch.LongTensor([temp[0] for temp in train_data])
    train_masks = torch.LongTensor([temp[1] for temp in train_data])
    train_types = torch.LongTensor([temp[2] for temp in train_data])
    train_tags = torch.LongTensor([temp[3] for temp in train_data])
    train_dataset = TensorDataset(train_ids, train_masks, train_types, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    dev_data = read_corpus_tri_cls(config.tri_cls_dev_file, max_length=config.max_length, vocab=vocab)
    dev_ids = torch.LongTensor([temp[0] for temp in dev_data])
    dev_masks = torch.LongTensor([temp[1] for temp in dev_data])
    dev_types = torch.LongTensor([temp[2] for temp in dev_data])
    dev_tags = torch.LongTensor([temp[3] for temp in dev_data])
    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_types, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)
    test_data = read_corpus_tri_cls(config.tri_cls_test_file, max_length=config.max_length, vocab=vocab)
    test_ids = torch.LongTensor([temp[0] for temp in test_data])
    test_masks = torch.LongTensor([temp[1] for temp in test_data])
    test_types = torch.LongTensor([temp[2] for temp in test_data])
    test_tags = torch.LongTensor([temp[3] for temp in test_data])
    test_dataset = TensorDataset(test_ids, test_masks, test_types, test_tags)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size)
    # init model
    model = BertQA(config.bert_path, 2)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_tri_cls_path)
    if config.use_cuda and torch.cuda.is_available():
        model.cuda()
    # train model
    print('begin training')
    model.train()
    optimizer = getattr(optim, config.optim)
    optimizer = optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    eval_loss = 10000
    for epoch in tqdm.tqdm(range(config.base_epoch)):
        step = 0
        bar = tqdm.tqdm(enumerate(train_loader))
        for i, batch in bar:
            step += 1
            model.zero_grad()
            inputs, masks, type_masks, label = batch
            inputs, masks, type_masks, label = Variable(inputs), Variable(masks), Variable(type_masks), Variable(label)
            masks = masks.bool()
            if config.use_cuda and torch.cuda.is_available():
                inputs, masks, type_masks, label = inputs.cuda(), masks.cuda(), type_masks.cuda(), label.cuda()
            feats = model(inputs, masks, type_masks)
            loss = model.loss(feats, label)
            loss.backward()
            optimizer.step()
            tqdm.tqdm.set_description(bar, desc="loss: %f" % loss.item())
        # save best model
        dev_loss_temp = evaluate(model, dev_loader, epoch, config)
        if dev_loss_temp < eval_loss:
            print('dev loss: ', eval_loss, ' -> ', dev_loss_temp)
            eval_loss = dev_loss_temp
            save_model(model, epoch, name='tri-cls--epoch:{}'.format(epoch))
    evaluate(model, test_loader, epoch, config)


def test():
    """Test Model in test file"""
    # load config
    config = Config()
    print('settings:\n', config)
    # load corpus
    print('loading corpus')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.tri_cls_label_file)
    # load train and dev and test dataset
    test_data = read_corpus_tri_cls(config.tri_cls_test_file, max_length=config.max_length, vocab=vocab)
    test_ids = torch.LongTensor([temp[0] for temp in test_data])
    test_masks = torch.LongTensor([temp[1] for temp in test_data])
    test_types = torch.LongTensor([temp[2] for temp in test_data])
    test_tags = torch.LongTensor([temp[3] for temp in test_data])
    test_dataset = TensorDataset(test_ids, test_masks, test_types, test_tags)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size)
    # init model
    model = BertQA(config.bert_path, 2)
    model = load_model(model, name=config.load_tri_cls_path)
    if config.use_cuda and torch.cuda.is_available():
        model.cuda()
    # test model
    evaluate(model, test_loader, 0, config)


def predict(config=None, model=None, sent=None):
    """
    Input: results of trigger identification saved in config.tri_id_result_file or single sentence
    Output: results of trigger classification
            format: [(event type, trigger begin pos, trigger end pos) * num of triggers] * num of sentences
    """
    # load config
    if not config:
        config = Config()
    # load corpus
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.tri_cls_label_file)
    # load trained model
    if not model:
        model = BertQA(config.bert_path, 2)
        model = load_model(model, name=config.load_tri_cls_path)
    if config.use_cuda:
        model.cuda()
    if (not config.input_file) and sent:
        test_datas = read_corpus_tr_id(config.tri_id_result_file, max_length=config.max_length,
                                 label_dic=load_vocab(config.tri_id_label_file), vocab=vocab, content=[sent])
    else:
        # load trigger identification result
        test_datas = read_corpus_tr_id(config.tri_id_result_file, max_length=config.max_length,
                                 label_dic=load_vocab(config.tri_id_label_file), vocab=vocab)
    sent_saves = []
    for i, test_data in tqdm.tqdm(enumerate(test_datas)):
        sent_save = []  # save sentence’s triggers
        sent = test_data[3]
        triggers = test_data[4]
        # predict event type for each trigger
        for trigger in triggers:
            inputs, masks, type_masks = [], [], []
            # data preprocess, [CLS] trigger, [unused0] begin_pos end_pos [unused1], event type [SEP] sentence [SEP]
            for event_type in label_dic.keys():
                tokens_a = []
                for w in sent[trigger['begin_pos']:trigger['end_pos']+1]:
                    tokens_a.append(w.lower())
                tokens_a.extend([',', '[unused0]', str(trigger['begin_pos']), str(trigger['end_pos']), '[unused1]', ','])
                for w in re.split('([:-])', event_type):
                    tokens_a.append(w.lower())
                tokens_b = sent
                if len(tokens_a) + len(tokens_b) > config.max_length - 3:
                    tokens_b = tokens_b[0:(config.max_length - 3 - len(tokens_a))]
                tokens_f = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
                input_mask = [1] * len(input_ids)
                type_mask = [0] * (2 + len(tokens_a)) + [1] * (config.max_length - 2 - len(tokens_a))
                while len(input_ids) < config.max_length:
                    input_ids.append(0)
                    input_mask.append(0)
                inputs.append(input_ids)
                masks.append(input_mask)
                type_masks.append(type_mask)
            inputs, masks, type_masks = Variable(torch.LongTensor(inputs)), \
                                        Variable(torch.LongTensor(masks)), \
                                        Variable(torch.LongTensor(type_masks))
            masks = masks.bool()
            if config.use_cuda and torch.cuda.is_available():
                inputs, masks, type_masks = inputs.cuda(), masks.cuda(), type_masks.cuda()
            # predict event type
            with torch.no_grad():
                feats_1 = model(inputs[:config.batch_size], masks[:config.batch_size], type_masks[:config.batch_size])
                feats_2 = model(inputs[config.batch_size:], masks[config.batch_size:], type_masks[config.batch_size:])
                feats = torch.cat([feats_1, feats_2])
            tag_score = torch.nn.functional.softmax(feats)
            pred_label = torch.argmax(tag_score, dim=0).cpu().numpy().tolist()[1]
            pred_label = list(label_dic.keys())[int(pred_label)]
            # save event type for trigger
            sent_save.append((pred_label, trigger['begin_pos'], trigger['end_pos']))
        sent_saves.append(sent_save)
    # save result
    with open(config.tri_cls_result_file, 'w', encoding='utf-8') as f:
        for sent_save in sent_saves:
            for trigger in sent_save:
                event_type = trigger[0]
                begin_pos = trigger[1]
                end_pos = trigger[2]
                f.write(event_type + ' ' + str(begin_pos) + ' ' + str(end_pos) + ', ')
            f.write('\n')
    # evaluate
    if config.gold_trigger_file:
        evaluate_cls(config.gold_trigger_file)
    return sent_saves


def evaluate_cls(gold_path=None, pred_path=None):
    config = Config()
    if not gold_path:
        gold_path = config.gold_trigger_test_file
    if not pred_path:
        pred_path = config.tri_cls_result_file
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_triggers = f.readlines()
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_triggers = f.readlines()
    gold_num, predict_num, correct_num = 0, 0, 0
    for idx in range(len(gold_triggers)):
        gold_trigger = gold_triggers[idx].strip().split(',')
        gold_trigger = [t.strip().split() for t in gold_trigger if t.strip()]
        gold_num += len(gold_trigger)
        pred_trigger = pred_triggers[idx].strip().split(',')
        pred_trigger = [t.strip().split() for t in pred_trigger if t.strip()]
        predict_num += len(pred_trigger)
        for t in pred_trigger:
            if t in gold_trigger:
                correct_num += 1
    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / gold_num if gold_num else 0
    f_score = (2 * precision * recall) / (precision + recall) if precision + recall else 0
    print('Gold Num: ', gold_num, ' Pred Num: ', predict_num, ' Corr Num: ', correct_num)
    print('Precision: ', precision, ' Recall: ', recall, ' F-score: ', f_score)


def evaluate(model, dev_loader, epoch, config):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    for i, batch in enumerate(dev_loader):
        inputs, masks, type_masks, label = batch
        length += inputs.size(0)
        inputs, masks, type_masks, label = Variable(inputs), Variable(masks), Variable(type_masks), Variable(label)
        masks = masks.bool()
        if config.use_cuda and torch.cuda.is_available():
            inputs, masks, type_masks, label = inputs.cuda(), masks.cuda(), type_masks.cuda(), label.cuda()
        feats = model(inputs, masks, type_masks)
        loss = model.loss(feats, label)
        eval_loss += loss.item()
        tag_score = torch.nn.functional.softmax(feats)
        pred_label = torch.argmax(tag_score, dim=1)
        true_label = label
        pred.extend([label.cpu().numpy().tolist() for label in pred_label])
        true.extend([label.cpu().numpy().tolist() for label in true_label])
    # evaluate model's performace in data loader
    total_num = len(pred)
    correct_num = 0
    for idx in range(len(pred)):
        if pred[idx] == true[idx]:
            correct_num += 1
    accuracy = correct_num / total_num
    print('Total Num: ', total_num, ' Corr Num: ', correct_num, ' Accuracy: ', accuracy)
    print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss/length))
    model.train()
    return eval_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-train', type=bool, default=False,
                        help='Whether to retrain the model.')
    parser.add_argument('--do-eval', type=bool, default=False,
                        help='Whether to perform evaluation.')
    parser.add_argument('--with-golden-trigger', type=bool, default=False,
                        help='Whether to evaluate with golden triggers.')
    args = parser.parse_args()
    if args.do_train:
        train()
    if args.do_eval:
        if not args.with_golden_trigger:
            predict()
        else:
            config = Config()
            config.tri_id_result_file = './data/tri_id_test.txt'
            config.gold_trigger_file = './data/golden_test.txt'
            predict(config)