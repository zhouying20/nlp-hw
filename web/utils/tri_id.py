# coding=utf-8
import torch
from torch.autograd import Variable
from bert_trigger.config import Config
from bert_trigger import BertLstmCrf
import torch.optim as optim
from .tri_utils import load_vocab, read_corpus_tr_id, load_model, save_model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import tqdm
import argparse


def train(config=None):
    """Train Model"""
    # load config
    if not config:
        config = Config()
    print('settings:\n', config)
    # load corpus
    print('loading corpus.')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.tri_id_label_file)
    tagset_size = len(label_dic)
    # load train and dev dataset
    train_data = read_corpus_tr_id(config.tri_id_train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    train_ids = torch.LongTensor([temp[0] for temp in train_data])
    train_masks = torch.LongTensor([temp[1] for temp in train_data])
    train_tags = torch.LongTensor([temp[2] for temp in train_data])
    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_data = read_corpus_tr_id(config.tri_id_dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_ids = torch.LongTensor([temp[0] for temp in dev_data])
    dev_masks = torch.LongTensor([temp[1] for temp in dev_data])
    dev_tags = torch.LongTensor([temp[2] for temp in dev_data])
    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)
    # init model
    model = BertLstmCrf(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()
    # train model
    print('begin training.')
    model.train()
    optimizer = getattr(optim, config.optim)
    optimizer = optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    eval_loss = 10000
    for epoch in tqdm.tqdm(range(config.base_epoch)):
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            model.zero_grad()
            inputs, masks, tags = batch
            inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            masks = masks.bool()
            if config.use_cuda:
                inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
            feats = model(inputs, masks)
            loss = model.loss(feats, masks, tags)
            loss.backward()
            optimizer.step()
        # save best model
        dev_loss_temp = evaluate(model, dev_loader, epoch, config)
        if dev_loss_temp < eval_loss:
            print('dev loss: ', eval_loss, ' -> ', dev_loss_temp)
            eval_loss = dev_loss_temp
            save_model(model, epoch)
    return model


def test(config=None, model=None):
    """Test Model in test file"""
    # load config
    if not config:
        config = Config()
    print('settings:\n', config)
    # load corpus
    print('loading corpus')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.tri_id_label_file)
    tagset_size = len(label_dic)
    # load test dataset
    test_data = read_corpus_tr_id(config.tri_id_test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    test_ids = torch.LongTensor([temp[0] for temp in test_data])
    test_masks = torch.LongTensor([temp[1] for temp in test_data])
    test_tags = torch.LongTensor([temp[2] for temp in test_data])
    test_dataset = TensorDataset(test_ids, test_masks, test_tags)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size)
    # load trained model
    if not model:
        model = BertLstmCrf(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()
    # evaluate model in test file
    print('begin predicting')
    evaluate(model, test_loader, 0, config, True)


def predict(config=None, model=None, sent=None):
    """
    Input: raw sentences saved in config.input_file or sent
    Output: results of trigger identification saved in config.tri_id_result_file
            format: sentence ||| tag (BIO)
    """
    # load config
    if not config:
        config = Config()

    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.tri_id_label_file)
    tagset_size = len(label_dic)
    # load trained model
    if not model:
        model = BertLstmCrf(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer,
                            dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()
    # begin predicting
    if (not config.input_file) and sent:
        # preprocess sent
        sent = sent.lower()
        tokens = sent.split()
        tokens = tokens[0:min(config.max_length-2, len(tokens))]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = torch.LongTensor([[int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]])
        input_masks = torch.LongTensor([[1] * len(input_ids[0])])
        if config.use_cuda and torch.cuda.is_available():
            input_ids, input_masks = input_ids.cuda(), input_masks.cuda()
        # predict tags
        with torch.no_grad():
            feats = model(input_ids, input_masks)
            path_score, best_path = model.crf(feats, input_masks)
        pred_label = best_path[0].cpu().numpy().tolist()
        pred_label = [list(label_dic.keys())[int(x)] for x in pred_label[1:-1]]
        return pred_label
    else:
        with open(config.input_file, 'r', encoding='utf-8') as f:
            sents = f.readlines()
        data = []
        for line in sents:
            line = line.lower()
            tokens = line.split()
            tokens = tokens[0:min(config.max_length - 2, len(tokens))]
            tokens_f = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            input_masks = [1] * len(input_ids)
            while len(input_ids) < config.max_length:
                input_ids.append(0)
                input_masks.append(0)
            data.append((input_ids, input_masks))
        ids = torch.LongTensor([temp[0] for temp in data])
        masks = torch.LongTensor([temp[1] for temp in data])
        dataset = TensorDataset(ids, masks)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size)
        sents = []
        pred = []
        for i, batch in tqdm.tqdm(enumerate(loader)):
            inputs, masks = batch
            inputs, masks = Variable(inputs), Variable(masks)
            masks = masks.bool()

            # save sentences
            for idx in range(inputs.shape[0]):
                sents.append(inputs[idx][masks[idx]].cpu().numpy().tolist())

            # predict labels
            if config.use_cuda:
                inputs, masks = inputs.cuda(), masks.cuda()
            with torch.no_grad():
                feats = model(inputs, masks)
                path_score, best_path = model.crf(feats, masks.byte())

            # save labels
            for idx in range(inputs.shape[0]):
                pred.append(best_path[idx][masks[idx]].cpu().numpy().tolist())
        # save result
        save_results(sents, pred, config)
        return pred


def evaluate(model, data_loader, epoch, config, save_result=False):
    model.eval()
    eval_loss = 0
    true = []  # true labels
    pred = []  # predicted labels
    sents = []  # sentences
    length = 0
    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        masks = masks.bool()

        # save sentences
        for idx in range(inputs.shape[0]):
            sents.append(inputs[idx][masks[idx]].cpu().numpy().tolist())

        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
        feats = model(inputs, masks)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()

        # save labels
        for idx in range(inputs.shape[0]):
            pred.append(best_path[idx][masks[idx]].cpu().numpy().tolist())
            true.append(tags[idx][masks[idx]].cpu().numpy().tolist())
    # evaluate model's performace in data loader
    label_dic = load_vocab(config.tri_id_label_file)
    label_dic = dict(zip(label_dic.values(), label_dic.keys()))
    gold_num, predict_num, correct_num = 0, 0, 0
    for idx, tag in enumerate(pred):
        pred_tag = [label_dic[int(x)] for x in pred[idx]]
        true_tag = [label_dic[int(x)] for x in true[idx]]
        new_gold_num, new_predict_num, new_correct_num = evaluate_sent(true_tag, pred_tag)
        gold_num += new_gold_num
        predict_num += new_predict_num
        correct_num += new_correct_num
    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / gold_num if gold_num else 0
    f_score = (2 * precision * recall) / (precision + recall) if precision + recall else 0
    print('Gold Num: ', gold_num, ' Pred Num: ', predict_num, ' Corr Num: ', correct_num)
    print('Precision: ', precision, ' Recall: ', recall, ' F-score: ', f_score)
    # save result
    if save_result:
        save_results(sents, pred, config)
    # return loss
    print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss/length))
    model.train()
    return eval_loss


def save_results(sents, pred, config, path=None):
    vocab = []
    label_dic = load_vocab(config.tri_id_label_file)
    label_dic = dict(zip(label_dic.values(), label_dic.keys()))
    with open(config.vocab, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab.append(token)
    if not path:
        path = config.tri_id_result_file
    with open(path, 'w', encoding='utf-8') as f:
        readable_sents = []
        readable_preds = []
        for idx in range(len(sents)):
            readable_sents.append([vocab[w] for w in sents[idx][1:-1]])
            readable_preds.append([label_dic[int(x)] for x in pred[idx][1:-1]])
        for idx in range(len(readable_sents)):
            f.write(' '.join(readable_sents[idx]))
            f.write('|||')
            f.write(' '.join(readable_preds[idx]))
            f.write('\n')


def evaluate_sent(true_tag, predict_tag):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    sent_len = len(true_tag)
    start_flag = False
    equal_flag = True

    for i in range(sent_len):
        gold_num = gold_num + 1 if 'B' in true_tag[i] else gold_num
        predict_num = predict_num + 1 if 'B' in predict_tag[i] else predict_num
        if 'B' in true_tag[i]:
            start_flag = True
        if start_flag and true_tag[i] != predict_tag[i]:
            equal_flag = False
        if start_flag and ((i < sent_len - 1 and 'I' not in true_tag[i+1]) or i == sent_len - 1):
            start_flag = False
            if equal_flag and ((i < sent_len - 1 and 'I' not in predict_tag[i+1]) or i == sent_len - 1):
                correct_num += 1
            equal_flag = True
    return gold_num, predict_num, correct_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-train', type=bool, default=False,
                        help='Whether to retrain the model.')
    parser.add_argument('--do-eval', type=bool, default=False,
                        help='Whether to perform evaluation.')
    args = parser.parse_args()
    if args.do_train:
        train()
    if args.do_eval:
        test()








