from data_preprocess import ace_preprocess
from config import Config
from model import BertLstmCrf, BertQA
from tri_id import predict as tri_id_pre
from tri_cls import predict as tri_cls_pre
from utils import load_model, load_vocab

if __name__ == '__main__':
    config = Config()
    # data preprocess
    ace_preprocess()
    # load model
    label_dic = load_vocab(config.tri_id_label_file)
    tagset_size = len(label_dic)
    model_id = BertLstmCrf(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer,
                        dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    model_id = load_model(model_id, name=config.load_path)
    model_cls = BertQA(config.bert_path, 2)
    model_cls = load_model(model_cls, name=config.load_tri_cls_path)
    # predict
    if not config.input_file:
        while True:
            # get input
            sent = input()
            if sent == 'exit':
                break
            # trigger identification
            pred_label = tri_id_pre(config, model_id, sent)
            for label in pred_label:
                print(label, end=' ')
            print()
            # trigger classification
            input_cls = sent + '|||' + ' '.join(pred_label)
            triggers = tri_cls_pre(config, model_cls, input_cls)[0]
            for trigger in triggers:
                print(trigger[0], str(trigger[1]), str(trigger[2]), end=',')
            print()
    else:
        tri_id_pre(config, model_id)
        tri_cls_pre(config, model_cls)