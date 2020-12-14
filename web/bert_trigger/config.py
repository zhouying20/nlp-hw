# coding=utf-8


class Config(object):
    def __init__(self):
        # self.input_file = './data/sample.txt'
        # self.gold_trigger_file = './data/golden_test.txt'
        self.input_file = None
        self.gold_trigger_file = None

        self.tri_id_label_file = './data/tri_id_tag.txt'
        self.tri_id_train_file = './data/tri_id_train.txt'
        self.tri_id_dev_file = './data/tri_id_dev.txt'
        self.tri_id_test_file = './data/tri_id_test.txt'
        self.tri_id_result_file = './data/tri_id_result.txt'
        self.tri_cls_train_file = './data/tri_cls_train.txt'
        self.tri_cls_dev_file = './data/tri_cls_dev.txt'
        self.tri_cls_test_file = './data/tri_cls_test.txt'
        self.tri_cls_label_file = './data/tri_cls_tag.txt'
        self.tri_cls_result_file = './data/tr_cls_result.txt'
        self.gold_trigger_test_file = './data/golden_test.txt'
        self.vocab = './data/bert/vocab.txt'

        self.max_length = 200
        self.use_cuda = True
        self.gpu = 1
        self.batch_size = 16
        self.bert_path = '/home/zy/.pytorch_pretrained_bert/bert-base-uncased'
        self.rnn_hidden = 512
        self.bert_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 1e-5
        self.lr_decay = 1e-5
        self.weight_decay = 0.00005

        self.checkpoint = 'model/bert-trigger/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = 'tri_id'
        self.load_tri_cls_path = 'tri_cls_1:24'
        self.base_epoch = 100

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
