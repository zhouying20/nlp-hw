import json
import os
import random
import re


def ace_preprocess(baseline=False):
    root_path = 'data/raw_data/'
    save_path = 'data/'
    file_names = ['en-dev.json', 'en-test.json', 'en-train.json']
    tag_set = set()
    max_len = 0
    event_types = []
    with open('data/tri_cls_tag.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            event_types.append(line.strip())
    for file in file_names:
        file_name = file[:-5]
        for t in ['dev', 'test', 'train']:
            if t in file_name:
                file_type = t
        for data_type in ['tri_cls_', 'tri_id_', 'golden_', 'sentence_']:
            if os.path.exists(save_path + data_type + file_type + '.txt'):
                os.remove(save_path + data_type + file_type + '.txt')
        with open(root_path + file, encoding='utf-8') as f:
            sents = json.load(f)
            for sent in sents:
                text = sent['tokens']
                label = ['O'] * len(text)
                events = []

                if max_len < len(text):
                    max_len = len(text)
                for event in sent['golden-event-mentions']:
                    event_type = event['event_type']
                    tag_set.add(event_type)
                    event_begin_index = event['trigger']['position'][0]
                    event_end_index = event['trigger']['position'][1]
                    events.append((event_begin_index, event_end_index, event_type))
                    if baseline:
                        label[event_begin_index] = 'B-' + event_type
                        for i in range(event_begin_index+1, event_end_index+1):
                            label[i] = 'I-' + event_type
                    else:
                        label[event_begin_index] = 'B-TRI'
                        for i in range(event_begin_index+1, event_end_index+1):
                            label[i] = 'I-TRI'
                # get corpus for trigger identification
                with open(save_path + 'tri_id_' + file_type + '.txt', 'a', encoding='utf-8') as f:
                    for i in range(len(text)):
                        f.write(text[i].lower() + ' ')
                    f.write('|||')
                    for i in range(len(label)):
                        f.write(label[i] + ' ')
                    f.write('\n')
                # get corpus for trigger classification
                with open(save_path + 'tri_cls_' + file_type + '.txt', 'a', encoding='utf-8') as f:
                    for event in events:
                        random.shuffle(event_types)
                        for i, t in enumerate(event_types):
                            if not (event_types[i] == event_type or
                                    event_types[(i + 1) % len(event_types)] == event_type or
                                    event_types[(i + 2) % len(event_types)] == event_type or
                                    event_types[(i + 3) % len(event_types)] == event_type or
                                    event_types[(i + 4) % len(event_types)] == event_type or
                                    event_types[(i + 5) % len(event_types)] == event_type or
                                    event_types[(i + 6) % len(event_types)] == event_type or
                                    event_types[(i + 7) % len(event_types)] == event_type or
                                    event_types[(i + 8) % len(event_types)] == event_type or
                                    event_types[(i + 9) % len(event_types)] == event_type or
                                    event_types[(i + 10) % len(event_types)] == event_type or
                                    event_types[(i + 11) % len(event_types)] == event_type or
                                    event_types[(i + 12) % len(event_types)] == event_type or
                                    event_types[(i + 13) % len(event_types)] == event_type or
                                    event_types[(i + 14) % len(event_types)] == event_type or
                                    event_types[(i + 15) % len(event_types)] == event_type or
                                    event_types[(i + 16) % len(event_types)] == event_type or
                                    event_types[(i + 17) % len(event_types)] == event_type or
                                    event_types[(i + 18) % len(event_types)] == event_type or
                                    event_types[(i + 19) % len(event_types)] == event_type or
                                    event_types[(i + 20) % len(event_types)] == event_type or
                                    event_types[(i + 21) % len(event_types)] == event_type or
                                    event_types[(i + 22) % len(event_types)] == event_type or
                                    event_types[(i + 23) % len(event_types)] == event_type or
                                    event_types[(i + 24) % len(event_types)] == event_type or
                                    event_types[(i + 25) % len(event_types)] == event_type or
                                    event_types[(i + 26) % len(event_types)] == event_type or
                                    event_types[(i + 27) % len(event_types)] == event_type or
                                    event_types[(i + 28) % len(event_types)] == event_type or
                                    event_types[(i + 29) % len(event_types)] == event_type or
                                    event_types[(i + 30) % len(event_types)] == event_type or
                                    event_types[(i + 31) % len(event_types)] == event_type or
                                    event_types[(i + 32) % len(event_types)] == event_type
                            ):
                                continue
                            for tt in text[event[0]:event[1]+1]:
                                f.write(tt.lower() + ' ')
                            f.write(', ')
                            f.write('[unused0] ' + str(event[0]) + ' ' + str(event[1]) + ' [unused1] ')
                            f.write(', ')
                            for w in re.split('([:-])', t):
                                f.write(w.lower() + ' ')
                            f.write('|||')
                            for i in range(len(text)):
                                f.write(text[i].lower() + ' ')
                            f.write('|||')
                            if event_type == t:
                                f.write(str(1))
                            else:
                                f.write(str(0))
                            f.write('\n')
                # get golden trigger files
                with open(save_path + 'golden_' + file_type + '.txt', 'a', encoding='utf-8') as f:
                    for event in events:
                        f.write(event[2] + ' ' + str(event[0]) + ' ' + str(event[1]) + ', ')
                    f.write('\n')
                # get sentence
                with open(save_path + 'sentence_' + file_type + '.txt', 'a', encoding='utf-8') as f:
                    f.write(sent['sentence'])
                    f.write('\n')
    # print('max len: ', max_len)
    # get tag file
    # with open(save_path + 'tri_cls_tag.txt', 'w', encoding='utf-8') as f:
    #     for tag in tag_set:
    #         f.write(tag + '\n')
    # with open(save_path + 'tri_id_tag.txt', 'w', encoding='utf-8') as f:
    #     for tag in ['<pad>', 'B-TRI', 'I-TRI', 'O', '<start>' ,'<eos>']:
    #         f.write(tag + '\n')


def sw100_preprocess():
    root_path = 'data/SW100'
    train_samples = []
    for file in os.walk(root_path):
        if os.path.isdir(file[0]):
            file_names = [file[0] + '/' + f for f in file[2] if f.endswith('.txt')]
            for file_name in file_names:
                events = []
                with open(file_name[:-4] + '.ann', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    lines = [l for l in lines if l[0] == 'T']
                    for line in lines:
                        line = line.split(sep='\t')
                        if ';' in line[1]:
                            continue
                        begin_index = int(line[1].split()[1])
                        end_index = int(line[1].split()[2])
                        trigger = line[2].strip()
                        events.append((begin_index, end_index, trigger))
                with open(file_name, 'r', encoding='utf-8') as f:
                    text = f.read()
                text = list(text)
                for event in events:
                    if ''.join(text[event[0]:event[1]]) == event[2]:
                        for i in range(event[0], event[1]):
                            if text[i].strip():
                                text[i] = '*'
                    else:
                        print('no event: ', event)
                text = ''.join(text)
                label_sentences = [l.strip() + '.' for l in text.strip().split('.') if l.strip()]
                label_sentences = [sent.split() for sent in label_sentences]
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                sentences = []
                for line in content:
                    sentence = [l.strip() + '.' for l in line.strip().split('.') if l.strip()]
                    if sentence:
                        sentences.extend(sentence)
                sentences = [sent.split() for sent in sentences]
                for i in range(len(sentences)):
                    sent = sentences[i]
                    if i < len(label_sentences):
                        label = label_sentences[i]
                    else:
                        continue
                    if len(sent) == len(label):
                        for idx, word in enumerate(label):
                            if '*' not in word:
                                label[idx] = 'O'
                            else:
                                label[idx] = 'I-TRI'
                        train_samples.append((sent, label))
    for sample in train_samples:
        for idx, word in enumerate(sample[1]):
            if idx == 0 and sample[1][idx] == 'I-TRI':
                sample[1][idx] = 'B-TRI'
            if idx >= 1 and sample[1][idx-1] == 'O' and sample[1][idx] == 'I-TRI':
                sample[1][idx] = 'B-TRI'
    with open('data/tri_id_train.txt', 'w', encoding='utf-8') as f:
        for sample in train_samples:
            for word in sample[0]:
                f.write(word + ' ')
            f.write('|||')
            for word in sample[1]:
                f.write(word + ' ')
            f.write('\n')


if __name__ == '__main__':
    ace_preprocess()