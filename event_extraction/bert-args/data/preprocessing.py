import json


def convert_to_sequence_labeling(input_file, target_file):
    with open(input_file, 'r') as rf, open(target_file, 'w') as wf:
        for sent in rf:
            row = json.loads(sent)
            sentence, s_start, events = row['sentence'], row['s_start'], row['event']
            if not events:
                continue
            for event in events:
                event_idx = event[0][0] - s_start
                event_type = event[0][1]
                for sub_event in event_type.split('.'):
                    wf.write('{}\t{}\n'.format(sub_event, 'Event'))
                idx2tag = ['O'] * len(sentence)
                idx2tag[event_idx] = 'Event'
                for e in event[1:]:
                    left = e[0] - s_start
                    right = e[1] - s_start
                    idx2tag[left] = 'B-{}'.format(e[2])
                    for i in range(left+1, right+1):
                        idx2tag[i] = 'I-{}'.format(e[2])
                for (idx, token) in enumerate(sentence):
                    wf.write('{}\t{}\n'.format(token, idx2tag[idx]))
                wf.write('\n')


if __name__ == "__main__":
    file_type = 'dev'
    source_file = '/home/zy/data/ace2005-processed/default/' + file_type + '_convert.json'
    target_file = './w_event/' + file_type + '.txt'
    convert_to_sequence_labeling(source_file, target_file)
