import numpy as np


def convert_to_label_pos(labels):
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
            ans.append((label_B[2:], start_of_label, end_of_label))
    return ans


def prf(gold, pred):
    # gold_labels = 0
    # pred_labels = 0
    # right_labels = 0
    # for i in range(len(gold)):
    #     if gold[i] != 'O' and gold[i] != 'Event':
    #         gold_labels += 1
    #     if pred[i] != 'O' and pred[i] != 'Event':
    #         pred_labels += 1
    #     if gold[i] == pred[i] and gold[i] != 'O' and gold[i] != 'Event':
    #         right_labels += 1
    pred_arg = convert_to_label_pos(pred)
    gold_arg = convert_to_label_pos(gold)
    print(pred_arg)
    print(gold_arg)
    gold_arg_n, pred_arg_n = len(gold_arg), len(pred_arg)
    pred_in_gold_n, gold_in_pred_n = 0, 0
    # pred_in_gold_n
    for argument in pred_arg:
        if argument in gold_arg:
            pred_in_gold_n += 1
    # gold_in_pred_n
    for argument in gold_arg:
        if argument in pred_arg:
            gold_in_pred_n += 1
    p, r, f = 0, 0, 0
    if pred_arg_n != 0: 
        p = 100.0 * pred_in_gold_n / pred_arg_n
    if gold_arg_n != 0: 
        r = 100.0 * gold_in_pred_n / gold_arg_n
    if p or r: 
        f = 2 * p * r / (p + r)
    return p, r, f


def test_prf(pred_file):
    with open(pred_file, 'r') as rf, open('sorted_result.txt', 'w') as wf:
        sentences = rf.read().split('\n\n')
        all_sents = []
        all_tokens = [[], [], []]
        for sent in sentences:
            rows = sent.split('\n')
            if len(rows) == 1:
                continue
            tokens = [r.split(' ')[0] for r in rows]
            golds = [r.split(' ')[1] for r in rows]
            preds = [r.split(' ')[2] for r in rows]
            all_sents.append((tokens, golds, preds))
            all_tokens[0].extend(tokens)
            all_tokens[1].extend(golds)
            all_tokens[2].extend(preds)
        macro_p, macro_r, macro_f = 0, 0, 0
        all_p, all_r, all_f = [], [], []
        for sent in all_sents:
            p, r, f = prf(sent[1], sent[2])
            all_p.append(p)
            all_r.append(r)
            all_f.append(f)
            macro_p += p
            macro_r += r
            macro_f += f
        macro_p /= len(all_sents)
        macro_r /= len(all_sents)
        macro_f /= len(all_sents)
        micro_p, micro_r, micro_f = prf(all_tokens[1], all_tokens[2])
        print("macro: \n\t{}\t{}\t{}\nmicro: \n\t{}\t{}\t{}".format(macro_p, macro_r, macro_f, micro_p, micro_r, micro_f))

        p_sents = zip(all_f, all_sents)
        p_sents = sorted(p_sents, reverse=True)
        for sent in p_sents:
            wf.write('{}\n'.format(sent[0]))
            for t, g, l in zip(sent[1][0], sent[1][1], sent[1][2]):
                wf.write("{}\t{}\t{}\n".format(t, g, l))
            wf.write('\n')



if __name__ == "__main__":
    pred_file = './model_save/test_result/token_labels_.txt'
    test_prf(pred_file)
