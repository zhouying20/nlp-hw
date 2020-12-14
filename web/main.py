from collections import defaultdict
import json

from flask import Flask, render_template
from flask import request, jsonify
from flask_cors import CORS

from eeqa_model import BertQATrigger, BertQAArgs
from bert_seq_model import BertLstmCrfArgs, BertLstmCrfTriggers
from utils.preprocessing import seg_sentence

app = Flask(__name__)
CORS(app)


qa_trigger = BertQATrigger()
qa_args = BertQAArgs()
seq_trigger = BertLstmCrfTriggers()
seq_args = BertLstmCrfArgs()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


def process_result(tokens, events, args):
    tokens_list = list()
    for t in tokens:
        tokens_list.append(
            [t, 0]
        )
    events_list = list()
    for e in events:
        one_event = {
            'type': e[1],
            'token': tokens[e[0]],
            'arguments': list()
        }
        for arg in args[e[1]]:
            one_arg = {
                'type': arg[0],
                'token': ' '.join(tokens[arg[1][0]:arg[1][1]+1])
            }
            one_event['arguments'].append(one_arg)
        events_list.append(one_event)
        tokens_list[e[0]][1] = 1
        for arg in args[e[1]]:
            start_pos = arg[1][0]
            end_pos = arg[1][1]
            for i in range(start_pos, end_pos+1):
                tokens_list[i][1] = 2

    res = {
        "tokens": tokens_list,
        "events": events_list
    }
    return res


def project_to_token(doc, events, sentence):
    ans = []
    words = sentence.split(' ')
    for event in events:
        idx = event[0]
        offset = len(' '.join(words[:idx])) + 1 # for space
        print(offset)
        for t in doc.to_json()['tokens']:
            if t['start'] == offset:
                ans.append((t['id'], event[1]))
                break
    return ans



@app.route("/api/seqseq", methods=['POST'])
def infer_seq_seq():
    if request.method == 'POST':
        request_json = request.get_json()

        sentence = request_json["sentence"]
        doc = seg_sentence(sentence)
        tokens = [token.text for token in doc]

        events = seq_trigger.predict(sentence)
        events = project_to_token(doc, events, sentence)
        args = seq_args.predict(tokens, events)

        res = process_result(tokens, events, args)
        return jsonify(res)


@app.route("/api/seqqa", methods=['POST'])
def infer_seq_qa():
    if request.method == 'POST':
        request_json = request.get_json()

        sentence = request_json["sentence"]
        doc = seg_sentence(sentence)
        tokens = [token.text for token in doc]

        events = seq_trigger.predict(sentence)
        events = project_to_token(doc, events, sentence)
        args = qa_args.predict(tokens, events)

        res = process_result(tokens, events, args)
        return jsonify(res)


@app.route("/api/qaseq", methods=['POST'])
def infer_qa_seq():
    if request.method == 'POST':
        request_json = request.get_json()

        sentence = request_json["sentence"]
        doc = seg_sentence(sentence)
        tokens = [token.text for token in doc]
        # tokens = sentence.lower().split(' ')
        events = qa_trigger.predict(tokens)
        args = seq_args.predict(tokens, events)

        res = process_result(tokens, events, args)
        return jsonify(res)


@app.route("/api/qaqa", methods=['POST'])
def infer_eeqa():
    if request.method == 'POST':
        request_json = request.get_json()

        sentence = request_json["sentence"]
        doc = seg_sentence(sentence)
        tokens = [token.text for token in doc]
        # tokens = sentence.lower().split(' ')
        events = qa_trigger.predict(tokens)
        args = qa_args.predict(tokens, events)

        res = process_result(tokens, events, args)
        return jsonify(res)