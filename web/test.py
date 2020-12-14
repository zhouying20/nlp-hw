from eeqa_model import BertQATrigger, BertQAArgs
from bert_seq_model import BertLstmCrfArgs, BertLstmCrfTriggers
from utils.preprocessing import seg_sentence


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


if __name__ == "__main__":
    # sentence = "Russian President Vladimir Putin's summit with the leaders of Germany and France may have been a failure that proves there can be no long - term \"peace camp\" alliance following the end of war in Iraq, government sources were quoted as saying at the weekend."
    # sentence = "Prison authorities have given the nod for Anwar to be taken home later in the afternoon to marry his eldest daughter, Nurul Izzah, to engineer Raja Ahmad Sharir Iskandar in a traditional Malay ceremony, he said."
    # sentence = "British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country's energy regulator as the new chairman of finance watchdog the Financial Services Authority(FSA)."
    # sentence = "She is being held on 50,000 dollars bail on a charge of first-degree reckless homicide and hiding a corpse in the death of the infant born in January. "
    # sentence = "Kristin Scott, the mother, told police she gave birth secretly to both babies at her parents' home in Byrds Creek, Richland County, one of unknown sex in April 2001 and the other, a fullterm girl, January 14. "
    # sentence = "Police have arrested four people in connection with the killings. "
    # sentence = "US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward to seeing him at the upcoming summit of major industrialized nations and Russia, the White House said Tuesday."
    sentence = "The Belgrade district court said that Markovic will be tried along with 10 other Milosevic-era officials who face similar charges of 'inappropriate use of state property' that carry a sentence of up to five years in jail."
    doc = seg_sentence(sentence)
    # print(doc.to_json())
    # print(doc[0])
    tokens = [token.text for token in doc]
    # print(tokens)
    # tokens = sentence.lower().split(' ')
    # sent = ' '.join(tokens)


    qa_trigger = BertQATrigger()
    qa_args = BertQAArgs()
    bert_trigger = BertLstmCrfTriggers()
    bert_args = BertLstmCrfArgs()
    events1 = qa_trigger.predict(tokens)
    events2 = bert_trigger.predict(sentence)
    args1 = qa_args.predict(tokens, events1)
    args2 = bert_args.predict(tokens, events1)

    print(tokens)
    print(events1)
    print(project_to_token(doc, events2, sentence))
    print(args1)
    print(args2)
