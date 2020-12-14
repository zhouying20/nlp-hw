import spacy


# Prevent edge case where there are sentence breaks in bad places
def seg_sentence(sentence):
    nlp = spacy.load('en_core_web_sm')

    single_tokens = ['sgt.',
                    'sen.',
                    'col.',
                    'brig.',
                    'gen.',
                    'maj.',
                    'sr.',
                    'lt.',
                    'cmdr.',
                    'u.s.',
                    'mr.',
                    'p.o.w.',
                    'u.k.',
                    'u.n.',
                    'ft.',
                    'dr.',
                    'd.c.',
                    'mt.',
                    'st.',
                    'snr.',
                    'rep.',
                    'ms.',
                    'capt.',
                    'sq.',
                    'jr.',
                    'ave.']
    for special_case in single_tokens:
        nlp.tokenizer.add_special_case(special_case, [dict(ORTH=special_case)])
        upped = special_case.upper()
        nlp.tokenizer.add_special_case(upped, [dict(ORTH=upped)])
        capped = special_case.capitalize()
        nlp.tokenizer.add_special_case(capped, [dict(ORTH=capped)])

    doc = nlp(sentence)
    assert doc.text == sentence
    return doc