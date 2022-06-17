import spacy
import pytextrank
from summa import summarizer


def _get_rank(name):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(name,
                config={
                    'stopwords':{
                        'word':['NOUN']
                    }
                })
    return nlp


algos = {
    'textrank' : _get_rank('textrank'),
    'topicrank' : _get_rank('topicrank'),
    'positionrank' : _get_rank('positionrank'),
}


def summarize(doc: str, rank_type = 'textrank', limit_word=3) -> str:
    if rank_type == 'summatextrank':
        return summarizer.summarize(doc, words=limit_word)
    doc = algos[rank_type](doc)
    tr = doc._.textrank

    res = []
    t_len = 0
    for sent in tr.summary(limit_phrases=15, limit_sentences=limit_word):
        res.append(sent.text)

    return ' '.join(res)


if __name__ == '__main__':
    # text = open('l5/sample_text.txt', 'r', encoding='utf-8').read()
    # print(summarize(text, 250))

    text = open('l5/sample_article.txt', 'r', encoding='utf-8').read()
    print(summarize(text, limit_word = 50))
    print('-----------------------------------')
    print(summarize(text, 'topicrank', 50))
    print('-----------------------------------')
    print(summarize(text, 'positionrank', 50))
    print('-----------------------------------')
    print(summarize(text, 'summatextrank', 50))