import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer


algos = {
    'lexrank' : LexRankSummarizer(),
    'lsa' : LsaSummarizer(),
    'luhn' : LuhnSummarizer(),
    'kl' : KLSummarizer(),
}


def summarize(doc: str, algo = 'lexrank', limit_sent=5) -> str:
    parser = PlaintextParser.from_string(doc,Tokenizer('english'))
    summary = algos[algo](parser.document, sentences_count=limit_sent)

    res = []
    for sent in summary:
        sent = str(sent)
        res.append(sent)

    return ' '.join(res)


if __name__ == '__main__':
    # text = open('l5/sample_text.txt', 'r', encoding='utf-8').read()
    # print(summarize(text, 250))

    text = open('l5/sample_article.txt', 'r', encoding='utf-8').read()
    print(summarize(text, limit_sent = 3))
    print('-----------------------------------')
    print(summarize(text, 'lsa', 3))
    print('-----------------------------------')
    print(summarize(text, 'luhn', 3))
    print('-----------------------------------')
    print(summarize(text, 'kl', 3))