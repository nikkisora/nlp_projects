from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import text_rank
import transformers
import sumy_lib
from summarizer import TransformerSummarizer


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

words_per_sentence = 15

xlnetmodel = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
t5_model = transformers.pipeline('summarization', model="t5-base", tokenizer="t5-base", framework="tf")
bart_model = transformers.pipeline('summarization')

@app.post("/summary")
async def generate(text: str = Body(), model: str = Body(), max_words: int = Body()):
    if model == 'textrank':
        s = text_rank.summarize(text, rank_type=model, limit_word=max_words//words_per_sentence)
        ret = {"summary": s}
    elif model == 'topicrank':
        s = text_rank.summarize(text, rank_type=model, limit_word=max_words//words_per_sentence)
        ret = {"summary": s}
    elif model == 'positionrank':
        s = text_rank.summarize(text, rank_type=model, limit_word=max_words//words_per_sentence)
        ret = {"summary": s}
    elif model == 'summatextrank':
        s = text_rank.summarize(text, rank_type=model, limit_word=max_words)
        ret = {"summary": s}
    elif model == 'lexrank':
        s = sumy_lib.summarize(text, algo=model, limit_sent=max_words//words_per_sentence)
        ret = {"summary": s}
    elif model == 'lsa':
        s = sumy_lib.summarize(text, algo=model, limit_sent=max_words//words_per_sentence)
        ret = {"summary": s}
    elif model == 'luhn':
        s = sumy_lib.summarize(text, algo=model, limit_sent=max_words//words_per_sentence)
        ret = {"summary": s}
    elif model == 'kl':
        s = sumy_lib.summarize(text, algo=model, limit_sent=max_words//words_per_sentence)
        ret = {"summary": s}
    elif model == 'bart':
        s = bart_model(text, min_length=10, max_length=max_words)[0]['summary_text']
        # s = 'not enough memory'
        ret = {"summary": s}
    elif model == 'gpt2':
        s = ''.join(GPT2_model(text, min_length=max_words*2, max_length=max_words*3))
        # s = 'not enough memory'
        ret = {"summary": s}
    elif model == 't5':
        s = t5_model(text, min_length=10, max_length=max_words)[0]['summary_text']
        # s = 'not enough memory'
        ret = {"summary": s}
    elif model == 'xlnet':
        s = ''.join(xlnetmodel(text, min_length=max_words*2, max_length=max_words*3))
        # s = 'not enough memory'
        ret = {"summary": s}
    else:
        ret = {"summary": 'wrong model'}
    return ret