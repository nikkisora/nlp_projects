import transformers
from summarizer import Summarizer, TransformerSummarizer



# summarizer = transformers.pipeline('summarization')
text = open('l5/sample_article.txt', 'r', encoding='utf-8').read()
# t = summarizer(text, min_length=10, max_length=50)[0]['summary_text']
# print(t)

# from transformers import GPT2Tokenizer,GPT2LMHeadModel

GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
# model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full = ''.join(GPT2_model(text, min_length=50, max_length=100))
print(full)









