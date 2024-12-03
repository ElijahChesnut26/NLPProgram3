#Eli Chesnut, Tom Kerson, Ani Valluru


from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import random

#function gets the vector for a word in a sentence
def get_word_vec(word, sentence, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state.squeeze(0)
    
    # Find the index of the target word in the tokenized input
    word_idx = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]).index(word)
    
    # Get the vector for the target word
    word_vector = hidden_states[word_idx]#can add [:50] to limit the size of the vector
    return word_vector


def get_df_of_word():
    words = ["overtime", "rubbish", "tissue"]
    overtime_sentences_1 = []
    overtime_sentences_2 = []
    pass


def read_file(word):
    


    pass

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

sentences = []