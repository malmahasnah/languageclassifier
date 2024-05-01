# %%
import numpy as np 
import pandas as pd

# %%
def generate_two_letter_sequences():
    ### Generate every two letter sequence in the alphabet ###
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    sequences = []

    for first_letter in alphabet:
        for second_letter in alphabet:
            sequence = first_letter + second_letter
            sequences.append(sequence)

    return sequences

# %%

def feature(word):
    ### bi-gram feature + some common Spanish ###
    ###         and French preffixes.         ###
    vec = [1]
    spanish_seq = ['dad', 'al', 'de', 'la', 'en', 'el', 'que', 'y', 'lo', 'se', 'dad']
    french_seq = ['ui', 'ei', 'ti', 'ue', 'oi', 'un', 'ie', 'ua', 'rr', 'ee', 'ss', 'eu']
               
    vec += [1 if i in word else 0 for i in spanish_seq]
    vec += [1 if i in word else 0 for i in french_seq]

    spanish_preffixes = ['el', 'ag', 'al', 'an', 'lo']
    french_preffixes = ['de', 'mi', 'mau', 'mi', 'je', 'su', 'do']

    spanish_suffixes = ['or', 'ajo', 'aj', 'aja', 'ana', 'ante', 'blo', 'iego', 'to',
                        'te', 've', 'os']
    french_suffixes = ['at', 're', 'ir', 'ur', 'et', 'ue', 'se', 
                       'le', 'nt']

    vec += [1 if word.startswith(i) else 0 for i in spanish_preffixes]
    vec += [1 if word.startswith(i) else 0 for i in french_preffixes]
    two_letter = generate_two_letter_sequences()
    vec += [1 if i in word else 0 for i in two_letter]

    return vec


# %%
def ohe(y):
    # 1 for spanish, -1 for french
    return np.array([1 if i == 'spanish' else -1 for i in y])

def pred_function(word, w):
    return 'spanish' if np.dot(w, feature(word)) > 0 else 'french'

def simple_model(X, y):
    dsgn_mtrx = np.array([feature(i) for i in X])
    w, residuals, _, _ = np.linalg.lstsq(dsgn_mtrx, ohe(y), rcond=None)
    return w

# %%
def classify(train_words, train_labels, test_words):
    w_optimal = simple_model(train_words, train_labels)
    predictions = []
    for word in test_words:
        predictions.append(pred_function(word, w_optimal))
    return predictions
