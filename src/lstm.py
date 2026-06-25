from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
import random
import os

os.makedirs('result', exist_ok=True)

with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

words = text.split()
print("Всего слов в тексте:", len(words))

vocab = sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print("Размер словаря (уникальных слов):", vocab_size)

max_length = 10
step = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, step):
    sentences.append(words[i:i + max_length])
    next_words.append(words[i + max_length])

print("Количество сформированных примеров:", len(sentences))

X = np.zeros((len(sentences), max_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool)

for i, sent in enumerate(sentences):
    for t, w in enumerate(sent):
        X[i, t, word_to_idx[w]] = 1
    y[i, word_to_idx[next_words[i]]] = 1

print("Размер X:", X.shape)
print("Размер y:", y.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(max_length, vocab_size)))
model.add(Dense(vocab_size, activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, diversity):
    start_index = random.randint(0, max(0, len(words) - max_length - 1))
    sentence = words[start_index:start_index + max_length]
    generated = ' '.join(sentence)

    for _ in range(length):
        x_pred = np.zeros((1, max_length, vocab_size), dtype=np.bool)
        for t, w in enumerate(sentence):
            idx = word_to_idx.get(w, 0)
            x_pred[0, t, idx] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_index(preds, diversity)
        next_word = idx_to_word[next_idx]

        generated += ' ' + next_word
        sentence = sentence[1:] + [next_word]

    return generated

print("\nПроцесс обучения обучение...")
model.fit(X, y, batch_size=128, epochs=50, verbose=1)

print("\nГенерация текста...")
generated_text = generate_text(length=2000, diversity=0.5)

with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)

print("Результат сохранён в result/gen.txt")
print("\n=== ФРАГМЕНТ СГЕНЕРИРОВАННОГО ТЕКСТА ===\n")
print(generated_text[:1000])
print("\n... (всего слов:", len(generated_text.split()), ")")
