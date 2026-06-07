from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import random
import sys
import os

os.makedirs('../result', exist_ok=True)

with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print("Длина загруженного текста (символов):", len(text))

vocabulary = sorted(list(set(text)))
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))
print("Размер алфавита (уникальных символов):", len(vocabulary))

max_length = 40
steps = 1
sentences = []
next_chars = []

for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i:i + max_length])
    next_chars.append(text[i + max_length])

print("Количество сформированных цепочек:", len(sentences))

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1

print("Размер X:", X.shape)
print("Размер y:", y.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, diversity):
    start_index = random.randint(0, max(0, len(text) - max_length - 1))
    generated = ''
    sentence = text[start_index:start_index + max_length]
    generated += sentence
    
    for i in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, char in enumerate(sentence):
            idx = char_to_indices.get(char, 0)
            x_pred[0, t, idx] = 1.
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_char = indices_to_char[next_index]
        
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

print("\nНачинаем обучение...")
model.fit(X, y, batch_size=128, epochs=50, verbose=1) 

print("\nГенерация текста...")

generated_text = generate_text(5000, diversity=0.5)

with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)

print("Результат сохранён в ../result/gen.txt")
print("\n=== ФРАГМЕНТ СГЕНЕРИРОВАННОГО ТЕКСТА ===\n")
print(generated_text[:1000])
print("\n... (всего", len(generated_text), "символов)")
