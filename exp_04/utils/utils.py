import os
import re
import urllib.request
from collections import Counter
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from konlpy.tag import Okt
from konlpy.tag import Mecab
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_data():
    os.chdir('..')
    train_data = pd.read_table(os.path.join(os.getcwd(),'data/ratings_train.txt')
    test_data = pd.read_table(os.path.join(os.getcwd(),'data/ratings_train.txt')

    return train_data, test_data

def load_data(train_data, test_data):
    tokenizer = Mecab()
    stopwords = ['의','가','이','은','들','는','좀','잘','과','도','를','으로','자','에','와','한','하다',",","..."]

    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data = train_data.dropna(how = 'any')
    test_data.drop_duplicates(subset=['document'], inplace=True)
    test_data = test_data.dropna(how = 'any')

    X_train = []
    for sentence in train_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)

    X_test = []
    for sentence in test_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_test.append(temp_X)

    words = np.concatenate(X_train).tolist()
    counter = Counter(words)
    counter = counter.most_common(10000-6)
    vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>','꿀잼','노잼'] + [key for key, _ in counter]

    word_to_index = {word:index for index, word in enumerate(vocab)}

    def wordlist_to_indexlist(wordlist):
        return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in wordlist]

    X_train = list(map(wordlist_to_indexlist, X_train))
    X_test = list(map(wordlist_to_indexlist, X_test))

    index_to_word = {index:word for word, index in word_to_index.items()}

    return X_train, np.array(list(train_data['label'])), X_test, np.array(list(test_data['label'])), word_to_index, index_to_word

def text_discribe(X_train, X_test):
    total_data_text = list(X_train) + list(X_train)
    # 텍스트데이터 문장길이의 리스트를 생성한 후
    num_tokens = [len(tokens) for tokens in total_data_text]
    num_tokens = np.array(num_tokens)
    # 문장길이의 평균값, 최대값, 표준편차를 계산해 본다.
    print('문장길이 평균 : ', np.mean(num_tokens))
    print('문장길이 최대 : ', np.max(num_tokens))
    print('문장길이 표준편차 : ', np.std(num_tokens))

    # 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,
    max_tokens = np.mean(num_tokens) + 1.5 * np.std(num_tokens)
    maxlen = int(max_tokens)
    print('pad_sequences maxlen : ', maxlen)
    print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))

    return maxlen

def get_train_test(train_data, test_data, word_to_index):
    maxlen = text_discribe(train_data, test_data)

    X_train = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_to_index["<PAD>"],
                                                        padding='pre', # 혹은 'pre'
                                                        maxlen=maxlen)

    X_test = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_to_index["<PAD>"],
                                                       padding='pre', # 혹은 'pre'
                                                       maxlen=maxlen)
    return X_train, X_test

def set_validation(X_train, y_train, n = 50000):
    X_val = X_train[:n]
    y_val = y_train[:n]

    partial_X_train = X_train[n:]
    partial_y_train = y_train[n:]

    print(X_val.shape, y_val.shape)
    print(partial_X_train.shape, partial_y_train.shape)

    return partial_X_train, partial_y_train, X_val, y_val

def model(model_name='lstm', vocab_size = 10000, word_vector_dim = 32):
    if model_name == 'lstm':
        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
        model.add(keras.layers.LSTM(2, dropout=0.5, recurrent_dropout = 0.5))
        model.add(keras.layers.Dense(4, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

    elif model_name == 'cnn':
        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))   # trainable을 True로 주면 Fine-tuning
        model.add(keras.layers.Conv1D(16, 3, activation='relu'))
        model.add(keras.layers.Conv1D(16, 3, activation='relu'))
        model.add(keras.layers.MaxPool1D(5))
        model.add(keras.layers.Conv1D(16, 3, activation='relu'))
        model.add(keras.layers.GlobalMaxPooling1D())
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

    elif model_name == 'gmp':
        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
        model.add(keras.layers.GlobalMaxPooling1D())
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

    return model

def plot_metrics(model_history):
    history_dict = model_history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo"는 "파란색 점"입니다
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b는 "파란 실선"입니다
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def train_model(partial_X_train, partial_y_train, X_val, y_val,model,lr=0.001, epochs = 15, batch_size = 512,
                plot_graph=True):

    def scheduler(epoch, lr):
        if epoch < 3:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy'])

    callback = LearningRateScheduler(scheduler)

    model_history = model.fit(partial_X_train, partial_y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val,y_val),
                        verbose=1, callbacks=[callback])

    if plot_graph == True:
        plot_metrics(model_history)

    return model_history

def save_embedding_vector(model, index_to_word, vocab_size = 10000, word_vector_dim = 32, file_name='word2vec_lstm.txt'):
    # 학습한 Embedding 파라미터를 파일에 써서 저장합니다.
    word2vec_file_path = os.path.join(os.getcwd(),'data') + file_name
    f = open(word2vec_file_path, 'w')
    f.write('{} {}\n'.format(vocab_size-4, word_vector_dim))  # 몇개의 벡터를 얼마 사이즈로 기재할지 타이틀을 씁니다.

    # 단어 개수(에서 특수문자 4개는 제외하고)만큼의 워드 벡터를 파일에 기록합니다.
    vectors = model.get_weights()[0]
    for i in range(4,vocab_size):
        f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
    f.close()

def evaluate_embedding_layer(file_name='word2vec_lstm.txt',word = '강추'):
    word2vec_file_path = os.path.join(os.getcwd(),'data') + file_name
    word_vectors = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path, binary=False)
    vector = word_vectors['강추']
    print('vector.shape : \n', vector.shape)
    print('vector : \n',vector)
    print('similar_by_word : \n', word_vectors.similar_by_word(word))
