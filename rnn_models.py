from keras.layers import Input, RNN, Flatten, Dense, Embedding, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping


def create_rnn_model_1(input_size):
    model = Sequential()
    model.add(LSTM(4, recurrent_dropout=0, dropout=0))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model


def create_rnn_model_2(input_size):
    model = Sequential()
    model.add(LSTM(12, recurrent_dropout=0, dropout=0))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def create_rnn_model_3(input_size):
    model = Sequential()
    model.add(LSTM(36, recurrent_dropout=0, dropout=0))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def create_rnn_model_4(input_size):
    model = Sequential()
    model.add(LSTM(50, recurrent_dropout=0, dropout=0))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model


