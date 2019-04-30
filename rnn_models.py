from keras.layers import Input, RNN, Flatten, Dense, Embedding, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping


def create_rnn_model_1(input_size):
    model = Sequential()
    model.add(LSTM(4, recurrent_dropout=0, dropout=0))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def run_rnn_model_1(data, patience = 3, epochs = 100):
    input_size = data.train1.shape[1]
    model = create_rnn_model_1(input_size)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, callbacks = [es], validation_data=(data.validate1, data.validate_ans))
    return model, history


def create_rnn_model_2(input_size):
    model = Sequential()
    model.add(LSTM(12, recurrent_dropout=0, dropout=0))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model