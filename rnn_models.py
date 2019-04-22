from keras.layers import Input, RNN, Flatten, Dense, Embedding, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping


def create_rnn_model_1():
    model = Sequential()
    model.add(LSTM(10, recurrent_dropout=0, dropout=0))
    model.add(Dense(4,activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def run_rnn_model_1(data, patience = 3, epochs = 100):
    model = create_rnn_model_1()
    es = EarlyStopping(monitor='val_loss', verbose=1, patience = patience)
    history = model.fit(data.train1, data.train_ans, epochs=epochs, validation_split = 0.1, callbacks = [es])
    return model, history