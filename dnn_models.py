import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
    

def create_dnn_model_1(input_size):
    x = Input(shape=(input_size, 4))
    y = Flatten()(x)
    z = Dense(4, activation='softmax')(y)
    model = Model(x, z)
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

def run_dnn_model_1(data, patience = 3):
    input_size = data.train1.shape[1]
    model = create_dnn_model_1(input_size)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, callbacks = [es], validation_data=(data.validate1, data.validate_ans))  
    return model, history


def create_dnn_model_2(input_size):
    x = Input(shape=(input_size, 4))
    y = Flatten()(x)
    a = Dense(4)(y)
    z = Dense(4, activation='softmax')(a)
    model = Model(x, z)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run_model(create_f, data, patience = 2):
    input_size = data.train1.shape[1]
    model = create_f(input_size)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, callbacks = [es], validation_data=(data.validate1, data.validate_ans))  
    return model, history

