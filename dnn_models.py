import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
    

def create_model_1(input_size = 10):
    x = Input(shape=(input_size, 4))
    y = Flatten()(x)
    z = Dense(4, activation='softmax')(y)
    model = Model(x, z)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_model_1(data, patience = 5):
    model = create_model_1()
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, validation_split = 0.1, callbacks = [es])  
    return model, history


def create_model_2(input_size = 10):
    x = Input(shape=(input_size, 4))
    y = Flatten()(x)
    a = Dense(10)(y)
    z = Dense(4, activation='softmax')(a)
    model = Model(x, z)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_model_2(data, patience = 4):
    model = create_model_2()
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, validation_split = 0.1, callbacks = [es])  
    return model, history