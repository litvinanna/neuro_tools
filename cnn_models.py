from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Input, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras


def create_cnn_model_1(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(3, kernel_size=3, activation='relu')(inp)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def run_cnn_model_1(data, patience = 1):
    input_size = data.train1.shape[1]
    model = create_cnn_model_1(input_size)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, callbacks = [es], validation_data=(data.validate1, data.validate_ans))    
    return model, history


def create_cnn_model_2(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(3, kernel_size=6, activation='relu')(inp)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model_3(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(3, kernel_size= 3, activation='relu', strides = 3)(inp)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model_4(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(6, kernel_size = 3, activation='relu')(inp)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model_5(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(12, kernel_size = 3, activation='relu')(inp)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_cnn_model_6(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(12, kernel_size= 3, activation='relu', strides = 3)(inp)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_cnn_model_20(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(6, kernel_size=3, activation='relu')(inp)

    inp1 = Input(shape=(input_size, 4))
    x1 = Conv1D(6, kernel_size=3, activation='relu')(inp1)

    y = keras.layers.concatenate([x, x1])
    y = Flatten()(y)
    y = Dense(4, activation='softmax')(y)

    model = Model([inp, inp1], y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_cnn_model_21(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(12, kernel_size=3, activation='relu')(inp)

    inp1 = Input(shape=(input_size, 4))
    x1 = Conv1D(12, kernel_size=3, activation='relu')(inp1)

    y = keras.layers.concatenate([x, x1])
    y = Flatten()(y)
    y = Dense(4, activation='softmax')(y)

    model = Model([inp, inp1], y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_cnn_model_22(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(6, kernel_size=3, activation='relu')(inp)

    inp1 = Input(shape=(input_size, 4))
    x1 = Conv1D(6, kernel_size=3, activation='relu')(inp1)

    y = keras.layers.concatenate([x, x1])
    y = Flatten()(y)
    y = Dense(4, activation='relu')(y)
    y = Dense(4, activation='softmax')(y)

    model = Model([inp, inp1], y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_cnn_model_30(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(6, kernel_size=3, activation='relu')(inp)
    x = LSTM(6, return_sequences=True, return_state = False)(x)
    y = Flatten()(x)
    y = Dense(4, activation='softmax')(y)
    model = Model(inp, y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_cnn_model_31(input_size = 10):
    inp = Input(shape=(input_size, 4))
    x = Conv1D(6, kernel_size=3, activation='relu')(inp)
    x = LSTM(6, return_sequences=True, return_state = False)(x)
    
    inp1 = Input(shape=(input_size, 4))
    x1 = Conv1D(6, kernel_size=3, activation='relu')(inp1)
    x1 = LSTM(6, return_sequences=True, return_state = False)(x1)
    y = keras.layers.concatenate([x, x1])
    y = Flatten()(y)
    y = Dense(4, activation='softmax')(y)
    model = Model([inp, inp1], y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model