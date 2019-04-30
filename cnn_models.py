from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Input
from keras.models import Model
from keras.callbacks import EarlyStopping


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

