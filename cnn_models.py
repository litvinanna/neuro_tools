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
    
    #model.add(Conv1D(3, kernel_size=3, activation='relu', input_shape=(input_size, 4)))
    #model.add(Flatten())
    #model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def run_cnn_model_1(data, patience = 5):
    model = create_cnn_model_1()
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, validation_split = 0.1, callbacks = [es], verbose = 0)  
    return model, history