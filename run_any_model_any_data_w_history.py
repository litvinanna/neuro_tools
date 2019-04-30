import argparse

parser = argparse.ArgumentParser(description='runs any model with any data')
parser.add_argument('-t', '--type', type = str, help = "dnn, cnn, rnn")
parser.add_argument('-m', '--model', type = int, help = "model number to run")
parser.add_argument('-e', '--enviroment', type= int, help = "enviroment size 6, 12, 24")
parser.add_argument('-p', '--patience', type = int, help = 'patience')
parser.add_argument('-s', '--shift', type = int, default = 0, help="default 0")
parser.add_argument('-ids', type=str, default='ecoli_100000_10000', help="ids list ecoli_100000_10000")


args = parser.parse_args()

net_type = args.type
model_number = args.model
enviroment_size = args.enviroment
patience = args.patience
shift = args.shift
ids = args.ids

import datetime
import time
import os
import pickle

from rnn_models import *
from cnn_models import *
from dnn_models import *
from data_loading import *

def create_f(*args, **kwargs):
    if net_type == "dnn":
        if model_number == 1:
            return create_dnn_model_1(*args, **kwargs)
        elif model_number == 2:
            return create_dnn_model_2(*args, **kwargs)
        
    elif net_type == "cnn":
        if model_number == 1:
            return create_cnn_model_1(*args, **kwargs)
        elif model_number == 2:
            return create_cnn_model_1(*args, **kwargs)
    elif net_type == "rnn":
        if model_number == 1:
            return create_rnn_model_1(*args, **kwargs)
        elif model_number == 2:
            return create_rnn_model_2(*args, **kwargs)



def run_model(create_f, data, patience = 2):
    input_size = data.train1.shape[1]
    model = create_f(input_size =input_size )
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
    history = model.fit(data.train1, data.train_ans, epochs=100, callbacks = [es], validation_data=(data.validate1, data.validate_ans))  
    return model, history


        
times = 30

data_list = generate_data("../results/" + ids, enviroment_size, shift)
name = ids + "_{:02d}_{:02d}".format(enviroment_size, shift)
date = "{:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
path = "../results/{}/{}".format(net_type, date)

if os.path.exists("../results/" + net_type):
    os.mkdir(path)
else:
    print("error")
    
    
    
test_accs = [] 
train_accs= []
t = 0

for i in range(times):
    print(i, t)

    data = data_list[i]
    start = time.time()
    model, history = run_model(create_f, data, patience)
    t += time.time() - start

    acc = model.evaluate(data.test1, data.test_ans, verbose=0)[1]
    acc_train = model.evaluate(data.train1, data.train_ans, verbose=0)[1]

    test_accs.append(acc)
    train_accs.append(acc_train) 
    
    if i == 0:
        model_json = model.to_json()
        with open(os.path.join(path, "model.json"), "w") as json_file:
            json_file.write(model_json)          
          
          
    model.save_weights(os.path.join(path, "{}.weights".format(i)))
    
    with open(os.path.join(path, "{}_history.pyob".format(i)), "wb") as f:
        pickle.dump(history.history, f)
    
    with open(os.path.join(path, "{}_model_{}_all_runs_p{}_{}.pyob".format(net_type, model_number, patience, name)), "wb") as f:
        pickle.dump((test_accs, train_accs), f)