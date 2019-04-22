import datetime
import os
from rnn_models import *
import pickle
import time


date = "{:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
if os.path.exists("../results/rnn"):
    path = "../results/rnn/{}".format(date)
    os.mkdir(path)
else:
    print("error")

length = 10
patience = 3

with open("../results/dnn/ecoli_{}_10000_1000.data".format(length), "rb") as file:
    data_list = pickle.load(file)
    


          
test_accs = [] 
train_accs= []
t = 0

for i in range(1):
    print(i, t)

    data = data_list[i]
    start = time.time()
    model, history = run_rnn_model_1(data, patience, epochs = 1)
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
    
    with open(os.path.join(path, "{}_history.pyob".format(i)), "wb") as file:
        pickle.dump(history.history, file)
    
    with open(os.path.join(path, "rnn_model_1_all_runs_{}_p{}.pyob".format(length, patience)), "wb") as file:
        pickle.dump((test_accs, train_accs), file)