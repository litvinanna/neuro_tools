from cnn_models import *
import pickle
import time


length = 10


with open("../results/dnn/ecoli_{}_10000_1000.data".format(length), "rb") as file:
    data_list = pickle.load(file)
    
    

for patience in [1, 2]:
    test_accs = [] 
    train_accs = []
    t = 0

    for i in range(30):
        print(i, t)

        data = data_list[i]
        start = time.time()
        model, history = run_cnn_model_1(data, patience)
        t += time.time() - start

        acc = model.evaluate(data.test1, data.test_ans, verbose=0)[1]
        acc_train = model.evaluate(data.train1, data.train_ans, verbose=0)[1]

        test_accs.append(acc)
        train_accs.append(acc_train)

        with open("../results/cnn/cnn_model_1_all_runs_{}_p{}.pyob".format(length, patience), "wb") as file:
            pickle.dump((test_accs, train_accs), file)