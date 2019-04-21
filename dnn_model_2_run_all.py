from dnn_models import *
import pickle
import time


length = 10


with open("../results/dnn/ecoli_{}_10000_1000.data".format(length), "rb") as file:
    data_list = pickle.load(file)
    
for patience in [1, 2, 3, 4, 5]:
    
    test_accs_1 = [] 
    train_accs_1 = []
    t = 0


    for i in range(30):
        print(i, t)

        data = data_list[i]
        start = time.time()
        model, history = run_model_2(data, patience)
        t += time.time() - start

        acc = model.evaluate(data.test1, data.test_ans, verbose=0)[1]
        acc_train = model.evaluate(data.train1, data.train_ans, verbose=0)[1]

        test_accs_1.append(acc)
        train_accs_1.append(acc_train)
        
    with open("../results/dnn/dnn_model_2_all_runs_{}_p{}.pyob".format(length, patience), "wb") as file:
        pickle.dump((test_accs_1, train_accs_1), file)