import tensorflow as tf
import numpy as np 



#Collection of functions that can calculate fisher information matrix statistics

def FIM_trace(fim):
    return np.prod(fim)

def Get_weights(model):
    #get the weights from a tensorflow model in a list
    


def FIM_diag(weights,data,total_classes,model):
    #weights    = list of weights [1xn] 
    #data       = data ittorator like a genorator
    #total_classes = the number of class outputs
    #model      = tf model that returns the logits outputs for all classes

    fim = np.zeros(len(weights))
    data_count = 0
    for d in data:
        data_count += 1
        #get the log output from the model
        output = model.predict(d) #output should be all classes
        for k in range(total_classes):
            for i in range(len(weights)):
                fim[i] += (weights[i]*output[k])^2

    fim /= data_count
    return fim

