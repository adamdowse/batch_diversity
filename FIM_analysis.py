import tensorflow as tf
import numpy as np 
from keras.utils.layer_utils import count_params 



#Collection of functions that can calculate fisher information matrix statistics

def FIM_trace(fim):
    return np.prod(fim)

def Get_weights(model,split_layers=False):
    #get the weights from a tensorflow model in a list
    weights = model.trainable_variables
    for i in range(len(weights)):
        weights[i] = weights[i].numpy()
    
    if split_layers:
        #convert weights into a long vector
        weights = [np.ravel(w) for w in weights]
        weights = np.concatenate(weights)
        print(len(weights), 'weights found')
    return weights
    
@tf.function
def Get_Z(model,data_input,y):#TODO
    #returns the z value for a given x and y
    with tf.GradientTape() as tape:
        #NEED SOMETHING HERE TO SAY WHAT WEIGHTS TO USE TODO
        output = model(data_input)
        loss = tf.math.log(output[0][y])
    grads = tape.gradient(loss,model.trainable_variables) #all grads 
    grads = [tf.reshape(g,[-1]) for g in grads] #flatten grads
    grads = tf.concat(grads,0) #concat grads
    grads = tf.math.square(grads) #all grads ^2
    grads = tf.math.reduce_sum(grads) #sum of grads
    grads = tf.math.sqrt(grads) #sqrt of sum of grads
    return grads

def FIM_trace(data,total_classes,model): 
    #data       = data ittorator like a genorator
    #total_classes = the number of class outputs
    #model      = tf model that returns the logits outputs for all classes

    #calc fim diag
    fim = 0
    data_count = 0
    for b in range(data.num_batches):
        batch_data_input = data.__getitem__(b) [0]
        for data_input in batch_data_input:
            data_count += 1
            if data_count % 1000 == 0:
                print(data_count)
            for y in range(total_classes):
                #calc sum of squared grads for a data point and class square rooted
                fim += Get_Z(model,np.expand_dims(data_input, axis=0),y)

    fim /= data_count
    return fim

