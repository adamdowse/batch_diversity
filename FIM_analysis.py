import tensorflow as tf
import numpy as np 
from keras.utils.layer_utils import count_params 
import time



#Collection of functions that can calculate fisher information matrix statistics
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
def oldGet_Z(model,data_input,y):#TODO
    #returns the z value for a given x and y
    with tf.GradientTape() as tape:
        #NEED SOMETHING HERE TO SAY WHAT WEIGHTS TO USE TODO
        output = model(data_input)
        loss = output[0][y]
    grads = tape.gradient(loss,model.trainable_variables) #all grads 
    #TODO - DO we need just the weights? or all the grads?
    grads = [tf.reshape(g,[-1]) for g in grads] #flatten grads
    grads = tf.concat(grads,0) #concat grads
    grads = tf.math.square(grads) #all grads ^2
    grads = tf.math.reduce_sum(grads) #sum of grads
    #grads = tf.math.sqrt(grads) #sqrt of sum of grads
    return grads

@tf.function
def Get_Z(model,data_input,y):#TODO
    #returns the z value for a given x and y
    with tf.GradientTape() as tape:
        #NEED SOMETHING HERE TO SAY WHAT WEIGHTS TO USE TODO
        output = model(data_input)[0][y]
        loss = tf.math.log(output)
        #loss = tf.math.reduce_mean(output,axis=0)

    grads = tape.gradient(loss,model.trainable_variables) #all grads 
    #select the weights
    grads = [g for g in grads if ('Filter' in g.name) or ('MatMul' in g.name)]
    #TODO - DO we need just the weights? or all the grads?
    grads = [tf.reshape(g,[-1]) for g in grads] #flatten grads
    grads = tf.concat(grads,0) #concat grads
    grads = tf.math.square(grads) #all grads ^2
    grads = tf.math.reduce_sum(grads) #sum of grads
    #grads = grads
    grads = tf.math.sqrt(grads) #sqrt of sum of grads
    return grads




def FIM_trace(data,total_classes,model): 
    #data       = data ittorator like a genorator
    #total_classes = the number of class outputs
    #model      = tf model that returns the logits outputs for all classes
    #calc fim diag
    #t = time.time()
    fim = 0
    data_count = 0
    for b in range(data.num_batches):
        batch_data_input = data.__getitem__(b) [0]
        for data_input in batch_data_input:
            data_count += 1
            if data_count % 1000 == 0:
                print(data_count)
                break
            for y in range(total_classes):
                #calc sum of squared grads for a data point and class square rooted
                z = Get_Z(model,tf.expand_dims(data_input, axis=0),tf.convert_to_tensor(y, dtype=tf.int32))
                fim += z
        if data_count % 1000 == 0:
                print(data_count)
                break
    #think about adding var calc too
    fim /= data_count
    #print('FIM trace calc time:',time.time()-t)

    return fim


def Batch_FIM_trace(data,total_classes,model): 
    #data       = data ittorator like a genorator
    #total_classes = the number of class outputs
    #model      = tf model that returns the logits outputs for all classes
    #calc fim diag
    #t = time.time()
    fim = 0
    data_count = 0
    for b in range(data.num_batches):
        batch_data_input = data.__getitem__(b) [0]
        data_count += batch_data_input.shape[0]
        for y in range(total_classes):
            #calc sum of squared grads for a data point and class square rooted
            z = Get_Z(model,batch_data_input,tf.convert_to_tensor(y, dtype=tf.int32))
            fim += z


    fim /= data_count
    #print('FIM trace calc time:',time.time()-t)

    return fim

def Emperical_FIM_trace(data,total_classes,model): 
    #data       = data ittorator like a genorator
    #total_classes = the number of class outputs
    #model      = tf model that returns the logits outputs for all classes
    #calc fim diag
    #t = time.time()
    fim = 0
    data_count = 0
    for b in range(data.num_batches):
        batch_data_input = data.__getitem__(b)
        x_batch = batch_data_input[0]
        y_batch = batch_data_input[1]
        for x,y in zip(x_batch,y_batch):
            data_count += 1
            if data_count % 10000 == 0:
                print(data_count)
                break

            y = tf.argmax(y)
            #calc sum of squared grads for a data point and class square rooted
            z = Get_Z(model,tf.expand_dims(x,axis=0),tf.convert_to_tensor(y,dtype=tf.int32))
            fim += z
        if data_count % 10000 == 0:
            print(data_count)
            break

    fim /= data_count
    #print('FIM trace calc time:',time.time()-t)
    return fim