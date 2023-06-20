import tensorflow as tf
import numpy as np 
from keras.utils.layer_utils import count_params 
import time
from scipy.spatial.distance import cdist



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
def Get_Z(model,data_input):#TODO
    #returns the z value for a given x and y
    with tf.GradientTape() as tape:
        output = model(data_input,training=False)[0]
        #sample from the output distribution
        output = output[0,tf.random.categorical(tf.math.log(output), 1)[0][0]]
        loss = tf.math.log(output)

    grads = tape.gradient(loss,model.trainable_variables) #all grads 
    #select the weights
    grads = [g for g in grads if ('Filter' in g.name) or ('MatMul' in g.name)]
    grads = [tf.reshape(g,[-1]) for g in grads] #flatten grads
    grads = tf.concat(grads,0) #concat grads
    grads = tf.math.square(grads) #all grads ^2
    grads = tf.math.reduce_sum(grads) #sum of grads
    #grads = tf.math.sqrt(grads) #sqrt of sum of grads
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
            if data_count % 10000 == 0:
                print(data_count)
                break
            #calc sum of squared grads for a data point and class square rooted
            z = Get_Z(model,tf.expand_dims(data_input, axis=0))
            fim += z
        if data_count % 10000 == 0:
                print(data_count)
                break
    #think about adding var calc too
    fim /= data_count
    #print('FIM trace calc time:',time.time()-t)
    return 
    
def FIM_trace_with_var(data,total_classes,model): 
    #data       = data ittorator like a genorator
    #total_classes = the number of class outputs
    #model      = tf model that returns the logits outputs for all classes
    #calc fim diag
    #t = time.time()
    data_count = 0
    msq = 0
    for b in range(data.num_batches):
        batch_data_input = data.__getitem__(b) [0]
        for data_input in batch_data_input:
            data_count += 1
            #calc sum of squared grads for a data point and class square rooted
            z = Get_Z(model,tf.expand_dims(data_input, axis=0))
            if data_count == 1:
                mean = z
            delta = z - mean
            mean += delta / (data_count+1) #Welford_cpp from web
            msq += delta * (z - mean)

        if data_count % 10000 == 0:
            print(data_count)
            break

    return mean, msq/(data_count-1)



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

@tf.function
def Get_EZ(model,x,y):#TODO
    
    #returns the z value for a given x and y
    with tf.GradientTape() as tape:
        y_arg = tf.squeeze(tf.argmax(y,axis=1))
        output = model(x,training=False)[0]
        output = output[0,y_arg]
        output = tf.math.log(output)

    grads = tape.gradient(output,model.trainable_variables) #all grads 
    #select the weights
    grads = [g for g in grads if ('Filter' in g.name) or ('MatMul' in g.name)]
    grads = [tf.reshape(g,[-1]) for g in grads] #flatten grads
    grads = tf.concat(grads,0) #concat grads
    grads = tf.math.square(grads) #all grads ^2
    grads = tf.math.reduce_sum(grads) #sum of grads
    #grads = tf.math.sqrt(grads) #sqrt of sum of grads
    return grads

#CAN COMBINE WITH NORMAL REALISTICALLY
def Emperical_FIM_trace_with_var(data,total_classes,model): 
    #data       = data ittorator like a genorator
    #total_classes = the number of class outputs
    #model      = tf model that returns the logits outputs for all classes
    #calc fim diag
    #t = time.time()
    data_count = 0
    msq = 0
    for b in range(data.num_batches):
        batch_data_input = data.__getitem__(b)
        x_batch = batch_data_input[0]
        y_batch = batch_data_input[1]
        for x,y in zip(x_batch,y_batch):
            data_count += 1
            #calc sum of squared grads for a data point and class square rooted
            z = Get_EZ(model,tf.expand_dims(x,axis=0),tf.expand_dims(y,axis=0))
            if data_count == 1:
                mean = z
            delta = z - mean
            mean += delta / (data_count + 1)
            msq += delta * (z - mean)
            
        if data_count % 10000 == 0:
            print(data_count)
            break

    return mean, msq/(data_count - 1)

@tf.function
def Get_raw_grads(model,x,y):#TODO
    #returns the z value for a given x and y
    lf = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    with tf.GradientTape() as tape:
        output = model(x,training=False)[0]
        loss = lf(y,output)
    grads = tape.gradient(loss,model.trainable_variables) #all grads 
    grads = [g for g in grads if ('Filter' in g.name) or ('MatMul' in g.name)]
    grads = [tf.reshape(g,[-1]) for g in grads] #flatten grads
    grads = tf.concat(grads,0) #concat grads
    return grads

def Grad_Div_Ensemble_Method(data,model):
    #take some batches of fixed size and show variance between them
    batch_count = 0
    max_batch_count = 30
    while (batch_count < data.num_batches)and(batch_count < max_batch_count):
        batch_count+=1
        batch_data = data.__getitem__(batch_count)
        if batch_count == 1:
            gs = np.expand_dims(Get_raw_grads(model,batch_data[0],batch_data[1]),axis=0)
        else:
            gs = np.append(gs,np.expand_dims(Get_raw_grads(model,batch_data[0],batch_data[1]),axis=0),axis=0)
        print (gs.shape)
    
    #batch grad mean
    g_mean = np.mean(gs,axis=0)
    print(g_mean)
    g_cos_to_mean = np.mean(cdist([g_mean],gs,'Cosine'))
    
    return np.sqrt(np.sum(np.power(g_mean, 2))) , g_cos_to_mean

