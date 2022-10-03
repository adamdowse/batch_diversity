#collection of helpper functions
import multiprocessing
from multiprocessing import Pool
from typing import no_type_check_decorator
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import backend as K
from keras.models import Model
import sqlite3
from sqlite3 import Error
import numpy as np
import random
import io
import os
from os import getpid
import wandb
import supporting_models as sm
import scipy
from scipy.spatial.distance import cdist
import time



#The train data generator
class CustomDBDataGen(tf.keras.utils.Sequence):
    def __init__(self, conn,
                 batch_size,
                 num_classes,
                 num_images,
                 warm_start_batches=1
                 ):
        #INIT:
        #randomly batch for the first epoch and update the db

        #init db connection and init vars
        self.conn = conn
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_images = num_images

        #dealing with warm starts
        if warm_start_batches != 0:
            self.num_batches = warm_start_batches
            print('There are ',self.num_batches,' batches in the warm start')

            arr = []
            for n in range(self.num_batches):
                to_add = [n]*batch_size
                arr = arr + to_add
            
            arr = arr[:num_images] #Limit
            random.shuffle(arr) #shuffle

            #update the database
            curr = conn.cursor()
            for i, a in enumerate(arr):
                curr.execute('''UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(a),int(i),))
            conn.commit()
        else:   
            print('There are 0 batches in the warm start')
            curr = conn.cursor()
            curr.execute('''UPDATE imgs SET batch_num = -1''')
            conn.commit()

        
    def ret_batch_info(self):
        return self.num_batches

    def on_epoch_end(self):
        #reset the batch numbers to -1 for all images
        curr = self.conn.cursor()
        curr.execute('''UPDATE imgs SET batch_num = -1''')
        self.conn.commit()

    def __getitem__(self, index):
        #select only the batch where batch = index
        curr = self.conn.cursor()
        curr.execute('''SELECT id,data, label_num FROM imgs WHERE batch_num = (?)''',(int(index),))
        ids = []
        imgs = []
        labels = []
        for id,img, label in curr:
            ids.append(id)
            imgs.append(img)
            labels.append(label)

        #convert to tensors
        ids = np.array(ids)
        ids = tf.cast(ids, 'int32')

        imgs = np.array(imgs)
        imgs = tf.cast(imgs,'float32')

        labels = np.array(labels)
        labels = tf.one_hot(labels,self.num_classes)

        return tuple([ids,imgs]), labels

    def __len__(self):
        #calculates the number of batches to use
        curr = self.conn.cursor()
        curr.execute('''SELECT MAX(batch_num) FROM imgs''')
        return curr.fetchone()[0] + 1

def setup_db(config):

    def DB_add_img(conn, img):
        """
        Create a new img into the img table
        :param conn:
        :param img: (label_num,data,)
        """
        if conn is not None:
            sql = ''' INSERT INTO imgs(label_num,data)
                    VALUES(?,?) '''
            cur = conn.cursor()
            cur.execute(sql, img)

        else:
            print('ERROR connecting to db')

    
    def array_to_bin(arr):
        #converts an arry into binary representation
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def bin_to_array(bin):
        #converts bin to np array
        out = io.BytesIO(bin)
        out.seek(0)
        return np.load(out)

    sqlite3.register_adapter(np.ndarray, array_to_bin)# Converts np.array to TEXT when inserting
    sqlite3.register_converter("array", bin_to_array) # Converts TEXT to np.array when selecting

    # create a database connection
    if config['random_db'] in ['true','True','TRUE']:
        conn_path = config['db_path'] + config['ds_name'] + str(random.randint(0,10000000))+'.db'
    else:
        conn_path = config['db_path'] + config['ds_name'] +'.db'
    print('db conn path =',conn_path)
    conn = None
    try:
        conn = sqlite3.connect(conn_path,detect_types=sqlite3.PARSE_DECLTYPES)
    except Error as e:
        print(e)
        

    #check table exists
    curr = conn.cursor()
    curr.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='imgs' ''')
    if curr.fetchone()[0]==1 :
        print('Table exists. Initilizing Data.')
        #initilise and reset data
        curr.execute(''' UPDATE imgs SET batch_num = (?) ''',('-1',))
        conn.commit()
        print('INIT: Using ',config['ds_name'], ' data, downloading now...')
        train_split = 'train[:'+str(int(config['train_percent']*100))+'%]' 
        test_split = 'test[:'+str(int(config['test_percent']*100))+'%]'
        train_ds, ds_info = tfds.load(config['ds_name'],with_info=True,shuffle_files=False,as_supervised=True,split=train_split)
        test_ds = tfds.load(config['ds_name'],with_info=False,shuffle_files=False,as_supervised=True,split=test_split)
    else:
        print('Table does not exist, building now...')
        sql_create_img_table = """ CREATE TABLE IF NOT EXISTS imgs (
                                        id INTEGER PRIMARY KEY,
                                        label_num INTEGER,
                                        data array,
                                        batch_num INTEGER,
                                        used INTEGER
                                    ); """

        # create table
        if conn is not None:
            try:
                c = conn.cursor()
                c.execute(sql_create_img_table)
            except Error as e:
                print(e)
        else:
            print("Error! cannot create the database connection.")
        
        #populate table
        #take the tfds dataset and produce a dataset and dataframe this is deterministic
        print('INIT: Using ',config['ds_name'], ' data, downloading now...')
        train_split = 'train[:'+str(int(config['train_percent']*100))+'%]' 
        test_split = 'test[:'+str(int(config['test_percent']*100))+'%]'
        train_ds, ds_info = tfds.load(config['ds_name'],with_info=True,shuffle_files=False,as_supervised=True,split=train_split)
        test_ds = tfds.load(config['ds_name'],with_info=False,shuffle_files=False,as_supervised=True,split=test_split)

        #Take the dataset and add info to db
        i = 0
        for image,label in train_ds:
            if i % 5000 == 0 and i!=0: print('Images Complete = ',i)
            data_to_add = (str(label.numpy()),image.numpy(),)
            DB_add_img(conn, data_to_add)
            i += 1
        conn.commit()

    return test_ds,train_ds, ds_info, conn

def setup_logs(config,disabled):
    #Setup logs and records
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    if disabled:
        os.environ['WANDB_DISABLED'] = 'true'
    wandb.login()
    wandb.init(project='k_diversity',entity='adamdowse',config=config,group=config['group'])

def setup_model(config,num_classes,img_shape):
    tf.keras.backend.clear_session()
    model = sm.select_model(config['model_name'],num_classes,img_shape)
    model.build(img_shape+(1,))
    print('built model with input shape =',img_shape+(1,))
    model.summary()
    optimizer = keras.optimizers.SGD(learning_rate=config['learning_rate']),
    #loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
    loss_func = keras.losses.CategoricalCrossentropy(from_logits=False)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = keras.metrics.CategoricalAccuracy()
    return model,optimizer,loss_func,train_loss,train_acc_metric,test_loss,test_acc_metric

def calc_inner_scores(grads):
    return np.sum(np.sum(cdist(grads,grads,metric='euclidean')))

def calc_outer_scores(grads,mean_saved_gradients):
    if mean_saved_gradients is None:
        outer_div = 0
    else:
        mean_current_grads = np.mean(grads,axis=0)
        if mean_current_grads.ndim == 1:
            mean_current_grads = np.expand_dims(mean_current_grads,axis=0)
        dists = cdist(mean_current_grads,mean_saved_gradients,metric='euclidean')
        outer_div = np.sum(np.sum(dists)) / mean_saved_gradients.shape[0] ** 2
    return outer_div

def calc_scores(grads,mean_saved_gradients):
    #calc the inner diversity of the selected grads
    dists = cdist(grads,grads,metric='euclidean')
    inner_div = np.sum(np.sum(dists)) / len(dists)**2

    #calc outer div with the means of the prevous batches
    #mean_saved_gradients = [[batch_mean_grads],[...],...]
    
    if mean_saved_gradients is None:
        outer_div = 1
    else:
        mean_current_grads = np.mean(grads,axis=0)
        if mean_current_grads.ndim == 1:
            mean_current_grads = np.expand_dims(mean_current_grads,axis=0)
        dists = cdist(mean_current_grads,mean_saved_gradients,metric='euclidean')
        outer_div = np.sum(np.sum(dists)) / mean_saved_gradients.shape[0] ** 2

    return (inner_div, outer_div,)

def array_add(array, item, max_size):
    #item must be an array
    #shifts an array removing the last value and adding a new one to position 
    if array.shape[0] >= max_size:
        #need to remove the last value
        return np.concatenate((np.expand_dims(item,axis=0), array[:-1]))
    else:
        return np.concatenate((np.expand_dims(item,axis=0), array))

def sample_batches(run_type,model,train_it,train_ds,batch_size,num_classes,conn,des_inner=0.5,des_outer=0.5,mean_saved_gradients=None,k=1):

    #calculate the approximate grads from model for all data
    time1 = time.time()
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-1].input) #logits
    for i, (img,label) in enumerate(train_ds.batch(2**11)): #2**11
        logits = intermediate_layer_model.predict(img,verbose=0)
        grads = np.array(logits - tf.one_hot(label,num_classes))
        if i == 0:
            aprox_grads = grads
        else:
            aprox_grads = np.concatenate((aprox_grads,grads),axis=0) # list of the grads [[grads],[grads],...]
    
    print("Inference Time:",time.time() - time1)
    time1 = time.time()
    #create a large number of random number sequences (Possible to do repeates here but unlikely)
    total_seq = 100000
    sequences = np.random.randint(0,len(aprox_grads),size=(total_seq,batch_size)) #lists of random indexes of the grads

    #based on run type score the sequences
    if run_type == 'i':
        #score each sequence based on inner diversity
        with Pool(maxtasksperchild=1000) as pool:
            pool_input = [(aprox_grads[s],) for s in sequences]
            i_scores = pool.starmap(calc_inner_scores,pool_input,chunksize=50,)
        
        #normalise the scores
        n_i_scores = (i_scores - np.min(i_scores)) / (np.max(i_scores) - np.min(i_scores))

        #choose sequence closes to desired inputs and the sequence to use
        s_idx = np.abs(n_i_scores - des_inner).argmin()
        sequence = sequences[s_idx]

        #calc the difference between the desired value and the achieved
        i_des_descrep = np.abs(n_i_scores - des_inner).min()

        #wandb logging
        wandb.log({'mean_i_scores':np.mean(i_scores),
                    'chosen_i_score':i_scores[s_idx],
                    'inner_des_diff':i_des_descrep,
                    'all_i_scores_90':np.percentile(i_scores,90),
                    'all_i_scores_10':np.percentile(i_scores,10)},step=train_it)

    elif run_type == 'o':
        #score each sequence based on outer diversity
        with Pool(maxtasksperchild=50) as pool:
            pool_input = [(aprox_grads[s],mean_saved_gradients,) for s in sequences]
            o_scores = pool.starmap(calc_outer_scores,pool_input,chunksize=50,)
        
        #normalise the scores
        n_o_scores = (o_scores - np.min(o_scores)) / (np.max(o_scores) - np.min(o_scores))

        #choose sequence closes to desired inputs and the sequence to use
        s_idx = np.abs(n_o_scores - des_outer).argmin()
        sequence = sequences[s_idx]

        #calc the difference between the desired value and the achieved
        o_des_descrep = np.abs(n_o_scores - des_outer).min()

        #calc the mean of the chosen gradients and save
        if mean_saved_gradients is None:
            mean_saved_gradients = np.expand_dims(np.mean(aprox_grads[sequence],axis=0),axis=0)
        else:
            mean_saved_gradients = array_add(mean_saved_gradients,np.mean(aprox_grads[sequence],axis=0),k)

        #wandb logging
        wandb.log({ 'mean_o_scores':np.mean(o_scores),
                    'chosen_o_score':o_scores[s_idx],
                    'outer_des_diff':o_des_descrep,
                    'all_o_scores_90':np.percentile(o_scores,90),
                    'all_o_scores_10':np.percentile(o_scores,10)},step=train_it)

    elif run_type == 'io':
        #score each sequence based on inner and outer diversity
        with Pool(maxtasksperchild=50) as pool:
            pool_input = [(aprox_grads[s],mean_saved_gradients,) for s in sequences]
            scores = pool.starmap(calc_scores,pool_input,chunksize=50,)

        #split the scores into inner and outer and normalise
        i_scores = [s[0] for s in scores]
        o_scores = [s[1] for s in scores]
        n_i_scores = (i_scores - np.min(i_scores)) / (np.max(i_scores) - np.min(i_scores))
        n_o_scores = (o_scores - np.min(o_scores)) / (np.max(o_scores) - np.min(o_scores))

        #choose sequence closes to desired inputs and set the sequence
        s_idx = (np.abs(n_i_scores - des_inner)+np.abs(n_o_scores - des_outer)).argmin()
        sequence = sequences[s_idx]

        #calc the difference between the desired value and the achieved
        i_des_descrep = (np.abs(n_i_scores - des_inner)).min()
        o_des_descrep = (np.abs(n_o_scores - des_outer)).min()

        #calc the mean of the chosen gradients and save
        if mean_saved_gradients is None:
            mean_saved_gradients = np.expand_dims(np.mean(aprox_grads[sequence],axis=0),axis=0)
        else:
            mean_saved_gradients = array_add(mean_saved_gradients,np.mean(aprox_grads[sequence],axis=0),k)
        
        #wandb logging
        wandb.log({ 'mean_o_scores':np.mean(o_scores),
                    'chosen_o_score':o_scores[s_idx],
                    'outer_des_diff':o_des_descrep,
                    'all_o_scores_90':np.percentile(o_scores,90),
                    'all_o_scores_10':np.percentile(o_scores,10),
                    'mean_i_scores':np.mean(i_scores),
                    'chosen_i_score':i_scores[s_idx],
                    'inner_des_diff':i_des_descrep,
                    'all_i_scores_90':np.percentile(i_scores,90),
                    'all_i_scores_10':np.percentile(i_scores,10)},step=train_it)

    else:
        print("Invalid run type")
        return
    print("Sequence chosen time: ",time.time()-time1)
    time1 = time.time()
    #saving to the database
    curr = conn.cursor()
    b_num = 0
    for i,idx in enumerate(sequence):
        if i != 0:
            b_num = i // batch_size
        curr.execute(''' UPDATE imgs SET batch_num = (?) WHERE id = (?) ''',(int(b_num),int(idx),))
    conn.commit()
    print("Saving to database time: ",time.time()-time1)
    return mean_saved_gradients
    
