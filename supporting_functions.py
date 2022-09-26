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




#The train data generator
class CustomDBDataGen(tf.keras.utils.Sequence):
    def __init__(self, conn,
                 batch_size,
                 num_classes,
                 num_images
                 ):
        #INIT:
        #randomly batch for the first epoch and update the db

        #init db connection and init vars
        self.conn = conn
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_images = num_images

        #randomly batch for first warm start
        if num_images / batch_size == 0:
            self.num_batches = int(num_images/batch_size)
        else:
            self.num_batches = 1 + int(num_images/batch_size)

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
'''
def calc_inner_diversity(indexes,batch_num,i,grads):
    ''
    current_indexes = list of indexes in this batch [indexes]
    i = the item index that we are scoring
    grads = the list of aproximate gradients for this index [...]
    ''
    #if length is 0?TODO
    #calculate the euclidean distance between the last layer gradients
    #we want this to be minimal
    current_indexes = indexes[batch_num]
    current_indexes = current_indexes[current_indexes>=0]
    new_dists = cdist([grads[i]],grads[current_indexes],metric='euclidean')
    return np.min(new_dists,axis=1)

def calc_outer_diversity(indexes,i,grads,batch_num,k):
    #we want the new batch to be 
    #calc the distance between the modified batch with the item included to the other batches
    #the score is 1/ the distance 
    
    #loop though the other batches
    for b in range(k):
        current_indexes = indexes[b]
        current_indexes = np.array(current_indexes[current_indexes>=0],ndmin=1,dtype=int)
        if b == batch_num:
            current_indexes = np.append(current_indexes,i)
        
        #find mean of the new batch with added item
        mean_grads = np.expand_dims(np.mean(grads[current_indexes],axis=0),axis=0)
        if b == 0:
            batch_centers = mean_grads
        else:
            batch_centers = np.concatenate((batch_centers,mean_grads),axis=0)

    #calc dist between new center and the other centers
    new_dists = cdist([batch_centers[batch_num]],batch_centers[np.arange(len(batch_centers))!=batch_num])
    #new_dists = 1/new_dists
    #TODO does this represnt the right formula?
    return np.min(new_dists,axis=1)

def calc_scores(aid,batch_indexes,batch_num,approx_grads,k):
    #calc score for each sample
    inner_div_score = calc_inner_diversity(batch_indexes,batch_num,aid,aprox_grads)
    outer_div_score = calc_outer_diversity(batch_indexes,aid,aprox_grads,batch_num,k)
    #TODO add normalization
    #combine the scores based on the parameters alpha and beta
    score = alpha * inner_div_score + beta * outer_div_score
    return score

def k_diversity(model,train_ds,k,batch_size,alpha,beta,num_classes,conn):
    #collect the gradient information for all images
    #last layer gradients from coresets for data efficient training (logits - onehot_label)
    #TODO possibly wrong here
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-1].input) #logits
    for i, (img,label) in enumerate(train_ds.batch(100)):
        logits = intermediate_layer_model.predict(img)
        onehot = np.array([[1 if x == l else 0 for x in range(num_classes)] for l in label])
        grads = np.array(logits) - onehot
        if i == 0:
            aprox_grads = grads
        else:
            aprox_grads= np.concatenate((aprox_grads,grads),axis=0) # list of the grads [[grads],[grads],...]

    for i, (img,label) in enumerate(train_ds.batch(100)):
        softmax = model.predict(img)
        print("softmax",softmax)
        pnt()
        onehot = np.array([[1 if x == l else 0 for x in range(num_classes)] for l in label])
        entropy = np.sum(np.array(softmax) - onehot)
        if i == 0:
            e = entropy
        else:
            e = np.concatenate((e,entropy),axis=0) # list of the grads [entropy,entorpy]

            
    #run the submod evaluation
    #avalible = np.array([x for x in range(ds_size)]) #all indexes avalible
    curr = conn.cursor()
    curr.execute(''SELECT id FROM imgs'')
    avalible = np.array([int(x[0]) for x in curr.fetchall()]) # 1 to n for db ids
    print(len(avalible),'are avalible to select at start of process')

    batch_indexes = np.zeros((k,batch_size),dtype=int) -1 # [[batch0],[batch1],...,*k] init to -1

    
    pool = Pool()
    for b in range(batch_size): #each item goes into one batch at a time
        for batch_num in range(k):
            #calc inner batch distance
            if np.count_nonzero(batch_indexes[batch_num] >= 0) == 0:
                print('one random start')
                #if there are no items in the batch do this:
                #k_means cluster?
                #TODO some sort of initialization start
                #random
                choice = int(random.choice(avalible))
            else:
                #calculate inner and outer diversity for all the possible new samples
                #build the list that is used in the map
                pool_input = [(aid,batch_indexes,batch_num,approx_grads,k) for aid in avalible]
                item_scores = pool.starmap(calc_scores, pool_input)
                #pick greedly the best index and add it to the current batch
                choice = avalible[np.argmax(item_scores)]#choice is the db id
                
            #set the db values to the batch numbers of the images for each upcoming batch
            batch_indexes[batch_num][b] = choice 
            curr.execute('' UPDATE imgs SET batch_num = (?) WHERE id = (?) '',(int(batch_num),int(choice),))
            avalible = np.delete(avalible,avalible==choice)
            print(batch_indexes)

    #Should end up with a list of lists of indexes of batches here
    conn.commit()
'''



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

def sample_batches(model,train_ds,k,batch_size,num_classes,conn,des_inner,des_outer,images_used,mean_saved_gradients):
    #calculate the approximate grads from model for all data
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-1].input) #logits
    for i, (img,label) in enumerate(train_ds.batch(100)):
        logits = intermediate_layer_model.predict(img,verbose=0)
        onehot = np.array([[1 if x == l else 0 for x in range(num_classes)] for l in label])
        grads = np.array(logits) - onehot
        if i == 0:
            aprox_grads = grads
        else:
            aprox_grads = np.concatenate((aprox_grads,grads),axis=0) # list of the grads [[grads],[grads],...]

    #create a large number of random number sequences (Possible to do repeates here but unlikely)
    total_seq = 100000
    sequences = np.random.randint(0,len(aprox_grads),size=(total_seq,batch_size)) #lists of random indexes of the grads

    #score each sequence based on inner and outer diversity
    with Pool(maxtasksperchild=50) as pool:
        pool_input = [(aprox_grads[s],mean_saved_gradients,) for s in sequences]
        scores = pool.starmap(calc_scores,pool_input,chunksize=50,) #shape of sequences
    
    #normalise to 0 and 1
    i_scores = [s[0] for s in scores]
    o_scores = [s[1] for s in scores]
    n_i_scores = (i_scores - np.min(i_scores)) / (np.max(i_scores) - np.min(i_scores))
    n_o_scores = (o_scores - np.min(o_scores)) / (np.max(o_scores) - np.min(o_scores))

    #choose sequence closes to desired inputs and set the sequence in the db
    s_idx = (np.abs(n_i_scores - des_inner)+np.abs(n_o_scores - des_outer)).argmin()

    
    #calc the difference between the desired value and the achieved
    i_des_descrep = (np.abs(n_i_scores - des_inner)).min()
    o_des_descrep = (np.abs(n_o_scores - des_outer)).min()

    #the index of data to use
    sequence = sequences[s_idx]
    
    #calc the mean of the chosen gradients and save
    if mean_saved_gradients is None:
        mean_saved_gradients = np.expand_dims(np.mean(aprox_grads[sequence],axis=0),axis=0)
    else:
        mean_saved_gradients = array_add(mean_saved_gradients,np.mean(aprox_grads[sequence],axis=0),k)

    
    #saving to the database
    curr = conn.cursor()
    b_num = 0
    for i,idx in enumerate(sequence):
        if i != 0:
            b_num = i // batch_size
        images_used[idx] += 1
        curr.execute(''' UPDATE imgs SET batch_num = (?) WHERE id = (?) ''',(int(b_num),int(idx),))
    conn.commit()

    return i_scores,o_scores, s_idx, i_des_descrep, o_des_descrep, n_i_scores, mean_saved_gradients
    

