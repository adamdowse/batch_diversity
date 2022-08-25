#collection of helpper functions
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
            num_batches = int(num_images/batch_size)
        else:
            num_batches = 1 + int(num_images/batch_size)

        print('There are ',num_batches,' batches in the warm start')
        arr = []
        for n in range(num_batches):
            to_add = [n]*batch_size
            arr = arr + to_add
        
        arr = arr[:num_images] #Limit
        random.shuffle(arr) #shuffle

        #update the database
        curr = conn.cursor()
        for i, a in enumerate(arr):
            curr.execute('''UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(a),int(i),))
        conn.commit()

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

def k_diversity(model,train_ds,k,batch_size,alpha,beta,num_classes,conn):

    def calc_inner_diversity(indexes,batch_num,i,grads):
        '''
        current_indexes = list of indexes in this batch [indexes]
        i = the item index that we are scoring
        grads = the list of aproximate gradients for this index [...]
        '''
        
        #if length is 0?TODO
        #calculate the euclidean distance between the last layer gradients
        #we want this to be minimal
        current_indexes = indexes[batch_num]
        current_indexes = current_indexes[current_indexes>=0]
        new_dists = cdist([grads[i]],grads[current_indexes],metric='euclidean')
        return np.min(new_dists,axis=1)

    def calc_outer_diversity(indexes,i,grads,batch_num,k):
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

    #collect the gradient information for all images
    #last layer gradients from coresets for data efficient training (logits - onehot_label)
    #TODO possibly wrong here
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-1].input) #logits
    #TODO make this batched for speed
    for i, (img,label) in enumerate(train_ds.batch(100)):
        logits = intermediate_layer_model.predict(img)
        onehot = np.array([[1 if x == l else 0 for x in range(num_classes)] for l in label])
        grads = np.array(logits) - onehot
        if i == 0:
            aprox_grads = grads
        else:
            aprox_grads= np.concatenate((aprox_grads,grads),axis=0) # list of the grads [[grads],[grads],...]

    #run the submod evaluation
    #avalible = np.array([x for x in range(ds_size)]) #all indexes avalible
    curr = conn.cursor()
    curr.execute('''SELECT id FROM imgs''')
    avalible = np.array([int(x[0]) for x in curr.fetchall()])
    print(len(avalible),'are avalible to select at start of process')

    batch_indexes = np.zeros((k,batch_size),dtype=int) -1 # [[batch0],[batch1],...,*k] init to -1

    #TODO might be possible to parralize this creating each batch simultaniouly
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
                item_scores = []
                #TODO parralize the below
                for i,grads in enumerate(aprox_grads): 
                    if i in avalible:
                        #calc score for each sample
                        #TODO change the function inputs to match
                        inner_div_score = calc_inner_diversity(batch_indexes,batch_num,i,aprox_grads)
                        outer_div_score = calc_outer_diversity(batch_indexes,i,aprox_grads,batch_num,k)
                        #print(inner_div_score)
                        #print(outer_div_score)
                        #TODO add normalization
                        #combine the scores based on the parameters alpha and beta
                        score = alpha * inner_div_score + beta * outer_div_score
                        item_scores = np.concatenate((item_scores,score),axis=0)
                    #else:
                        #cant use this point but used to hold indexing position
                    #    item_scores.append(-1) #TODO check this dosent break anything
                #print(item_scores)
                #pick greedly the best index and add it to the current batch
                choice = np.argmax(item_scores) #choice is the index in avalible
                
            #set the db values to the batch numbers of the images for each upcoming batch
            batch_indexes[batch_num][b] = avalible[choice]  #avalible[choice] is the og index
            curr.execute(''' UPDATE imgs SET batch_num = (?) WHERE id = (?) ''',(int(batch_num),int(avalible[choice]),))
            avalible = np.delete(avalible,choice)
            print(batch_indexes)

    #Should end up with a list of lists of indexes of batches here
    conn.commit()
   