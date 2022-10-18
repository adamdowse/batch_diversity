
#collection of helpper functions for the main script
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras.models import Model
import sqlite3
import io
from sqlite3 import Error
import numpy as np
import scipy
from scipy.spatial.distance import cdist
import random

#The train data generator
class SubModDataGen(tf.keras.utils.Sequence):
    #This Generator is used to generate batches of data for the training of the model via submodular selection
    def __init__(self, conn, batch_size, modifiers):
        #INIT:
        #conn: the connection to the database
        #batch_size: the size of the batch to use

        print("Starting SubMod Data Generator")

        #init db connection and init vars
        self.conn = conn
        self.batch_size = batch_size
        self.batch_indexes = np.array([])
        self.modifiers = modifiers
        curr = conn.cursor()
        self.num_classes = curr.execute('''SELECT COUNT(DISTINCT label_num) FROM imgs''').fetchone()[0]
        self.num_images = curr.execute('''SELECT COUNT(id) FROM imgs''').fetchone()[0]
        self.num_batches = int(np.ceil(self.num_images/self.batch_size))

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", self.num_images)
        print("Batch size: ", self.batch_size)
        print("Number of batches: ", self.num_batches)

        
        
    def ret_batch_info(self):
        return self.num_batches

    def __getitem__(self, index):
        #gets the next batch of data

        #pull the data from the database
        curr = self.conn.cursor()
        imgs = []
        labels = []
        for bi in self.batch_indexes:
            curr.execute('SELECT data, label_num FROM imgs WHERE id = (?)', (int(bi),))
            data = curr.fetchall()[0]
            imgs.append(data[0])
            labels.append(data[1])

        
        #convert to tensors
        imgs = tf.cast(np.array(imgs),'float32')
        labels = tf.one_hot(np.array(labels),self.num_classes)

        

        return tuple(imgs, labels)

    def __len__(self):
        #calculates the number of batches to use
        return 1

    def __del__(self):
        #close the database connection
        self.conn.close()

    def score_images(self,model):
        #run the scoring functions through all the images in the database
        #TODO look into using multiprocessing to speed this up

        def Norm(A):
            if np.sum(A) == 0:
                return A
            else:
                return A/np.sum(A)

        def Uncertainty_Score(preds,indexes):
            #calculate the uncertainty score for each image in the database
            #how uncertain is the model about the prediction for each image
            #preds: the softmax outputs of the model
            #indexes: the indexes of the images to score
            
            #make sure that no -infs are present
            preds = np.where(preds < 1e-30, 1e-30, preds)
            return - np.sum(preds[indexes] * np.log(preds[indexes]),axis=1)

        def Redundancy_Score(pl_activations,set_indexes,indexes):
            #calculate the redundancy score for each image in the database
            # the point shoulde be as far away from the other points in the set as possible
            #model: the model to use to calculate the redundancy score
            
            if len(set_indexes) != 0:
                #calculate the redundancy score
                return  np.min(cdist(pl_activations[set_indexes],pl_activations[indexes]),axis=0)
            else:
                return np.zeros(len(pl_activations))

        def Mean_Close_Score(pl_activations,indexes):
            #calculate the mean close score for each image in the database
            #the selected points should be as close to the mean of the total data as possible

            #calculate the mean close score
            d = cdist([np.mean(pl_activations,axis=0)],pl_activations[indexes],metric='sqeuclidean')
            return -np.squeeze(d)

        
        def Feature_Match_Score(pl_activations,set_indexes,indexes):
            #calculate the feature match score for each image in the database

            #pl_activations: the penultimate layer activations of the model
            #set_indexes: the indexes of the images that are already in the set

            #convert to softmax
            sm_layer = tf.keras.layers.Softmax()
            pl_softmax = sm_layer(pl_activations).numpy()

            if len(set_indexes) == 0:
                return np.sum(np.sqrt(pl_softmax[indexes]),axis=1)
            else:
                subset_scores = np.sum(pl_softmax[set_indexes],axis=0)
                return np.sum(np.sqrt( subset_scores + pl_softmax[indexes]),axis=1)



        #calculate the ll_activations for all images
        #get the data from the database
        curr = self.conn.cursor()
        curr.execute('''SELECT data FROM imgs''')
        imgs = []
        for img in curr:
            imgs.append(img)

    
        #convert to tensor
        imgs = np.array(imgs)
        imgs = np.squeeze(imgs,axis=1)
        print(imgs.shape)
        imgs = tf.cast(imgs,'float32')

        #run the model on the data so we get last layer activations
        ll_model = Model(inputs=model.input, outputs=model.get_layer('last_layer').output)
        ll_activations = ll_model.predict(imgs)
        print("ll_activations:",ll_activations.shape)

        #run the model on the data so we get the penultimate layer activations
        pl_model = Model(inputs=model.input, outputs=model.get_layer('penultimate_layer').output)
        pl_activations = pl_model.predict(imgs)
        print("pl_activations:",pl_activations.shape)


        #run the model on the data so we get the softmax outputs
        preds = model.predict(imgs)
        print("preds:",preds.shape)


        #build a batch via submodular selection
        #calculate the scores for each image
        indexes = np.arange(self.num_images,dtype=int)
        set_indexes = np.array([],dtype=int)
        for i in range(self.batch_size):
            #calculate the scores for each image
            U_S = Uncertainty_Score(preds,indexes)
            R_S = Redundancy_Score(pl_activations,set_indexes,indexes)
            MC_S = Mean_Close_Score(pl_activations,indexes)
            FM_S = Feature_Match_Score(ll_activations,set_indexes,indexes)


            #calculate the total score
            scores = (Norm(U_S*self.modifiers[0]) + 
                        Norm(R_S*self.modifiers[1]) + 
                        Norm(MC_S*self.modifiers[2]) + 
                        Norm(FM_S*self.modifiers[3]))
            
            #print(scores)

            #get the index of the image with the highest score
            max_index = np.argmax(scores)
            #add the index to the set
            set_indexes = np.append(set_indexes,indexes[max_index])
            #remove the index from the indexes
            indexes = np.delete(indexes,max_index)
        
        #set the self variables as the indexes of the images in the batch
        print(set_indexes)
        self.batch_indexes = set_indexes
        
        


def setup_db(config):

    def DB_add_img(conn, to_insert):
        """
        Create a new img into the img table
        :param conn:
        :param to_insert: (label_num,data,)
        """
        if conn is not None:
            cur = conn.cursor()
            cur.execute(''' INSERT INTO imgs(label_num,data) VALUES(?,?) ''', to_insert)

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
        

    #Build out database
    curr = conn.cursor()
    print('Table does not exist, building now...')
    sql_create_img_table = """ CREATE TABLE IF NOT EXISTS imgs (
                                    id INTEGER PRIMARY KEY,
                                    label_num INTEGER,
                                    data array
                                ); """

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
    for i,(image,label) in enumerate(train_ds):
        if i % 5000 == 0 and i!=0: print('Images Complete = ',i)
        data_to_add = (str(label.numpy()),image.numpy(),)
        DB_add_img(conn, data_to_add)

    conn.commit()
    return test_ds, ds_info, conn