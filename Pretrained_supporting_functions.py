
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
#import sklearn
import random
import os
import wandb
import matplotlib.pyplot as plt

#the test data gen
class TestDataGen(tf.keras.utils.Sequence):
    def __init__(self, test_ds, batch_size,num_classes):
        self.test_ds = test_ds.batch(batch_size)
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.imgs = []
        self.labels = []
        self.num_batches = 0
        for imgs, labels in self.test_ds:
            self.num_batches += 1
            self.imgs.append(imgs)
            self.labels.append(tf.one_hot(labels, self.num_classes))
        
    def __getitem__(self, index):
        return (self.imgs[index], self.labels[index],)
    
    def __len__(self):
        return self.num_batches

#The train data generator
class SubModDataGen(tf.keras.utils.Sequence):
    #This Generator is used to generate batches of data for the training of the model via submodular selection
    def __init__(self, conn_path, config):
        print("Starting SubMod Data Generator")
        #init db connection and init vars
        self.config = config
        self.conn_path = conn_path

        try:
            conn = sqlite3.connect(self.conn_path,detect_types=sqlite3.PARSE_DECLTYPES)
        except Error as e:
            print(e)
        curr = conn.cursor()
        self.num_classes = curr.execute('''SELECT COUNT(DISTINCT label_num) FROM imgs''').fetchone()[0]
        self.num_images = curr.execute('''SELECT COUNT(id) FROM imgs''').fetchone()[0]
        conn.close()

        self.data_used = np.zeros(self.num_images,dtype=int)

        #get the data #TODO this is a hack to get the data
        self.__get_imgs()

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", self.num_images)
        print("Batch size: ", config['batch_size'])

    def __getitem__(self, index):
        #gets the next batch of data
        #build a batch via submodular selection

        #randomly select a batch of images
        if self.config['train_type'] in ['Random','random']:
            if len(self.set_indexes) < self.config['batch_size']:
                #if there are not enough images left in the set, use all the images
                choices = np.arange(0,len(self.set_indexes))
            else:
                #randomly select a batch of images
                choices = np.random.randint(0,len(self.set_indexes),self.config['batch_size'])

            #add the index to the set
            self.batch_indexes = self.set_indexes[choices]
            #remove the index from the indexes
            self.set_indexes = np.delete(self.set_indexes,choices)
        elif self.config['train_type'] in ['Submodular','submodular','SubModular','subModular','SubMod','subMod','Submod']:
            #gets the indexes of the images to use in the next batch and removes them form the set of available images
            self.batch_indexes = np.array([],dtype=int)
            self.__get_submod_indexes()
        
        else:
            print('ERROR: Invalid train type')

        #get the data for the batch
        imgs = self.imgs[self.batch_indexes]
        labels = self.labels[self.batch_indexes]

        #convert to tensors
        imgs = tf.cast(np.array(imgs),'float32')
        labels = tf.one_hot(np.array(labels),self.num_classes)

        #reset the set indexes
        self.batch_indexes = np.array([],dtype=int)

        return (imgs, labels,)

    def __len__(self):
        #calculates the number of batches to use
        return self.num_batches

    def get_data_subset(self, model,train_ds,defect=False):
        #create a subset of the data to use for training
        if self.config['subset_type'] in ['All','all'] and defect==False:
            #Use all the data
            print('No hard mining, full epoch used')
            self.set_indexes = np.arange(self.num_images)
            self.data_used += 1
            self.num_batches = int(np.ceil(self.num_images/self.config['batch_size']))
            self.losses = np.zeros(self.num_images)

        elif self.config['subset_type'] in ['HM','hard_mining','Hard_Mining','Hard_mining'] and defect==True:
            #score all images with loss and keep activations from the top k % of images
            print("Hard mining")
            #get losses for all images
            self.losses = self.__record_losses(model,train_ds)

            #get the top k % of images TODO this needs to be robust?
            self.set_indexes = np.argsort(self.losses)[:int(np.ceil(self.config['k_percent']*len(self.losses)))]
            self.num_batches = int(np.ceil(len(self.set_indexes)/self.config['batch_size']))
            self.data_used[self.set_indexes] += 1

        elif self.config['subset_type'] in ['EM','easy_mining','Easy_Mining', 'Easy_mining'] and defect==True:
            print('Easy mining')
            #get losses for al images
            self.losses = self.__record_losses(model,train_ds)

            #get the bottom k% images
            #invert argsort
            sort = np.flip(np.argsort(self.losses))
            self.set_indexes = sort[:int(np.ceil(self.config['k_percent']*len(self.losses)))]
            self.num_batches = int(np.ceil(len(self.set_indexes)/self.config['batch_size']))
            self.data_used[self.set_indexes] += 1


        elif self.config['subset_type'] in ['Random_Bucket','random_bucket','Random_bucket'] and defect==True:
            #select a random bucket of images from the full set
            print("Random Bucket")
            self.set_indexes = np.random.randint(0,self.num_images,int(np.ceil(self.config['k_percent']*self.num_images)))
            self.data_used[self.set_indexes] += 1
            self.num_batches = int(np.ceil(len(self.set_indexes)/self.config['batch_size']))
            self.losses = np.zeros(self.num_images)
            
        else:
            print('ERRORL: Invalid subset type')
        
        print('num batches in epoch: ',self.num_batches)

    def __get_submod_indexes(self):

        def Norm(A):
            if np.sum(A) == 0:
                return A
            else:
                return A/np.sum(A)

        def Dist_To_Mean_Score(activations,set_indexes,batch_indexes):
            #calculate the distance to the batch mean for each image in the subset
            #the point shoulde be as close to the other points in the set as possible
            #activations: the activations of the model (either the last layer or the penultimate layer)
            #set_indexes: the indexes of the images in the set
            #batch_indexes: the indexes of the images in the batch

            if len(batch_indexes) != 0:
                #calculate the mean of the batch activations
                batch_mean = np.mean(activations[batch_indexes],axis=0)
                #TODO ensure this is the right shape
                return 1/np.min(cdist([batch_mean],activations[set_indexes]),axis=0)
            else:
                return np.zeros(len(set_indexes))

        #get the indexes of the images to use in the next batch
        if len(self.set_indexes) < self.config['batch_size']: 
            self.batch_indexes = self.set_indexes
        else: 
            #k-means++ clustering
            #cluster_indexes = sklearn.cluster.KMeans(n_clusters=self.num_batches,init='k-means++').fit_predict(self.activations)

            for i in range(self.config['batch_size']):
                #calculate the scores for each image in subset and normalize
                scores = Dist_To_Mean_Score(self.activations,self.set_indexes,self.batch_indexes)

                #scores = Norm(mean_dist_score)
                
                #if the sum of the scores is 0, then select a random image
                if np.sum(scores) == 0:
                    #select a random image
                    #TODO: this should be improved
                    max_index = np.random.randint(0,len(scores))
                else:
                    #get the index of the image with the highest score
                    max_index = np.argmax(scores)

                #add the index to the set
                self.batch_indexes = np.append(self.batch_indexes,self.set_indexes[max_index])
                #remove the index from the indexes
                self.set_indexes = np.delete(self.set_indexes,max_index)
            
    def __get_imgs(self):
        #get the data from the database
        try:
            conn = sqlite3.connect(self.conn_path,detect_types=sqlite3.PARSE_DECLTYPES)
        except Error as e:
            print(e)
        curr = conn.cursor()
        curr.execute('''SELECT data, label_num FROM imgs''')
        self.imgs = []
        self.labels = []
        for img, label in curr:
            self.imgs.append(img)
            self.labels.append(label)
        conn.close()

        print('imgs shape: ',np.array(self.imgs).shape)
        print('labels shape: ',np.array(self.labels).shape)

        #convert to numpy arrays
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        
    def __record_losses(self,model,train_ds):
        #record the losses for the model into a histogram TODO this needs to be better
        @tf.function
        def loss_step(x,y):
            with tf.GradientTape() as tape:
                pred = model(x)
                loss = loss_func_no(y,pred)
            return loss

        print('recording losses')
        loss_func_no = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        losses = np.array([])

        for x,y in train_ds.batch(128).map(lambda x, y: (x, tf.one_hot(y, depth=self.num_classes))):
            losses = np.append(losses,loss_step(x,y).numpy())

        return losses

    def get_activations(self,model,batch_num):
        #TODO look into using multiprocessing to speed this up
        #Also for the every batch epoch we dont need all the activations
        #get the activations of the model for each image in the subset so that the subset_index aligns with activations
        if (self.config['train_type'] in ['Random','random']) or (batch_num % self.config['activations_delay'] != 0):
            return
        else:
            imgs = self.imgs[self.set_indexes]
            imgs = tf.cast(imgs,'float32')
        
            inter_model = Model(inputs=model.input, outputs=model.get_layer(self.config['activation_layer_name']).output)
            activations = inter_model.predict(imgs,batch_size = 128)
            del inter_model

            #modify indexes of outputs to maintain the order of the images
            # from [0,2,4] to [n,0,n,0,n,0] ect
            self.activations = np.zeros((self.num_images,activations.shape[1]))
            for count, idx in enumerate(self.set_indexes):
                self.activations[idx] = activations[count]
        



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
    train_ds, ds_info = tfds.load(config['ds_name'],with_info=True,shuffle_files=False,as_supervised=True,split=train_split,data_dir=config['ds_path'],download=False)
    test_ds = tfds.load(config['ds_name'],with_info=False,shuffle_files=False,as_supervised=True,split=test_split,data_dir=config['ds_path'],download=False)

    #Take the dataset and add info to db
    for i,(image,label) in enumerate(train_ds):
        if i % 5000 == 0 and i!=0: print('Images Complete = ',i)
        data_to_add = (str(label.numpy()),image.numpy(),)
        DB_add_img(conn, data_to_add)

    conn.commit()
    conn.close()
    return test_ds, ds_info, conn_path, train_ds

def wandb_setup(config,disabled):
    #Setup logs and records
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    if disabled:
        os.environ['WANDB_DISABLED'] = 'true'
    wandb.login()
    wandb.init(project='k_diversity',entity='adamdowse',config=config,group=config['group'])