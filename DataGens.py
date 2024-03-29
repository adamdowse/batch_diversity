#collection of data gens to use fro training
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
import time
import copy
from multiprocessing.pool import ThreadPool


def imgsAndLabelsFromTFDataset(DS):
    imgs_store = []
    labels_store = []
    num_batches = 0
    for imgs, labels in DS:
        num_batches += 1
        imgs_store.append(imgs)
        labels_store.append(labels)

    #normalise
    imgs_store = np.array(imgs_store) #size = [batches, imgsdimx, imgsdimy, depth]
    i_max = np.max(imgs_store)
    i_min = np.min(imgs_store)
    return (np.array(imgs_store),np.array(labels_store),num_batches)


def collectInfoFromTFDataset(DS):
    #itterate through a dataset and record stats to normalise with later.
    #this is done with tf vars.
    #DS can be batched or not, reduce max/min is taking a global reduction.
    t_min = tf.constant(10000.,dtype=tf.float32)
    t_max = tf.constant(0.,dtype=tf.float32)
    count = 0
    for img, label in DS:
        count += 1
        img = tf.cast(img,'float32')

        #loop though the dataset
        i_min = tf.reduce_min(img)
        if tf.math.less(i_min, t_min):
            t_min = i_min

        i_max = tf.reduce_max(img)
        if tf.math.greater(i_max, t_max):
            t_max = i_max
    
    return t_min,t_max,count


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
    preds = preds[indexes]
    preds = np.where(preds < 1e-30, 1e-30, preds)
    return - np.sum(preds * np.log(preds),axis=1)

def Redundancy_Score(activations,set_indexes,batch_indexes):
    #calculate the redundancy score for each image in the database
    # the point shoulde be as far away from the other points in the set as possible
    #model: the model to use to calculate the redundancy score
    if len(batch_indexes) != 0:
        #calculate the redundancy score
        return  np.min(cdist(activations[set_indexes],activations[batch_indexes]),axis=1)
    else:
        return np.zeros(len(set_indexes))

def Mean_Close_Score(activations,set_indexes,class_mean):
    #calculate the mean close score for each image in the database
    #the selected points should be as close to the mean of the total data as possible

    #calculate the mean close score
    d = cdist([class_mean],activations[set_indexes],metric='sqeuclidean')
    return -np.squeeze(d)

def Feature_Match_Score(activations,set_indexes,batch_indexes):
    #calculate the feature match score for each image in the database

    #pl_activations: the penultimate layer activations of the model
    #set_indexes: the indexes of the images that are already in the set

    #convert to softmax
    #sm_layer = tf.keras.layers.Softmax()
    #softmax = sm_layer(activations).numpy()

    #if len(batch_indexes) == 0:
    #    return np.sum(np.sqrt(softmax[set_indexes]),axis=1)
    #else:
    #    subset_scores = np.sum(softmax[batch_indexes],axis=0)
    #    return np.sum(np.sqrt( subset_scores + softmax[indexes]),axis=1)
    return np.zeros(len(set_indexes))




class TestDataGen(tf.keras.utils.Sequence):
    def __init__(self, ds_name, batch_size, size, ds_dir, Download=True):
        print('INIT: Using ',size*100,"'%' of",ds_name, ' test data')
        test_split = 'test[:'+str(int(size*100))+'%]'
        test_ds,info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=test_split,data_dir=ds_dir,download=Download)

        self.test_ds = test_ds.batch(batch_size)
        self.batch_size = batch_size

        self.num_classes = info.features['label'].num_classes
        self.imgs_min, self.imgs_max, self.num_batches = collectInfoFromTFDataset(self.test_ds)
        print("Found min = ",self.imgs_min," max = ", self.imgs_max,"Using this to normalise the data to [0,1]")
        print("Batches: ",self.num_batches)
        
    def __getitem__(self, index):
        #this is what is called when the generator is called.
        for imgs, labels in self.test_ds.skip(index).take(1):
            imgs = tf.cast(imgs,'float32')
            imgs = (imgs - self.imgs_min)/(self.imgs_max - self.imgs_min)
            labels = tf.one_hot(labels,self.num_classes)
        return (imgs, labels,)
    
    def __len__(self):
        return self.num_batches





#The train data generator
class LocalSubModDataGen(tf.keras.utils.Sequence):
    #This Generator is used to generate batches of data for the training of the model via submodular selection
    def __init__(self, ds_name, batch_size, size, ds_dir, Download=True, lambdas = [0.2,0.2,0.2,0.2]):
        print("Starting SubMod Data Generator")
        #pull data
        train_split = 'train[:'+str(int(size*100))+'%]' 
        train_ds, info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=train_split,data_dir=ds_dir,download=Download)


        #init db connection and init vars
        self.num_classes = info.features['label'].num_classes
        self.class_names = info.features['label'].names
        self.img_size = info.features['image'].shape
        self.num_images = info.splits[train_split].num_examples
        self.lambdas = lambdas
        self.batch_size = batch_size

        self.data_used = np.zeros(self.num_images,dtype=int)
        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(train_ds)

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", len(self.imgs))
        print("Batch size: ", batch_size)


    def __getitem__(self, index):
        #gets the next batch of data
        #build a batch via submodular selection
        if self.StandardOveride:
            if len(self.random_batch_indexes[index*self.batch_size:]) < self.batch_size: 
                batch_indexes = self.random_batch_indexes[index*self.batch_size:]
            else:
                batch_indexes = self.random_batch_indexes[index*self.batch_size:(index+1)*self.batch_size]

        else:
            #This is the new version with partitioning
            set_size = len(self.set_indexes)
            num_partitions = 5
            ltl_log_ep = 5

            if set_size >= num_partitions*self.batch_size:
                #use the multiprocessing methods
                part_size = int(set_size / num_partitions)
                r_size = int((part_size * ltl_log_ep)/ self.batch_size) #this is how many indexes to sample to reduce cost
                random.shuffle(self.set_indexes)
                print(part_size)
                print(set_size)
                partitions = [self.set_indexes[k:k+part_size] for k in range(0,set_size,part_size)] #splitting set indexes into parts

                pool = ThreadPool(processes=len(partitions))
                pool_handlers = []
                for partition in partitions:
                    handler = pool.apply_async(get_subset_indices, args=(partition, self.activations,self.preds,self.batch_size,r_size,self.lambdas))
                    pool_handlers.append(handler)
                pool.close()
                pool.join()

                intermediate_indices = []
                for handler in pool_handlers:
                    intermediate_indices.extend(handler.get())
            
            else:
                intermediate_indices = self.set_indexes
            
            r_size = int(len(intermediate_indices) / self.batch_size * ltl_log_ep)

            #now do submod on the combined set
            batch_indexes = get_subset_indices(intermediate_indices,self.activations,self.preds,self.batch_size,r_size,self.lambdas)

            self.set_indexes = self.set_indexes.tolist()
            for item in batch_indexes:
                self.set_indexes.remove(item)
            
            self.set_indexes = np.array(self.set_indexes)

        #get the data for the batch
        imgs = self.imgs[batch_indexes]
        labels = self.labels[batch_indexes]

        #convert to tensors
        imgs = tf.cast(np.array(imgs),'float32') 
        labels = tf.one_hot(np.array(labels),self.num_classes)


        '''

            #gets the indexes of the images to use in the next batch and removes them from the set of available images
            self.batch_indexes = np.array([],dtype=int)

            if len(self.set_indexes) < self.batch_size: 
                self.batch_indexes = self.set_indexes
            else: 
                if len(self.set_indexes) > 10000:
                    subset_indexes = self.set_indexes[np.random.choice(len(self.set_indexes),5000)]
                else:
                    subset_indexes = self.set_indexes
                for i in range(self.batch_size):
                    #calculate the scores for the subset if an item is added from the superset.
                    Uscores = self.lambdas[0] * Norm(Uncertainty_Score(self.preds,subset_indexes))
                    Rscores = self.lambdas[1] * Norm(Redundancy_Score(self.activations,subset_indexes,self.batch_indexes))
                    Mscores = self.lambdas[2] * Norm(Mean_Close_Score(self.activations,subset_indexes))
                    Fscores = self.lambdas[3] * Norm(Feature_Match_Score(self.activations,subset_indexes,self.batch_indexes))

                    scores = Uscores + Rscores + Mscores + Fscores

                    #if the sum of the scores is 0, then select a random image
                    if np.sum(scores) == 0:
                        #select a random image
                        print('Random Data taken as scores == 0')
                        max_index = np.random.randint(0,len(scores))
                    else:
                        #get the index of the image with the highest score
                        max_index = np.argmax(scores)

                    #add the index to the set
                    self.batch_indexes = np.append(self.batch_indexes,subset_indexes[max_index])
                    #remove the index from the indexes
                    subset_indexes = np.delete(subset_indexes,max_index)
                    self.set_indexes = np.delete(self.set_indexes,max_index)


        #get the data for the batch
        imgs = self.imgs[self.batch_indexes]
        labels = self.labels[self.batch_indexes]

        #convert to tensors
        imgs = tf.cast(np.array(imgs),'float32') 
        labels = tf.one_hot(np.array(labels),self.num_classes)

        #reset the set indexes
        self.batch_indexes = np.array([],dtype=int)

        '''
        return (imgs, labels,)

    def __len__(self):
        #calculates the number of batches to use
        return self.num_batches


    def Epoch_init(self,StandardOveride):
        #must be called before a training epoch
        self.StandardOveride = StandardOveride
        #Use all the data
        self.set_indexes = np.arange(self.num_images)
        self.num_batches = int(np.ceil(self.num_images/self.batch_size))
        print('Full amount of data used, batches: ',self.num_batches)
        if self.StandardOveride:
            #shuffle the set indexes
            self.random_batch_indexes = self.set_indexes
            np.random.shuffle(self.random_batch_indexes)



    def get_activations(self,model,index,layer_name,delay):
        #get the activations of the model for each image in the subset so that the subset_index aligns with activations
        if index % delay == 0 and self.StandardOveride == False:
            print("Collecting Activations")
            imgs = tf.cast(self.imgs[self.set_indexes],'float32')
        
            #inter_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output,model.output])
            #local_activations,preds = inter_model.predict(imgs,batch_size = 128)
            #del inter_model

            preds,local_activations = model.predict(imgs,batch_size = 128)
            print(local_activations.shape)

            #modify indexes of outputs to maintain the order of the images
            #from [0,2,4] to [n,0,n,0,n,0] ect
            self.activations = np.zeros((self.num_images,local_activations.shape[1]))
            self.preds = np.zeros((self.num_images,preds.shape[1]))
            for count, idx in enumerate(self.set_indexes):
                self.activations[idx] = local_activations[count]
                self.preds[idx] = preds[count]


def get_subset_indices(index_set_input,activations,preds,subset_size,r_size,lambdas):
    if r_size < len(index_set_input):
        index_set = np.random.choice(index_set_input,r_size, replace=False)
    else:
        index_set = copy.deepcopy(index_set_input)

    subset_indices = []

    class_mean = np.mean(activations, axis=0)

    subset_size = min(subset_size,len(index_set))


    for i in range(0, subset_size):
        if r_size < len(index_set):
            index_set = np.random.choice(index_set,r_size,replace=False)
        

        Uscores = lambdas[0] * Norm(Uncertainty_Score(preds,index_set))
        Rscores = lambdas[1] * Norm(Redundancy_Score(activations,index_set,subset_indices))
        Mscores = lambdas[2] * Norm(Mean_Close_Score(activations,index_set,class_mean))
        Fscores = lambdas[3] * Norm(Feature_Match_Score(activations,index_set,subset_indices))

        scores = Uscores + Rscores + Mscores + Fscores
        #get the index of the image with the highest score
        max_index = np.argmax(scores)
        
        subset_indices.append(index_set[max_index])
        index_set = np.delete(index_set,max_index,axis=0)

    return subset_indices

            


#Diversity batch data gen this has many problems 
class LocalDivDataGen(tf.keras.utils.Sequence):

    def __init__(self, ds_name, batch_size, size, ds_dir, Download=True):
        print("Starting Low Diversity Data Generator")
        #pull data
        train_split = 'train[:'+str(int(size*100))+'%]' 
        train_ds, info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=train_split,data_dir=ds_dir,download=Download)


        #init db connection and init vars
        self.num_classes = info.features['label'].num_classes
        self.class_names = info.features['label'].names
        self.img_size = info.features['image'].shape
        self.num_images = info.splits[train_split].num_examples
        self.batch_size = batch_size
        self.div_euc = 0
        self.div_cos = 0
        self.div_euc_true = 0
        self.div_cos_true = 0

        self.data_used = np.zeros(self.num_images,dtype=int)
        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(train_ds)

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", len(self.imgs))
        print("Batch size: ", batch_size)


    def __getitem__(self, index):
        #gets the next batch of data
        #build a batch via greedy low diversity
        if self.StandardOveride:
            if len(self.random_batch_indexes[index*self.batch_size:]) < self.batch_size: 
                batch_indexes = self.random_batch_indexes[index*self.batch_size:]
            else:
                batch_indexes = self.random_batch_indexes[index*self.batch_size:(index+1)*self.batch_size]
            self.div_euc = np.sum(np.sum(np.tril(cdist(self.grads[batch_indexes],self.grads[batch_indexes])))) / ((len(batch_indexes)**2 + len(batch_indexes))/2)
            self.div_euc_true = cdist([np.mean(self.grads[batch_indexes],axis=0)],[np.mean(self.grads,axis=0)])

            self.div_cos = np.sum(np.sum(np.tril(cdist(self.grads[batch_indexes],self.grads[batch_indexes],'cosine')))) / ((len(batch_indexes)**2 + len(batch_indexes))/2)
            self.div_cos_true = cdist([np.mean(self.grads[batch_indexes],axis=0)],[np.mean(self.grads,axis=0)],'cosine')
            
        
        
        else:
            if len(self.set_indexes) <= self.batch_size:
                batch_indexes = self.set_indexes

                #calc div metric for batch
                #
                self.div_euc = np.sum(np.sum(np.tril(cdist(self.grads[batch_indexes],self.grads[batch_indexes])))) / ((len(batch_indexes)**2 + len(batch_indexes))/2)
                self.div_euc_true = cdist([np.mean(self.grads[batch_indexes],axis=0)],[np.mean(self.grads,axis=0)])

                self.div_cos = np.sum(np.sum(np.tril(cdist(self.grads[batch_indexes],self.grads[batch_indexes],'cosine')))) / ((len(batch_indexes)**2 + len(batch_indexes))/2)
                self.div_cos_true = cdist([np.mean(self.grads[batch_indexes],axis=0)],[np.mean(self.grads,axis=0)],'cosine')
            else:
                #select a random avalible item
                batch_index = self.set_indexes[np.random.randint(len(self.set_indexes))]

                #fill the batch with the low diversity gradients
                #find closest n-1 grads
                largest = np.argpartition(np.array(cdist(self.grads[self.set_indexes],[self.grads[batch_index]])).flatten(), -self.batch_size)[-self.batch_size:]
                batch_indexes = self.set_indexes[largest]

                #calc div metric for batch
                #
                self.div_euc = np.sum(np.sum(np.tril(cdist(self.grads[batch_indexes],self.grads[batch_indexes])))) / ((self.batch_size**2 + self.batch_size)/2)
                self.div_euc_true = cdist([np.mean(self.grads[batch_indexes],axis=0)],[np.mean(self.grads,axis=0)])

                self.div_cos = np.sum(np.sum(np.tril(cdist(self.grads[batch_indexes],self.grads[batch_indexes],'cosine')))) / ((len(batch_indexes)**2 + len(batch_indexes))/2)
                self.div_cos_true = cdist([np.mean(self.grads[batch_indexes],axis=0)],[np.mean(self.grads,axis=0)],'cosine')

                
                #remove the batch indexes from the set indexes so they are not used again
                self.set_indexes = self.set_indexes.tolist()
                for item in batch_indexes:
                    self.set_indexes.remove(item)
                
                self.set_indexes = np.array(self.set_indexes)
                #print("images left in superset:",len(self.set_indexes))
                #print("images in batch:",len(batch_indexes))


        #get the data for the batch
        imgs = self.imgs[batch_indexes]
        labels = self.labels[batch_indexes]

        #convert to tensors
        imgs = tf.cast(np.array(imgs),'float32') 
        labels = tf.one_hot(np.array(labels),self.num_classes)
        return (imgs, labels,)
    
    def get_div_score(self):
        return self.div_euc,self.div_euc_true,self.div_cos,self.div_cos_true

    def __len__(self):
        #calculates the number of batches to use
        return self.num_batches


    def Epoch_init(self,StandardOveride):
        #must be called before a training epoch
        self.StandardOveride = StandardOveride
        #Use all the data
        self.set_indexes = np.arange(self.num_images)
        self.num_batches = int(np.ceil(self.num_images/self.batch_size))
        print('Full amount of data used, batches: ',self.num_batches)
        if self.StandardOveride:
            #shuffle the set indexes
            self.random_batch_indexes = self.set_indexes
            np.random.shuffle(self.random_batch_indexes)



    def get_grads(self,model,index,layer_name,delay):
        #get the approximate gradients from the last layer activations 
        if index % delay == 0: #and self.StandardOveride == False:
            print("Collecting Gradients")
            imgs = tf.cast(self.imgs[self.set_indexes],'float32')
            labels = tf.one_hot(np.array(self.labels[self.set_indexes]),self.num_classes)

            grads = model.predict(imgs,batch_size = 128)[0] 
            grads = grads - labels

            #modify indexes of outputs to maintain the order of the images
            #from [0,2,4] to [n,0,n,0,n,0] ect
            self.grads = np.zeros((self.num_images,grads.shape[1]))
            for count, idx in enumerate(self.set_indexes):
                self.grads[idx] = grads[count]


def calc_batch_div(batch_grads, mean_grad):
    #use the gradients to calculate the diversity with measures (euclidean and cosine)
    #Euc_score = np.sum(np.sum(np.tril(cdist(batch_grads,batch_grads,'Euclidean')))) / ((len(batch_grads)**2 + len(batch_grads))/2)
    #Euc_true = cdist([np.mean(batch_grads,axis=0)],[mean_grad])

    
    #Cos_score = np.nan_to_num(np.tril(cdist(batch_grads,batch_grads,'Cosine')))

    #Cos_score = np.sum(np.sum(Cos_score)) 
    #Cos_score /= ((len(batch_grads)**2 + len(batch_grads))/2)

    Cos_true = cdist([np.mean(batch_grads,axis=0)],[mean_grad],'Cosine')

    return Cos_true #[Euc_score,Euc_true,Cos_score,Cos_true]



#Diversity batch data gen
class LocalSUBMODGRADDataGen(tf.keras.utils.Sequence):

    def __init__(self, ds_name, batch_size, size, ds_dir,alpha, Download=True):
        print("Starting Low Diversity Data Generator")
        #pull data
        train_split = 'train[:'+str(int(size*100))+'%]' 
        train_ds, info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=train_split,data_dir=ds_dir,download=Download)


        #init db connection and init vars
        self.num_classes = info.features['label'].num_classes
        self.class_names = info.features['label'].names
        self.img_size = info.features['image'].shape
        self.num_images = info.splits[train_split].num_examples
        self.batch_size = batch_size
        self.div_score = 0
        self.alpha = alpha

        self.data_used = np.zeros(self.num_images,dtype=int)

        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(train_ds)

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", len(self.imgs))
        print("Batch size: ", batch_size)


    def __getitem__(self, index):
        #gets the next batch of data
        #build a batch via greedy low diversity
        if self.StandardOveride:
            if len(self.random_batch_indexes[index*self.batch_size:]) < self.batch_size: 
                batch_indexes = self.random_batch_indexes[index*self.batch_size:]
            else:
                batch_indexes = self.random_batch_indexes[index*self.batch_size:(index+1)*self.batch_size]

        else:
            if len(self.set_indexes) <= self.batch_size:
                batch_indexes = self.set_indexes
                #calc div metric for batch
            else:
                #calc the mean gradient of the training set
                mean_grad = np.mean(self.grads,axis=0)
                batch_indexes = []
                while len(batch_indexes) <= self.batch_size:
                    #score the distance of the batch and the item to the true gradient
                    if len(batch_indexes) == 0:
                        #batch does not yet exist so pick first item
                        r = np.random.randint(len(self.set_indexes))
                        batch_indexes.append(self.set_indexes[r])
                        self.set_indexes = np.delete(self.set_indexes,r) #r is the index
                    else:
                        #standard scoring as batch contains at least one item
                        #batch_grad_sum = np.sum(self.grads[batch_indexes],axis=0)
                        #batch_plus_items_grads = [(batch_grad_sum+i)/(len(batch_indexes)+1) for i in self.grads[self.set_indexes]]
                        #D_bg = cdist(batch_plus_items_grads,[batch_grad_sum]).flatten() #dist between the mean of the batch and item to the true grad
                        #print(np.take_along_axis(self.grads,self.set_indexes,0))
                        d = cdist(np.take(self.grads, self.set_indexes, 0),np.take(self.grads, batch_indexes, 0),'cosine')
                        print(d.shape)
                        D_ib = np.min(d,axis=1) #min dist from item to item in batch. size= avalible data
                        print(D_ib.shape)

                        #score = self.alpha * Norm(D_bg) + (1-self.alpha) * Norm(D_ib)
                        score = D_ib
                        batch_indexes.append(self.set_indexes[np.argmax(score)])
                        self.set_indexes = np.delete(self.set_indexes, np.argmax(score))


                self.set_indexes = np.array(self.set_indexes) ##Not sure why this is here???
                #print("images left in superset:",len(self.set_indexes))
                #print("images in batch:",len(batch_indexes))

        #self.scores = calc_batch_div(np.take(self.grads, batch_indexes, 0), np.mean(self.grads,axis=0)) 
        #get the data for the batch
        imgs = self.imgs[batch_indexes]
        labels = self.labels[batch_indexes]


        #convert to tensors THEY SHOULD REALLY BE STORED AS TENSORS
        imgs = tf.cast(np.array(imgs),'float32') 
        labels = tf.one_hot(np.array(labels),self.num_classes)
        return (imgs, labels,)
    
    def get_div_score(self):
        return self.scores

    def __len__(self):
        #calculates the number of batches to use
        return self.num_batches


    def Epoch_init(self,StandardOveride):
        #must be called before a training epoch
        self.StandardOveride = StandardOveride
        self.set_indexes = np.arange(self.num_images,dtype=np.int32) #indexes of avalible data [0,1,2,3,...n]
        self.num_batches = int(np.ceil(self.num_images/self.batch_size))
        print('Full amount of data used, batches: ',self.num_batches)
        if self.StandardOveride:
            #shuffle the set indexes
            self.random_batch_indexes = self.set_indexes
            np.random.shuffle(self.random_batch_indexes)



    def get_grads(self,model,index,layer_name,delay):
        #get the approximate gradients from the last layer activations 
        #This is done for all the images each itt
        #THIS HOLDS ALL GRADS 2x in memory and is baddddd
        #Turn into a tf function and do incremetally with gpu

        print("Collecting Gradients")
        imgs = tf.cast(self.imgs,'float32')
        labels = tf.one_hot(np.array(self.labels),self.num_classes)

        grads = model.predict(imgs,batch_size = 128)[0] 
        self.grads = grads - labels #this is always the full data for time being

        #modify indexes of outputs to maintain the order of the images
        #from [0,2,4] to [n,0,n,0,n,0] ect
        #self.grads = np.zeros((self.num_images,grads.shape[1]))
        #for count, idx in enumerate(self.set_indexes):
        #    self.grads[idx] = grads[count]

            

    def fast_grad_div(self):
        #calculate what the average random batch is doing based off Feng et al
        #probably would do this at epoch end as fairly compy
        #need to test both full gradients and last layer approximations
        print("Collecting Gradients")

        grads = model.predict(tf.cast(self.imgs[self.set_indexes],'float32'),batch_size = 128)[0] 
        grads = grads - tf.one_hot(np.array(self.labels[self.set_indexes]),self.num_classes)

        mean_grad = np.mean(grads,axis=0)

        #(np.dot(u, v)/np.dot(v, v))*v
        mean_dot = np.dot(mean_grad,mean_grad)
        grads_projected = [(np.dot(u,mean_grad)/mean_dot)*mean_grad for u in grads]
        #grads_orthoganal = grads - grads_projected

        #squared distance in the direction of the mean grad
        dist_para = [np.dot(u,u) for u in grads_projected] #this can be interpreted as a histogram 
        #dist_perp = [np.dot(u,u) for u in grads_orthoganal]

        return dist_para

        
class LocalSUBMODGRADDataGenV2(tf.keras.utils.Sequence):
    #added a method to try reduce the low diversity end of epoch batches (is veyr expencive)

    def __init__(self, ds_name, batch_size, size, ds_dir,alpha, Download=True,calc_stats=False):
        print("Starting Low Diversity Data Generator")
        #pull data
        train_split = 'train[:'+str(int(size*100))+'%]' 
        train_ds, info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=train_split,data_dir=ds_dir,download=Download)

        self.img_min, self.img_max, _ = collectInfoFromTFDataset(train_ds)

        #init vars
        self.calc_stats = calc_stats
        self.num_classes = info.features['label'].num_classes
        self.class_names = info.features['label'].names
        self.img_size = info.features['image'].shape
        self.num_images = info.splits[train_split].num_examples
        self.batch_size = batch_size
        self.div_score = 0
        self.alpha = alpha

        self.data_used = np.zeros(self.num_images,dtype=int)

        #sadly we need to store the dataset locally 
        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(train_ds)

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", len(self.imgs))
        print("Batch size: ", batch_size)


    def __getitem__(self, index):
        #gets the next batch of data
        #build a batch via greedy low diversity
        if self.StandardOveride:
            if len(self.random_batch_indexes[index*self.batch_size:]) < self.batch_size: 
                batch_indexes = self.random_batch_indexes[index*self.batch_size:]
            else:
                batch_indexes = self.random_batch_indexes[index*self.batch_size:(index+1)*self.batch_size]

        else:
            if len(self.set_indexes) <= self.batch_size:
                batch_indexes = self.set_indexes
                #calc div metric for batch
            else:
                #calc the mean gradient of the training set
                mean_grad = np.mean(self.grads,axis=0)
                batch_indexes = []
                while len(batch_indexes) <= self.batch_size:
                    #score the distance of the batch and the item to the true gradient
                    if len(batch_indexes) == 0:
                        #batch does not yet exist so pick first item
                        r = np.random.randint(len(self.set_indexes))
                        batch_indexes.append(self.set_indexes[r])
                        self.set_indexes = np.delete(self.set_indexes,r) #r is the index
                    else:
                        #standard scoring as batch contains at least one item
                        
                        if self.alpha != 0:
                            #dist between the mean of the batch and item to the true grad
                            batch_grad_sum = np.sum(np.take(self.grads, batch_indexes, 0),axis=0)
                            batch_plus_items_grads = [(batch_grad_sum+i)/(len(batch_indexes)+1) for i in np.take(self.grads, self.set_indexes, 0)] #batch + item for all items 
                            D_bg = cdist(batch_plus_items_grads,[mean_grad],'cosine') #angle between (batch+items , mean_grad) 2 is opposite, 1 is orthog, 0 is same direction
                            D_bg = 2 - D_bg #turn into a maximisation problem where larger is better 0 is opposite, 1 is orthog, 2 is same direction
                        else:
                            print("Skipping maximisation allignment step and setting zero")
                            D_bg = 1

                        if self.alpha != 1:
                            #min dist from item to item in batch.
                            d = cdist(np.take(self.grads, self.set_indexes, 0),np.take(self.grads, batch_indexes, 0),'cosine')
                            D_ib = np.min(d,axis=1)  #size= avalible data
                        else:
                            print("Skipping maximisation inner diversity step and setting zero")
                            D_ib = 1

                        #combine scores
                        if self.alpha == 0:
                            score = D_ib
                        elif self.alpha == 1:
                            score = D_bg
                        else:
                            score = [sum(x) for x in zip(self.alpha * Norm(D_bg), (1-self.alpha) * Norm(D_ib))]

                        batch_indexes.append(self.set_indexes[np.argmax(score)])
                        self.set_indexes = np.delete(self.set_indexes, np.argmax(score))


                self.set_indexes = np.array(self.set_indexes) ##Not sure why this is here???

        if self.calc_stats:
            self.scores = calc_batch_div(np.take(self.grads, batch_indexes, 0), np.mean(self.grads,axis=0)) 
        #get the data for the batch
        imgs = self.imgs[batch_indexes]
        labels = self.labels[batch_indexes]

        #convert to tensors THEY SHOULD REALLY BE STORED AS TENSORS
        imgs = tf.cast(np.array(imgs),'float32') 
        imgs = (imgs - self.img_min) / (self.img_max - self.img_min)
        labels = tf.one_hot(np.array(labels),self.num_classes)
        return (imgs, labels,)
    
    def get_div_score(self):
        return self.scores

    def __len__(self):
        #calculates the number of batches to use
        return self.num_batches


    def Epoch_init(self,StandardOveride):
        #must be called before a training epoch
        self.StandardOveride = StandardOveride
        self.set_indexes = np.arange(self.num_images,dtype=np.int32) #indexes of avalible data [0,1,2,3,...n]
        self.num_batches = int(np.ceil(self.num_images/self.batch_size))
        print('Full amount of data used, batches: ',self.num_batches)
        if self.StandardOveride:
            #shuffle the set indexes
            self.random_batch_indexes = self.set_indexes
            np.random.shuffle(self.random_batch_indexes)



    def get_grads(self,model,index,layer_name,delay):
        #get the approximate gradients from the last layer activations 
        #This is done for all the images each itt
        #THIS HOLDS ALL GRADS 2x in memory and is baddddd
        #Turn into a tf function and do incremetally with gpu

        print("Collecting Gradients")
        imgs = tf.cast(self.imgs,'float32')
        labels = tf.one_hot(np.array(self.labels),self.num_classes)

        grads = model.predict(imgs,batch_size = 128)[0] 
        self.grads = grads - labels #this is always the full data for time being

        #modify indexes of outputs to maintain the order of the images
        #from [0,2,4] to [n,0,n,0,n,0] ect
        #self.grads = np.zeros((self.num_images,grads.shape[1]))
        #for count, idx in enumerate(self.set_indexes):
        #    self.grads[idx] = grads[count]

            

    def fast_grad_div(self):
        #calculate what the average random batch is doing based off Feng et al
        #probably would do this at epoch end as fairly compy
        #need to test both full gradients and last layer approximations
        print("Collecting Gradients")

        grads = model.predict(tf.cast(self.imgs[self.set_indexes],'float32'),batch_size = 128)[0] 
        grads = grads - tf.one_hot(np.array(self.labels[self.set_indexes]),self.num_classes)

        mean_grad = np.mean(grads,axis=0)

        #(np.dot(u, v)/np.dot(v, v))*v
        mean_dot = np.dot(mean_grad,mean_grad)
        grads_projected = [(np.dot(u,mean_grad)/mean_dot)*mean_grad for u in grads]
        #grads_orthoganal = grads - grads_projected

        #squared distance in the direction of the mean grad
        dist_para = [np.dot(u,u) for u in grads_projected] #this can be interpreted as a histogram 
        #dist_perp = [np.dot(u,u) for u in grads_orthoganal]

        return dist_para


#Diversity batch data gen
class LocalDiffThresholdDataGen(tf.keras.utils.Sequence):

    def __init__(self, ds_name, batch_size, size, ds_dir, isEasy, k_percent, Download=True):
        print("Starting Loss Threshold Data Gen")
        #pull data
        train_split = 'train[:'+str(int(size*100))+'%]' 
        train_ds, info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=train_split,data_dir=ds_dir,download=Download)


        #init db connection and init vars
        self.isEasy = isEasy
        self.k_percent = k_percent
        self.num_classes = info.features['label'].num_classes
        self.class_names = info.features['label'].names
        self.img_size = info.features['image'].shape
        self.num_images = info.splits[train_split].num_examples
        self.batch_size = batch_size
        self.div_score = 0

        self.data_used = np.zeros(self.num_images,dtype=int)
        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(train_ds)

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", len(self.imgs))
        print("Batch size: ", batch_size)


    def __getitem__(self, index):
        #gets the next batch of data
        #build a batch by taking from the avalible data
        if len(self.random_batch_indexes[index*self.batch_size:]) < self.batch_size: 
            batch_indexes = self.random_batch_indexes[index*self.batch_size:]
        else:
            batch_indexes = self.random_batch_indexes[index*self.batch_size:(index+1)*self.batch_size]

        #get the data for the batch
        imgs = self.imgs[batch_indexes]
        labels = self.labels[batch_indexes]

        #convert to tensors
        imgs = tf.cast(np.array(imgs),'float32') 
        labels = tf.one_hot(np.array(labels),self.num_classes)
        return (imgs, labels,)
    

    def __len__(self):
        #calculates the number of batches to use
        return self.num_batches


    def Epoch_init(self,StandardOveride,model,loss_func):
        #must be called before a training epoch
        self.StandardOveride = StandardOveride
        if self.StandardOveride:
            #Use all the data
            self.set_indexes = np.arange(self.num_images)
            self.num_batches = int(np.ceil(self.num_images/self.batch_size))
            self.random_batch_indexes = self.set_indexes
            
            print('Full amount of data used, batches: ',self.num_batches)
        else:
            #use only the thresholded top easiest or hardest images
            imgs = tf.cast(self.imgs[self.set_indexes],'float32')
            labels = tf.one_hot(np.array(self.labels[self.set_indexes]),self.num_classes)

            preds = model.predict(imgs,batch_size = 128)[0] 
            losses = loss_func(labels,preds)


            if self.isEasy:
                #take the top easiest images
                smallest = np.argpartition(losses, int(len(losses)*self.k_percent))[:int(len(losses)*self.k_percent)]
                self.random_batch_indexes = self.set_indexes[smallest]
            else:
                #take the top hardest images
                largest = np.argpartition(losses, int(len(losses)*self.k_percent))[int(len(losses)*self.k_percent):]
                self.random_batch_indexes = self.set_indexes[largest]
        
        #shuffle the set indexes
        np.random.shuffle(self.random_batch_indexes)



