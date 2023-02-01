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
    return (np.array(imgs_store),np.array(labels_store),num_batches)

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
        print('INIT: Using ',ds_name, ' test data')
        test_split = 'test[:'+str(int(size*100))+'%]'
        test_ds,info = tfds.load(ds_name,with_info=True,shuffle_files=False,as_supervised=True,split=test_split,data_dir=ds_dir,download=Download)

        self.test_ds = test_ds.batch(batch_size)
        self.batch_size = batch_size

        self.num_classes = info.features['label'].num_classes
        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(self.test_ds)
        
        
    def __getitem__(self, index):
        return (tf.cast(self.imgs[index],'float32'), tf.one_hot(self.labels[index],self.num_classes),)
    
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

            


#Diversity batch data gen
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
            #select a random avalible item
            batch_index = self.set_indexes[np.random.randint(len(self.set_indexes))]

            #fill the batch with the low diversity gradients
            #find closest n-1 grads
            smallest = np.argpartition(np.array(cdist(self.grads[self.set_indexes],[self.grads[batch_index]])).flatten(), self.batch_size)[:self.batch_size]
            batch_indexes = self.set_indexes[smallest]
            
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
        if index % delay == 0 and self.StandardOveride == False:
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


