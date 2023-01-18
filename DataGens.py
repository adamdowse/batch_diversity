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


def imgsAndLabelsFromTFDataset(DS,num_classes):
    imgs = []
    labels = []
    num_batches = 0
    for imgs, labels in DS:
        num_batches += 1
        imgs.append(imgs)
        labels.append(tf.one_hot(labels, num_classes))
    return (imgs,labels,num_batches)


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
        return np.zeros(len(indexes))

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








class TestDataGen(tf.keras.utils.Sequence):
    def __init__(self, test_ds, batch_size, num_classes):
        self.test_ds = test_ds.batch(batch_size)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(self.test_ds,self.num_classes)
        
    def __getitem__(self, index):
        return (self.imgs[index], self.labels[index],)
    
    def __len__(self):
        return self.num_batches





#The train data generator
class LocalSubModDataGen(tf.keras.utils.Sequence):
    #This Generator is used to generate batches of data for the training of the model via submodular selection
    def __init__(self, train_ds, num_classes, config, lambdas = [0.2,0.2,0.2,0.2]):
        print("Starting SubMod Data Generator")
        #init db connection and init vars
        self.config = config
        self.num_classes = num_classes
        self.lambdas = lambdas
        self.StandardOveride = False

        self.data_used = np.zeros(self.num_images,dtype=int)
        self.imgs, self.labels, self.num_batches = imgsAndLabelsFromTFDataset(train_ds,self.num_classes)

        #Logging
        print("Number of classes: ", self.num_classes)
        print("Number of images: ", len(self.imgs))
        print("Batch size: ", config['batch_size'])


    def __getitem__(self, index,model,):
        #gets the next batch of data
        #build a batch via submodular selection
        if self.StandardOveride:
            random_batch_idxs = random.sample([i for i in range(self.set_indexes)], self.config["batch_size"])
            self.batch_indexes = self.set_indexes[random_batch_idxs]
            self.set_indexes = np.delete(self.set_indexes,random_batch_idxs)
        else:
            #gets the indexes of the images to use in the next batch and removes them from the set of available images
            self.batch_indexes = np.array([],dtype=int)

            if len(self.set_indexes) < self.config['batch_size']: 
                self.batch_indexes = self.set_indexes
            else: 
                
                for i in range(self.config['batch_size']):
                    #calculate the scores for the subset if an item is added from the superset.
                    scores = lambdas[0] * Norm(Uncertainty_Score(self.preds,self.batch_indexes)) +
                            lambdas[1] * Norm(Entropy_Score(self.activations,self.set_indexes,self.batch_indexes)) +
                            lambdas[2] * Norm(Dist_To_Mean_Score(self.activations,self.set_indexes,self.batch_indexes)) +
                            lambdas[3] * Norm(Feature_Match_Score(self.activations,self.set_indexes,self.batch_indexes))

                    #if the sum of the scores is 0, then select a random image
                    if np.sum(scores) == 0:
                        #select a random image
                        print('Random Data taken as scores == 0')
                        max_index = np.random.randint(0,len(scores))
                    else:
                        #get the index of the image with the highest score
                        max_index = np.argmax(scores)

                    #add the index to the set
                    self.batch_indexes = np.append(self.batch_indexes,self.set_indexes[max_index])
                    #remove the index from the indexes
                    self.set_indexes = np.delete(self.set_indexes,max_index)


        #get the data for the batch
        imgs = self.imgs[self.batch_indexes]
        labels = self.labels[self.batch_indexes]

        #convert to tensors
        imgs = tf.cast(np.array(imgs),'float32') #labels are already done in init phase

        #reset the set indexes
        self.batch_indexes = np.array([],dtype=int)

        return (imgs, labels,)

    def __len__(self):
        #calculates the number of batches to use
        return self.num_batches


    def Epoch_init(self,StandardOveride):

        self.StandardOveride = StandardOveride
        #must be called before a training epoch
        #Use all the data
        print('Full amount of data used')
        self.set_indexes = np.arange(self.num_images)
        self.num_batches = int(np.ceil(self.num_images/self.config['batch_size']))
        print('num batches in epoch: ',self.num_batches)

    def get_activations(self,model,index)
        #get the activations of the model for each image in the subset so that the subset_index aligns with activations
        if index % self.config['activations_delay'] == 0:
            print("Collecting Activations")
            imgs = self.imgs[self.set_indexes]
        
            inter_model = Model(inputs=model.input, outputs=[model.get_layer(self.config['activation_layer_name']).output,model.output])
            local_activations,preds = inter_model.predict(imgs,batch_size = 128)
            del inter_model

            #modify indexes of outputs to maintain the order of the images
            #from [0,2,4] to [n,0,n,0,n,0] ect
            self.activations = np.zeros((self.num_images,local_activations.shape[1]))
            self.preds = np.zeros((self.num_images,preds.shape[1]))
            for count, idx in enumerate(self.set_indexes):
                self.activations[idx] = activations[count]
                self.preds[idx] = preds[count]




        


