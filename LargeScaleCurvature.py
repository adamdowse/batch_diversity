import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import FIM_analysis as fim
import Pretrained_supporting_functions as sf
import DataGens
import supporting_models as sm
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
import time
import tracemalloc
import os


#This experiment is to measure the differences between points in network training.

def main():
    @tf.function
    def train_step(imgs,labels):
        with tf.GradientTape() as tape:
            preds = model(imgs,training=True)[0]
            loss = loss_func(labels,preds)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        train_loss(loss)
        train_acc_metric(labels,preds)
        train_prec_metric(labels,preds)
        train_rec_metric(labels,preds)
        return
    
    @tf.function
    def test_step(imgs,labels):
        with tf.GradientTape() as tape:
            preds = model(imgs,training=False)[0]
            loss = loss_func(labels,preds)
        test_loss(loss)
        test_acc_metric(labels,preds)
        test_prec_metric(labels,preds)
        test_rec_metric(labels,preds)

    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/
    #/com.docker.devenvironments.code/datasets/
    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
    config= {
        'ds_path' : "/com.docker.devenvironments.code/datasets/",
        'db_path' : "/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/",
        'ds_name' : "mnist",
        'group' : 'Mk1_leapfrog',
        'train_percent' : 0.1,
        'test_percent' : 0.1,
        'model_name' : 'ResNet18',
        'learning_rate' : 0.01,
        'learning_rate_decay' : 0,
        'optimizer' : 'SGD', #SGD, Adam, Momentum
        'momentum' : 0,
        'random_db' : 'True', #False is wrong it adds the datasets together
        'batch_size' : 128,
        'label_smoothing' : 0,
        'weight_decay' : 0,
        'data_aug' : '0', #0 = no data aug, 1 = data aug, 2 = data aug + noise
        'epochs'    : 20, #if this != 0 then it will override max_its    
        'n_record_epochs': 5, #the number of epochs to train before then curvature is calculated
        'error_scales' : [0.0001,0.001,0.01,0.1,1,10,100] , #the number of error scales to calculate the curvature at
        'n_data_points' : 30000 #the number of data points to calculate the curvature at
    }

    #Setup
    wandb.init(project='Large Scale Curvature',config=config)

    #Data Generator
    train_DG = DataGens.LocalSUBMODGRADDataGenV2(config['ds_name'],config['batch_size'],config['train_percent'],config['ds_path'],0,calc_stats=False)
    test_DG = DataGens.TestDataGen(config['ds_name'], 50, config['test_percent'], config['ds_path'])
    second_train_DG = DataGens.LocalSUBMODGRADDataGenV2(config['ds_name'],128,config['train_percent'],config['ds_path'],0,calc_stats=False)

    #Model
    tf.keras.backend.clear_session()
    model = sm.select_model(config['model_name'],train_DG.num_classes,train_DG.img_size,config['weight_decay'],getLLactivations=True)
    model.build(train_DG.img_size+(1,))
    model.summary()

    #Loss
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False,label_smoothing=config['label_smoothing'])
    
    #Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
    train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_prec_metric = tf.keras.metrics.Precision(name='test_precision')
    test_rec_metric = tf.keras.metrics.Recall(name='test_recall')

    #Optimizer
    if config['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    elif config['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'Momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'],momentum=config['momentum'])
    else:
        print('Optimizer not recognised')     

    print("Learning Rate :",optimizer.learning_rate)

    #Compile Model
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #Training
    batch_num = 0
    epoch_num = 0
    while epoch_num <= config['epochs']:
        epoch_num += 1
        print('Epoch #: ',epoch_num,' Batch #: ',batch_num)

        #reset metrics
        train_loss.reset_states()
        train_acc_metric.reset_states()
        train_prec_metric.reset_states()
        train_rec_metric.reset_states()
        test_loss.reset_states()
        test_acc_metric.reset_states()
        test_prec_metric.reset_states()
        test_rec_metric.reset_states()

        #this initilises the data generator with the normal data
        second_train_DG.Epoch_init(True)
        train_DG.Epoch_init(True)

        #if the epoch is a record epoch then calculate the curvature
        if epoch_num % config['n_record_epochs'] == 0 or epoch_num == 2:
            #calculate the curvature
            curvature, curv_var = fim.CurvatureEstimate(model,second_train_DG,config['error_scales'],config['n_data_points'])
            #log the curvature
            for i in config['error_scales']:
                wandb.log({'Curvature_'+str(i):curvature[i]},step=batch_num)
        
        #Train on the data
        print('Training')
        for i in range(train_DG.num_batches):
            t1 = time.time()
            batch_data = train_DG.__getitem__(i)
            t = time.time()
            train_step(batch_data[0],batch_data[1])
            print('Get data: ',t-t1,'train step time: ',time.time() - t)
            batch_num += 1

        #Test on the test data
        print('Evaluating')
        for i in range(test_DG.num_batches):
            batch_data = test_DG.__getitem__(i)
            test_step(batch_data[0],batch_data[1])

        #Log metrics
        wandb.log({'Train_loss':train_loss.result(),'Train_acc':train_acc_metric.result(),'Train_prec':train_prec_metric.result(),'Train_rec':train_rec_metric.result()},step=batch_num)
        wandb.log({'Test_loss':test_loss.result(),'Test_acc':test_acc_metric.result(),'Test_prec':test_prec_metric.result(),'Test_rec':test_rec_metric.result()},step=batch_num)
        wandb.log({'Learning_rate':optimizer.learning_rate},step=batch_num)
        wandb.log({'Epoch':epoch_num},step=batch_num)
    
        #Grad Analysis
        #second_train_DG.Epoch_init(True)
        #mean_grad_activity, mean_grad_var = fim.Grad_Div_Ensemble_Method(second_train_DG,model)
        #wandb.log({"mean_grad_activity":mean_grad_activity,"mean_grad_var":mean_grad_var},step=batch_num)

        #FIM Analysis
        #second_train_DG.Epoch_init(True)
        #FIM_trace, FIM_trace_var = fim.FIM_trace_with_var(second_train_DG,second_train_DG.num_classes,model) #return the approximate trace of the FIM
        #Test_FIM_trace, Test_FIM_trace_var = fim.FIM_trace_with_var(test_DG,second_train_DG.num_classes,model)
        #wandb.log({'Approx_Trace_FIM': FIM_trace,"Test_Trace_FIM":Test_FIM_trace,'FIM_var': FIM_trace_var,"Test_FIM_var":Test_FIM_trace_var},step=batch_num)




        #Early stopping
        # if test_acc_metric.result() > early_stop_max:
        #    early_stop_max = test_acc_metric.result()
        #    early_stop_count = 0
        # else:
        #    early_stop_count += 1
        #    if early_stop_count > config['early_stop']:
        #        break


    #wandb.log({'Images_used_hist':wandb.Histogram(train_DG.data_used)})

    #Finish - clear keras backend
    tf.keras.backend.clear_session()
    print('Finished')


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    main()