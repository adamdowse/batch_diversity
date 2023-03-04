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



#run a standard model with the trace FIM recorded each epoch

def main():

    @tf.function
    def train_step(imgs,labels):
        with tf.GradientTape() as tape:
            preds = model(imgs,training=True)
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
            preds = model(imgs,training=False)
            loss = loss_func(labels,preds)
        test_loss(loss)
        test_acc_metric(labels,preds)
        test_prec_metric(labels,preds)
        test_rec_metric(labels,preds)

    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/
    #/com.docker.devenvironments.code/datasets/
    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
    config= {
        'ds_path' : "/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/",
        'db_path' : "/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/",
        'ds_name' : "CIFAR10",
        'train_percent' : 1,
        'test_percent' : 1,
        'group' : 'full_SOTA',
        'model_name' : 'CIFAR10_ViT',
        'learning_rate' : 0.01,
        'learning_rate_decay' : 0.97,
        'optimizer' : 'Adam', #SGD, Adam, Momentum
        'momentum' : 0,
        'random_db' : 'True', #False is wrong it adds the datasets together
        'batch_size' : 128,
        'label_smoothing' : 0,
        'weight_decay' : 0,
        'data_aug' : '0', #0 = no data aug, 1 = data aug, 2 = data aug + noise
        'start_defect_epoch' : 1000,
        'defect_length' : 10, # length of defect in epochs
        'max_its' : 46900, #=100 full dataset epochs
        'epochs'    : 0, #if this != 0 then it will override max_its    
        'early_stop' : 5000,
        'subset_type' : 'Easy_Mining', #Random_Bucket, Hard_Mining, All
        'train_type' : 'Random', #SubMod, Random
        'activations_delay' : 4, #cannot be 0 (used when submod is used)
        'k_percent' : 0.5, #percent of data to use for RB and HM
        'activation_layer_name' : 'penultimate_layer',
    }

    #Setup
    wandb.init(project='CIFAR10_Sens',config=config)

    #Data Generator
    if config['subset_type'] == 'Easy_Mining': isEasy = True
    elif config['subset_type'] == 'Hard_Mining': isEasy = False
    else: print("ERROR with subset type")
    train_DG = DataGens.LocalDiffThresholdDataGen(config['ds_name'],config['batch_size'],config['train_percent'],config['ds_path'],isEasy,config['k_percent'])
    test_DG = DataGens.TestDataGen(config['ds_name'], 50,config['test_percent'], config['ds_path'])

    #Model
    tf.keras.backend.clear_session()
    model = sm.select_model(config['model_name'],train_DG.num_classes,train_DG.img_size,config['weight_decay'])
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
    if config['learning_rate_decay'] == 1 or config['learning_rate_decay'] == 0:
        lr_schedule = config['learning_rate']
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config['learning_rate'],
            decay_steps=train_DG.num_images/config['batch_size'],
            decay_rate=config['learning_rate_decay'],
            staircase=True)

    if config['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    elif config['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    elif config['optimizer'] == 'Momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,momentum=config['momentum'])
    else:
        print('Optimizer not recognised')     

    
    #Compile Model
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #Wandb
    #sf.wandb_setup(config,disabled)
    #logging_callback = WandbCallback(log_freq=1,save_model=False,save_code=False)

    #Training
    batch_num = 0
    early_stop_max = 0
    early_stop_count = 0
    epoch_num = 0
    while batch_num < config['max_its']:
        epoch_num += 1
        print('Batch Number: ',batch_num)
        print('Epoch Number: ',epoch_num)

        #reset metrics
        train_loss.reset_states()
        train_acc_metric.reset_states()
        train_prec_metric.reset_states()
        train_rec_metric.reset_states()
        test_loss.reset_states()
        test_acc_metric.reset_states()
        test_prec_metric.reset_states()
        test_rec_metric.reset_states()

        #Scores the training data and decides what to train on
        print('Getting Subset')
        if epoch_num > config['start_defect_epoch'] and epoch_num < config['start_defect_epoch']+(config['defect_length']*config['k_percent']):
            #this initilises the data generator with the defect data 
            train_DG.Epoch_init(False,model,loss_func)
        else:
            #this initilises the data generator with the normal data
            train_DG.Epoch_init(True,model,loss_func)

        #wandb.log({'Train_loss_hist':wandb.Histogram(train_DG.losses)},step=batch_num)
        
        #Train on the data subset
        print('Training')
        for i in range(train_DG.num_batches):
            #get the activations for the next batch selection
            batch_data = train_DG.__getitem__(i)
            train_step(batch_data[0],batch_data[1])
            
        #Test on the test data
        print('Evaluating')
        for i in range(test_DG.num_batches):
            batch_data = test_DG.__getitem__(i)
            test_step(batch_data[0],batch_data[1])

        #Log metrics
        wandb.log({'Train_loss':train_loss.result(),'Train_acc':train_acc_metric.result(),'Train_prec':train_prec_metric.result(),'Train_rec':train_rec_metric.result()},step=batch_num)
        wandb.log({'Test_loss':test_loss.result(),'Test_acc':test_acc_metric.result(),'Test_prec':test_prec_metric.result(),'Test_rec':test_rec_metric.result()},step=batch_num)
        wandb.log({'Epoch':epoch_num},step=batch_num)
        batch_num += train_DG.num_batches

        #get FIM
        train_DG.Epoch_init(True,model,loss_func)
        FIM_trace = fim.FIM_trace(train_DG,train_DG.num_classes,model)

        #Log FIM
        print('FIM Trace: ',FIM_trace)
        wandb.log({'Approx_Trace_FIM': FIM_trace},step=batch_num)

        #Early stopping
        #if test_metrics[1] > early_stop_max:
        #    early_stop_max = test_metrics[1]
        #    early_stop_count = 0
        #else:
        #    early_stop_count += train_DG.num_batches
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
    if False:
        sweep_configuration = {
            'method': 'grid',
            'name': 'Test Acc vs Batch Size big batches 2',
            'metric': {
                'goal': 'maximize', 
                'name': 'Test_acc'
                },
            'parameters': {
                'batch_size': {'values': [1024]},
                'learning_rate': {'values': [0.01,0.001,0.0001,0.00001]},
                'run_type': {'values': ['Random','SubMod']}, #,['Hard_Mining','SubMod'],['Hard_Mining','Random']
                'optimizer': {'values': ['SGD','Adam']},
                }
            }

        
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='k_diversity')
        wandb.agent(sweep_id, function=main)
