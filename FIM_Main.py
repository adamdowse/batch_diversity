import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import FIM_analysis as fim
import Pretrained_supporting_functions as sf
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

    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/datasets/
    #/com.docker.devenvironments.code/datasets/
    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
    config= {
        'ds_path' : "/com.docker.devenvironments.code/datasets/",
        'db_path' : "DBs/",
        'ds_name' : "cifar10",
        'train_percent' : 0.1,
        'test_percent' : 0.1,
        'group' : '0.1cifar10',
        'model_name' : 'Simple_CNN',
        'learning_rate' : 0.0001,
        'learning_rate_decay' : 0,
        'optimizer' : 'SGD', #SGD, Adam, Momentum
        'momentum' : 0,
        'random_db' : 'True', #False is wrong it adds the datasets together
        'batch_size' : 128,
        'label_smoothing' : 0,
        'weight_decay' : 0.01,
        'data_aug' : '0', #0 = no data aug, 1 = data aug, 2 = data aug + noise
        'max_its' : 420000,
        'epochs'    : 400, #if this != 0 then it will override max_its    
        'early_stop' : 5000,
        'subset_type' : 'All', #Random_Bucket, Hard_Mining, All
        'train_type' : 'Random', #SubMod, Random
        'activations_delay' : 4, #cannot be 0 (used when submod is used)
        'k_percent' : 1, #percent of data to use for RB and HM
        'activation_layer_name' : 'penultimate_layer',
    }

    #Setup
    wandb.init(project='FIM',config=config)
    test_ds,ds_info,conn_path, train_ds = sf.setup_db(config)

    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    img_size = ds_info.features['image'].shape
    num_train_imgs = ds_info.splits['train[:'+str(int(config['train_percent']*100))+'%]'].num_examples

    #Model
    tf.keras.backend.clear_session()
    model = sm.select_model(config['model_name'],num_classes,img_size,config['weight_decay'])
    model.build(img_size+(1,))
    model.summary()

    #Load pretrained weights TODO: make this a function

    #Loss
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False,label_smoothing=config['label_smoothing'])
    
    #Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
    train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

    #Optimizer
    if config['learning_rate_decay'] == 1 or config['learning_rate_decay'] == 0:
        lr_schedule = config['learning_rate']
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config['learning_rate'],
            decay_steps=num_train_imgs/config['batch_size'],
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

    #Data Generator
    train_DG = sf.SubModDataGen(conn_path,config)
    test_DG = sf.TestDataGen(test_ds, 50, num_classes)

    #Compile Model
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #Wandb
    #sf.wandb_setup(config,disabled)
    logging_callback = WandbCallback(log_freq=1,save_model=False,save_code=False)

    #Training
    batch_num = 0
    early_stop_max = 0
    early_stop_count = 0
    epoch_num = 0
    while batch_num < config['max_its'] and epoch_num < config['epochs']:
        epoch_num += 1
        print('Batch Number: ',batch_num)

        #reset metrics
        train_loss.reset_states()
        train_acc_metric.reset_states()
        train_prec_metric.reset_states()
        train_rec_metric.reset_states()

        #Scores the training data and decides what to train on
        print('Getting Subset')
        train_DG.get_data_subset(model,train_ds)
        #wandb.log({'Train_loss_hist':wandb.Histogram(train_DG.losses)},step=batch_num)
        
        #Train on the data subset
        print('Training')
        for i in range(train_DG.num_batches):
            #get the activations for the next batch selection
            train_DG.get_activations(model,i)
            batch_data = train_DG.__getitem__(i)
            train_step(batch_data[0],batch_data[1])
            
        
        
        #Test on the test data
        print('Evaluating')
        test_metrics = model.evaluate(test_DG)

        #Log metrics
        wandb.log({'Train_loss':train_loss.result(),'Train_acc':train_acc_metric.result(),'Train_prec':train_prec_metric.result(),'Train_rec':train_rec_metric.result()},step=batch_num)
        wandb.log({'Test_loss':test_metrics[0],'Test_acc':test_metrics[1],'Test_prec':test_metrics[2],'Test_rec':test_metrics[3]},step=batch_num)
        wandb.log({'Epoch':epoch_num},step=batch_num)
        batch_num += train_DG.num_batches

        #FIM Analysis
        FIM_trace = fim.FIM_trace(train_DG,num_classes,model) #return the approximate trace of the FIM

        #Log FIM
        wandb.log({'Approx_Trace_FIM': FIM_trace},step=batch_num)

        #Early stopping
        #if test_metrics[1] > early_stop_max:
        #    early_stop_max = test_metrics[1]
        #    early_stop_count = 0
        #else:
        #    early_stop_count += train_DG.num_batches
        #    if early_stop_count > config['early_stop']:
        #        break

    wandb.log({'Images_used_hist':wandb.Histogram(train_DG.data_used)})
    #clear keras backend
    tf.keras.backend.clear_session()
    #Finish
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



    