import Pretrained_supporting_functions as sf
import supporting_models as sm
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
import os


#/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
#DBs/



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

    wandb.init(project='k_diversity',entity='adamdowse')
    config= {
        'db_path' : "/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/",
        'ds_name' : "cifar10",
        'train_percent' : 1,
        'test_percent' : 1,
        'group' : 't9_hyperparam_search',
        'model_name' : 'Simple_CNN',
        'learning_rate' : wandb.config.learning_rate,
        'optimizer' : wandb.config.optimizer,
        'momentum' : 0,
        'random_db' : 'True',
        'batch_size' : wandb.config.batch_size,
        'data_aug' : '0', #0 = no data aug, 1 = data aug, 2 = data aug + noise
        'max_its' : 30000,
        'early_stop' : 5000,
        'mod_type' : 'Random_Bucket_full',
        'subset_type' : wandb.config.run_type[0], #Random_Bucket, Hard_Mining, All
        'train_type' : wandb.config.run_type[1], #SubMod, Random
        'activations_delay' : 4, #cannot be 0
        'k_percent' : 1, #percent of data to use for RB and HM
        'activation_layer_name' : 'penultimate_layer',
    }

    #Setup
    test_ds,ds_info,conn_path, train_ds = sf.setup_db(config)

    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    img_size = ds_info.features['image'].shape
    num_train_imgs = ds_info.splits['train[:'+str(int(config['train_percent']*100))+'%]'].num_examples

    #Model
    tf.keras.backend.clear_session()
    model = sm.select_model(config['model_name'],num_classes,img_size)
    model.build(img_size+(1,))
    model.summary()

    #Load pretrained weights TODO: make this a function

    #Loss
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
    #Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
    train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

    #Optimizer
    if config['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    elif config['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'Momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'],momentum=config['momentum'])
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
    while batch_num < config['max_its']:
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
        batch_num += train_DG.num_batches

        if test_metrics[1] > early_stop_max:
            early_stop_max = test_metrics[1]
            early_stop_count = 0
        if early_stop_count > config['early_stop']:
            break

    wandb.log({'Images_used_hist':wandb.Histogram(train_DG.data_used)})
    #clear keras backend
    tf.keras.backend.clear_session()
    #Finish
    print('Finished')


if __name__ == "__main__":
    sweep_configuration = {
        'method': 'grid',
        'name': 'Test Acc vs Batch Size',
        'metric': {
            'goal': 'maximize', 
            'name': 'Test_acc'
            },
        'parameters': {
            'batch_size': {'values': [4,8,16,32,64,128,256,512]},
            'learning_rate': {'values': [0.00001,0.0001,0.001,0.01]},
            'run_type': {'values': [['All','Random'],['All','SubMod']]}, #,['Hard_Mining','SubMod'],['Hard_Mining','Random']
            'optimizer': {'values': ['SGD','Adam','Momentum']},
            }
        }

    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='k_diversity')
    wandb.agent('adamdowse/k_diversity/s9l3q894', function=main, count=1)



    