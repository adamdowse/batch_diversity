import Pretrained_supporting_functions as sf
import supporting_models as sm
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import numpy as np
import matplotlib.pyplot as plt



#/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
#DBs/
config= {
    'db_path' : "DBs/",
    'ds_name' : "cifar10",
    'train_percent' : 0.1,
    'test_percent' : 0.1,
    'group' : 't8_pretrained',
    'model_name' : 'EfficientNetV2B0_pretrained',
    'learning_rate' : 0.001,
    'momentum' : 0,
    'random_db' : 'True',
    'batch_size' : 50,
    'max_its' : 300,
    'mod_type' : 'Div_min_k0.1',
    'k_percent' : 0.5,
    'activation_layer_name' : 'penultimate_layer',
    'test_log_its' : 5,
    'train_log_its' : 25,
    }

disabled = True

if __name__ == "__main__":

    #Setup
    test_ds,ds_info,conn_path = sf.setup_db(config)

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
    #no_red_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    #Optimizer
    if config['momentum'] == 0:
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'],momentum=config['momentum'])

    #Metrics
    #train_loss = tf.keras.metrics.Mean(name='train_loss')
    #train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #test_loss = tf.keras.metrics.Mean(name='test_loss')
    #test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    #Data Generator
    train_DG = sf.SubModDataGen(conn_path,config)
    test_DG = sf.TestDataGen(test_ds, 50, num_classes)

    #Compile Model
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #Wandb
    sf.wandb_setup(config,disabled)
    logging_callback = WandbCallback(log_freq=1,save_model=False)

    #Training
    for i in range(config['max_its']):
        #Scores the training data and decides what to train on
        train_DG.get_data_subset(model)

        #Log metrics on selected data
        wandb.log({'Train_loss_hist':wandb.Histogram(train_DG.losses)},step=i)

        #Train on the data subset
        print('Training')
        model.fit(train_DG,epochs=1,verbose=1,callbacks=[logging_callback])


        #Testing - may have problem with batching and lable shape
        print('Evaluating')
        model.evaluate(test_DG,verbose=1,callbacks=[logging_callback])


    wandb.log({'Images_used_hist':wandb.Histogram(train_DG.used_imgs)})
    #Finish
    print('Finished')