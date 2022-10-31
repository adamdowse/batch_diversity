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
    'max_its' : 36000,
    'mod_type' : 'Random_Bucket_full',
    'subset_type' : 'Random_Bucket', #Random_Bucket, Hard_Mining, All
    'train_type' : 'Submod', #SubMod, Random
    'activations_delay' : 1, #cannot be 0
    'k_percent' : 0.2, #percent of data to use for RB and HM
    'activation_layer_name' : 'penultimate_layer',
    }

disabled = False
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
if __name__ == "__main__":
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
    if config['momentum'] == 0:
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'],momentum=config['momentum'])

    #Data Generator
    train_DG = sf.SubModDataGen(conn_path,config)
    test_DG = sf.TestDataGen(test_ds, 50, num_classes)

    #Compile Model
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #Wandb
    sf.wandb_setup(config,disabled)
    logging_callback = WandbCallback(log_freq=1,save_model=False,save_code=False)

    #Training
    batch_num = 0
    while batch_num < config['max_its']:

        print('Batch Number: ',batch_num)

        #reset metrics
        train_loss.reset_states()
        train_acc_metric.reset_states()
        train_prec_metric.reset_states()
        train_rec_metric.reset_states()

        #Scores the training data and decides what to train on
        train_DG.get_data_subset(model)
        wandb.log({'Train_loss_hist':wandb.Histogram(train_DG.losses)},step=batch_num)

        #Train on the data subset
        for i in range(train_DG.num_batches):
            #get the activations for the next batch selection
            train_DG.get_activations(model,i)
            batch_data = train_DG.__getitem__(i)
            train_step(batch_data[0],batch_data[1])
        batch_num += train_DG.num_batches
        wandb.log({'Train_loss':train_loss.result(),'Train_acc':train_acc_metric.result(),'Train_prec':train_prec_metric.result(),'Train_rec':train_rec_metric.result()},step=batch_num)


        #Test on the test data
        test_metrics = model.evaluate(test_DG)
        wandb.log({'Test_loss':test_metrics[0],'Test_acc':test_metrics[1],'Test_prec':test_metrics[2],'Test_rec':test_metrics[3]},step=batch_num)

    wandb.log({'Images_used_hist':wandb.Histogram(train_DG.data_used)})
    #Finish
    print('Finished')