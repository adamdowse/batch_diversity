import SubMod_supporting_functions as sf
import supporting_models as sm
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import numpy as np
import matplotlib.pyplot as plt



#/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
#/DBs/
config= {
    'db_path' : "DBs/",
    'ds_name' : "cifar10",
    'train_percent' : 0.1,
    'test_percent' : 0.1,
    'group' : 't7_SubMod',
    'model_name' : 'Simple_CNN',
    'modifiers' : [0.2,0.1,0.5,0.2],
    'learning_rate' : 0.001,
    'momentum' : 0.9,
    'random_db' : 'True',
    'batch_size' : 50,
    'max_its' : 100
    }

disabled = False

if __name__ == "__main__":

    @tf.function
    def train_step(imgs,labels):
        with tf.GradientTape() as tape:
            preds = model(imgs,training=True)
            loss = loss_func(labels,preds)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        train_loss(loss)
        train_acc_metric(labels,preds)
        return

    @tf.function
    def test_step(imgs, labels):
        preds = model(imgs, training=False)
        t_loss = loss_func(labels,preds)
        m_loss = tf.math.reduce_mean(t_loss)
        test_loss(m_loss)
        test_acc_metric(labels, preds)
        return
    
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

    #Loss
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    #Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'],momentum=config['momentum'])

    #Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    #Data Generator
    train_DG = sf.SubModDataGen(conn_path,config['batch_size'],config['modifiers'])

    #Compile Model
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy','mse'])

    #Wandb
    sf.wandb_setup(config,disabled)
    logging_callback = WandbCallback(log_freq=10,save_model=False)


    #Training
    for b in range(config['max_its']):
        #Reset the metrics at the start of the next batch
        train_loss.reset_states()
        train_acc_metric.reset_states()
        test_loss.reset_states()
        test_acc_metric.reset_states()

        #Training
        train_DG.get_activations(model)
        model.fit(train_DG,epochs=1,callbacks= [logging_callback])
        #print(hist.history['accuracy'])
        train_DG.on_epoch_end()
        
        #Testing
        for img, label in test_ds.batch(config['batch_size']):
            label = tf.one_hot(label,num_classes)
            test_step(img,label)
        wandb.log({'test_loss': test_loss.result(),'test_acc': test_acc_metric.result()})
        '''
        #Logging
        if b % 20 == 0 and not disabled:
            wandb.log({'batch_num': b,
                    'train_loss': hist.history['loss'][0],
                    'train_acc': hist.history['accuracy'][0],
                    'test_loss': test_loss.result(),
                    'test_acc': test_acc_metric.result()})
        elif not disabled:
            wandb.log({'batch_num': b,
                    'train_loss': hist.history['loss'][0],
                    'train_acc': hist.history['accuracy'][0]})

        if b % 100 == 0:
            wandb.log({'used_images': train_DG.img_count_store})
        '''
    #Finish
    print('Finished')