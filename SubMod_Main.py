import SubMod_supporting_functions as sf
import supporting_models as sm
import tensorflow as tf
import wandb
import numpy as np
import matplotlib.pyplot as plt


@tf.function
def train_step(imgs,labels):
    print(labels)
    print(imgs)
    with tf.GradientTape() as tape:
        preds = model(imgs,training=True)
        loss = loss_func(labels,preds)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer[0].apply_gradients(zip(grads,model.trainable_variables))
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
#/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
#/DBs/
config= {
    'db_path' : "DBs/",
    'ds_name' : "mnist",
    'train_percent' : 0.01,
    'test_percent' : 0.01,
    'group' : 't7',
    'model_name' : 'Simple_CNN',
    'modifiers' : [0.1,0.1,0.1,0.1],
    'learning_rate' : 0.01,
    'random_db' : 'True',
    'batch_size' : 32,
    'max_its' : 2000
    }

disabled = True

if __name__ == "__main__":
    #Setup
    test_ds,ds_info,conn = sf.setup_db(config)

    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    img_size = ds_info.features['image'].shape
    num_train_imgs = ds_info.splits['train[:'+str(int(config['train_percent']*100))+'%]'].num_examples

    #Model
    model = sm.select_model(config['model_name'],num_classes,img_size)
    model.build(img_size+(1,))
    model.summary()

    #Loss
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    #Optimizer
    optimizer = [tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])]

    #Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    #Data Generator
    train_DG = sf.SubModDataGen(conn,config['batch_size'],config['modifiers'])

    #Training
    for b in range(config['max_its']):
        #Reset the metrics at the start of the next batch
        train_loss.reset_states()
        train_acc_metric.reset_states()
        test_loss.reset_states()
        test_acc_metric.reset_states()

        #Training
        train_DG.score_images(model)
        for i, (imgs,labels) in enumerate(train_DG): #this should just be one batch
            print("Batch shape: ", imgs.shape, labels.shape)
            train_step(imgs,labels)

        #Testing
        if b % 50 == 0:
            for imgs,labels in test_ds:
                test_step(imgs,labels)
        

        #Logging
        if b % 50 == 0 and not disabled:
            wandb.log({'batch_num': b,
                    'train_loss': train_loss.result(),
                    'train_acc': train_acc_metric.result(),
                    'test_loss': test_loss.result(),
                    'test_acc': test_acc_metric.result()})
        elif not disabled:
            wandb.log({'batch_num': b,
                    'train_loss': train_loss.result(),
                    'train_acc': train_acc_metric.result()})

        
        #Print
        print('Batch: {}, Train Loss: {}, Train Acc: {}, Test Loss: {}, Test Acc: {}'.format(b,train_loss.result(),train_acc_metric.result(),test_loss.result(),test_acc_metric.result()))

    #Finish
    print('Finished')