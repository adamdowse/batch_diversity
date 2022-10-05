import supporting_functions as sf
import tensorflow as tf
#import tensorflow_addons as tfa
import wandb
import numpy as np
#import sklearn
import matplotlib.pyplot as plt
#import time
#import tracemalloc


@tf.function
def train_step(imgs,labels):
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
    'ds_name' : "cifar10",
    'train_percent' : 1,
    'test_percent' : 1,
    'group' : 't3_o_test',
    'model_name' : 'AlexNet',
    'learning_rate' : 0.01,
    'warm_start_batches' : 0,
    'batch_size' : 32,
    'max_its' : 2000,
    'k' : 1,
    'des_inner' : np.random.rand(),
    'des_outer' : np.random.rand(),
    'random_db' : 'False',
    'run_type'  : 'o'
    }

disabled = True

if __name__ == "__main__":
    #Setup
    test_ds,train_ds,ds_info,conn = sf.setup_db(config)

    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    img_size = ds_info.features['image'].shape
    num_train_imgs = ds_info.splits['train[:'+str(int(config['train_percent']*100))+'%]'].num_examples
    num_test_imgs = ds_info.splits['test[:'+str(int(config['test_percent']*100))+'%]'].num_examples

    print('# Classes=',num_classes)
    print('Class Names=',class_names)
    print('Img size=',img_size)
    print('train imgs=',num_train_imgs)
    print('test imgs=',num_test_imgs)

    sf.setup_logs(config,disabled)

    model,optimizer,loss_func,train_loss,train_acc_metric,test_loss,test_acc_metric = sf.setup_model(config,num_classes,img_size)

    train_data_gen = sf.CustomDBDataGen(
        conn = conn,
        batch_size = config['batch_size'], 
        num_classes = num_classes,
        num_images = num_train_imgs,
        warm_start_batches = config['warm_start_batches']
    )

    #Main loop
    print("Started Training")
    train_it = 0
    test_it = 0 
    test_cap = 20

    images_used = np.zeros(num_train_imgs)
    mean_saved_gradients = None
    
    while train_it < config['max_its']:
        print("Iteration:",train_it)
        if train_it >= config['warm_start_batches']:
            mean_saved_gradients = sf.sample_batches(config['run_type'],model,train_it,train_ds,config['batch_size'],num_classes,conn,des_inner=config['des_inner'],des_outer=config['des_outer'],mean_saved_gradients=mean_saved_gradients,k=config['k'])

        for i, (X,Y) in enumerate(train_data_gen):
            #train function
            train_step(X[1],Y)
            train_it += 1
        
        if train_it % 10 == 0:
            #calc stats and logging
            wandb.log({ 'train_acc':train_acc_metric.result().numpy(),
                        'train_loss':train_loss.result().numpy()},step=train_it)

        #reset the batch numbers to 0
        train_data_gen.on_epoch_end()

        #test model
        if test_it > test_cap:
            for X,Y in test_ds.batch(100):
                Y = tf.one_hot(Y,num_classes)
                test_step(X,Y)
            test_it = 0

            wandb.log({'test_acc':test_acc_metric.result().numpy(),
                    'test_loss':test_loss.result().numpy()},step=train_it)
        else:
            test_it += 1

    #finished training
    #final logging
    cm = np.zeros((num_classes,num_classes))
    for X,Y in test_ds.batch(1):
        cm[model.predict(X).argmax(),Y] += 1

    fig,ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(num_classes), labels=class_names)
    ax.set_yticks(np.arange(num_classes), labels=class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="w")
    ax.set_title("CM")
    fig.tight_layout()
    wandb.log({"conf_mat": fig})

    fig1,ax1 = plt.subplots()
    im1 = ax1.hist(images_used)
    fig1.tight_layout()

    wandb.log({'images_used':wandb.Image(fig1)})
        
