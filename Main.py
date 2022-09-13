import supporting_functions as sf
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
import numpy as np
import sklearn


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

config= {
    'db_path' : "db/",
    'ds_name' : "mnist",
    'train_percent' : 0.01,
    'test_percent' : 0.01,
    'group' : 'i_score_results',
    'model_name' : 'Simple_CNN',
    'learning_rate' : 0.01,
    'warm_start' : 1,
    'batch_size' : 32,
    'max_its' : 200,
    'k' : 1,
    'des_inner' : 0.1,
    'des_outer' : 1,
    'random_db' : 'True',
    }

disabled = False

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
        num_images = num_train_imgs
    )

    #Main loop
    print("Started Training")
    train_it = 0
    test_it = 0 
    test_cap = 20

    images_used = np.zeros(num_train_imgs)

    while train_it < config['max_its']:

        print("Train it: ",train_it)

        if train_it >= (config['warm_start'] * train_data_gen.ret_batch_info()):
            i_scores,idx,i_des_descrep, n_i_scores = sf.sample_batches(model,train_ds,config["k"],config["batch_size"],num_classes,conn,config["des_inner"],config["des_outer"],images_used)

        for i, (X,Y) in enumerate(train_data_gen):
            #do k iterations on batches (first round is at least 1 full epoch run)
            #train function
            train_step(X[1],Y)
            train_it += 1
        
        #calc stats
        if train_it > (config['warm_start'] * train_data_gen.ret_batch_info()):
            wandb.log({ 'train_acc':train_acc_metric.result().numpy(),
                        'train_loss':train_loss.result().numpy(),
                        'mean_i_scores':np.mean(i_scores),
                        'std_i_scores':np.std(i_scores),
                        'chosen_i_score':i_scores[idx],
                        'inner_des_diff':i_des_descrep,
                        'all_i_scores_90':np.percentile(i_scores,90),
                        'all_i_scores_10':np.percentile(i_scores,10)},step=train_it)
        else:
            wandb.log({ 'train_acc':train_acc_metric.result().numpy(),
                        'train_loss':train_loss.result().numpy()},step=train_it)


        train_data_gen.on_epoch_end()

        test_it += 1
        if test_it > test_cap:
            for X,Y in test_ds.batch(100):
                Y = tf.one_hot(Y,num_classes)
                test_step(X,Y)
            test_it = 0

            wandb.log({'test_acc':test_acc_metric.result().numpy(),
                    'test_loss':test_loss.result().numpy()},step=train_it)
    #finished training
    pred = []
    t = []
    for X,Y in test_ds:
        pred.append(model.predict(X).argmax())
        t.append(Y)
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,y_true=t,preds=preds,class_names=class_names)})


    #final logging
    wandb.log({'images_used':wandb.Histogram(images_used)})
        
