import supporting_functions as sf
import tensorflow as tf
import wandb

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
    'group' : 'testing',
    'model_name' : 'Simple_CNN',
    'learning_rate' : 0.01,
    'batch_size' : 32,
    'max_its' : 2000,
    'k' : 5,
    'alpha' : 1,
    'beta' : 1,
    'random_db' : 'False',
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
        num_images = num_train_imgs
    )

    #Main loop
    print("Started Training")
    train_it = 0
    test_it = 0 
    test_cap = 20
    while train_it < config['max_its']:
        for i, (X,Y) in enumerate(train_data_gen):
            #do k iterations on batches (first round is at least 1 full epoch run)
            #train function
            train_step(X[1],Y)

            train_it += 1
        
        train_data_gen.on_epoch_end()
        sf.k_diversity(model,train_ds,config['k'],config['batch_size'],config['alpha'],config['beta'],num_classes,conn)
        test_it += 1

        wandb.log({ 'train_acc':train_acc_metric.result().numpy(),
                    'train_loss':train_loss.result().numpy()},step=train_it)

        if test_it > test_cap:
            for X,Y in test_ds:
                test_step(X[1],Y)
            test_it = 0

            wandb.log({'test_acc':test_acc_metric.result().numpy(),
                    'test_loss':test_loss.result().numpy()},step=train_it)
        
