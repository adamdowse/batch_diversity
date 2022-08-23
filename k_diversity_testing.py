#imports

@tf.function
def train_step(imgs,labels):
    with tf.GradientTape() as tape:
        preds = model(imgs,training=True)
        loss = tf.math.reduce_mean(batch_loss)
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


def main(max_iterations,k,batch_size):

    #setup
    '''amount of data''' = sf.setup_db()
    sf.setup_logs()
    '''things''' = sf.setup_model()
    #TODO make sure above outputs the right infomation

    train_data_gen = sf.CustomDBDataGen(
        conn = conn,
        X_col = 'img',
        Y_col = 'label',
        batch_size = batch_size, 
        num_classes = num_classes,
        num_images = num_images,
        input_size = img_shape
    )

    #Main loop
    print("Started Training")

    warm_i = 0
    it = 0
    while it < max_iterations:
        
        for i, (X,Y) in enumerate(train_data_gen):
            #do k iterations on batches (first round is at least 1 full epoch run)

            #train function
            train_step()

            it += 1
        
        train_data_gen.on_epoch_end()
        



        