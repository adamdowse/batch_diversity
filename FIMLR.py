import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import FIM_analysis as fim
import Pretrained_supporting_functions as sf
import DataGens
import supporting_models as sm
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback
import time
import tracemalloc
import os


#This experiment is to measure the corrolation between the FIM and learning rate and final test accuracy.

def main():
    @tf.function
    def train_step(imgs,labels):
        with tf.GradientTape() as tape:
            preds = model(imgs,training=True)[0]
            loss = loss_func(labels,preds)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        train_loss(loss)
        train_acc_metric(labels,preds)
        train_prec_metric(labels,preds)
        train_rec_metric(labels,preds)
        return
    
    @tf.function
    def test_step(imgs,labels):
        with tf.GradientTape() as tape:
            preds = model(imgs,training=False)[0]
            loss = loss_func(labels,preds)
        test_loss(loss)
        test_acc_metric(labels,preds)
        test_prec_metric(labels,preds)
        test_rec_metric(labels,preds)

    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/
    #/com.docker.devenvironments.code/datasets/
    #/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/
    config= {
        'ds_path' : "/com.docker.devenvironments.code/datasets/",
        'db_path' : "/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/DBs/",
        'ds_name' : "mnist",
        'group' : 'FIMLR_Mk1',
        'train_percent' : 0.1,
        'test_percent' : 0.1,
        'model_name' : 'FullyConnected',
        'learning_rate' : 0.01,
        'learning_rate_decay' : 0,
        'optimizer' : 'SGD', #SGD, Adam, Momentum
        'momentum' : 0,
        'random_db' : 'True', #False is wrong it adds the datasets together
        'batch_size' : 128,
        'label_smoothing' : 0,
        'weight_decay' : 0,
        'data_aug' : '0', #0 = no data aug, 1 = data aug, 2 = data aug + noise
        'start_defect_epoch' : 1000,
        'defect_length' : 1000, # length of defect in epochs
        'max_its' : 46900,
        'epochs'    : 30, #if this != 0 then it will override max_its    
        'early_stop' : 50,
        'subset_type' : 'All', #Random_Bucket, Hard_Mining, All
        'train_type' : 'Normal', #SubMod, Random
        'activation_delay' : 1, #cannot be 0 (used when submod is used)
        'activation_layer_name' : 'fc',
        'alpha' : 0.5, #0 is max seperation 1 is max alignment to mean
    }

    #Setup
    wandb.init(project='FIM_Div_Review',config=config)
    calc_stats = False

    #Data Generator
    train_DG = DataGens.LocalSUBMODGRADDataGenV2(config['ds_name'],config['batch_size'],config['train_percent'],config['ds_path'],config['alpha'],calc_stats=calc_stats)
    test_DG = DataGens.TestDataGen(config['ds_name'], 50, config['test_percent'], config['ds_path'])
    second_train_DG = DataGens.LocalSUBMODGRADDataGenV2(config['ds_name'],config['batch_size'],config['train_percent'],config['ds_path'],config['alpha'],calc_stats=calc_stats)

    #Model
    tf.keras.backend.clear_session()
    model = sm.select_model(config['model_name'],train_DG.num_classes,train_DG.img_size,config['weight_decay'],getLLactivations=True)
    model.build(train_DG.img_size+(1,))
    model.summary()

    #Loss
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False,label_smoothing=config['label_smoothing'])
    
    #Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_prec_metric = tf.keras.metrics.Precision(name='train_precision')
    train_rec_metric = tf.keras.metrics.Recall(name='train_recall')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_prec_metric = tf.keras.metrics.Precision(name='test_precision')
    test_rec_metric = tf.keras.metrics.Recall(name='test_recall')

    #Optimizer
    class lr_schedule():
        def __init__(self,k,init_lr=config['learning_rate'],method=0):
            self.itt_num = 0
            self.FIM_store = []
            self.k = k
            self.LR_store = []
            self.init_lr = init_lr
            self.lr = init_lr
            self.exp_mult = 0.01
            self.method = method
            

        def __call__(self):
            if len(self.FIM_store) <= self.k:
                self.LR_store.append(self.lr)
                return self.lr
            else:
                if self.method == 0:
                    #calculate the average change of the FIM on the latest k items
                    print(self.FIM_store)
                    print(self.LR_store)
                    #fit = np.polyfit(self.LR_store[-self.k:],self.FIM_store[-self.k:],1) #can add a weighting here if needed
                    fit = np.polyfit(np.arange(self.k),self.FIM_store[-self.k:],1)
                    g = fit[0]
                    print('g: ',g)
                    #update the learning rate
                    self.lr = self.lr * np.exp(self.exp_mult*g)
                    #store the learning rate
                    self.LR_store.append(self.lr)
                    return self.lr
                elif self.method == 1:
                    #method to change lr based on the convergence of FIM
                    #if FIM has been decreasing for the last k iterations then decrease the learning rate

                    fit = np.polyfit(np.arange(self.k),self.FIM_store[-self.k:],1)
                    g = fit[0]
                    if g < 0:
                        self.lr = self.lr * np.exp(self.exp_mult*g)
                    self.LR_store.append(self.lr)
                    return self.lr
        
        def update(self,FIM):
            if FIM == None:
                FIM = 0
            self.FIM_store.append(FIM)
            return

                
    FIM_lr_schedule = lr_schedule(4,config['learning_rate'])

    if config['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    elif config['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'Momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'],momentum=config['momentum'])
    else:
        print('Optimizer not recognised')     

    #optimizer.learning_rate

    #Compile Model
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #Training
    batch_num = 0
    early_stop_max = 0
    early_stop_count = 0
    epoch_num = 0
    batch_FIM_store = []
    while epoch_num < config['epochs']:
        epoch_num += 1
        print('Epoch #: ',epoch_num,' Batch #: ',batch_num)

        #reset metrics
        train_loss.reset_states()
        train_acc_metric.reset_states()
        train_prec_metric.reset_states()
        train_rec_metric.reset_states()
        test_loss.reset_states()
        test_acc_metric.reset_states()
        test_prec_metric.reset_states()
        test_rec_metric.reset_states()

        #Scores the training data and decides what to train on
        if epoch_num > config['start_defect_epoch'] and epoch_num < config['start_defect_epoch']+config['defect_length']:
            #this initilises the data generator with the defect data 
            train_DG.Epoch_init(False)
        else:
            #this initilises the data generator with the normal data
            train_DG.Epoch_init(True)

        #wandb.log({'Train_loss_hist':wandb.Histogram(train_DG.losses)},step=batch_num)
        
        #Train on the data subset
        print('Training')
        for i in range(train_DG.num_batches):
            t1 = time.time()
            batch_data = train_DG.__getitem__(i)
            t = time.time()
            train_step(batch_data[0],batch_data[1])
            print('Get data: ',t-t1,'train step time: ',time.time() - t)
            batch_num += 1

            #Test on the test data
            print('Evaluating')
            for i in range(test_DG.num_batches):
                batch_data = test_DG.__getitem__(i)
                test_step(batch_data[0],batch_data[1])

            #Log metrics
            wandb.log({'Train_loss':train_loss.result(),'Train_acc':train_acc_metric.result(),'Train_prec':train_prec_metric.result(),'Train_rec':train_rec_metric.result()},step=batch_num)
            wandb.log({'Test_loss':test_loss.result(),'Test_acc':test_acc_metric.result(),'Test_prec':test_prec_metric.result(),'Test_rec':test_rec_metric.result()},step=batch_num)
            wandb.log({'Learning_rate':optimizer.learning_rate},step=batch_num)
            wandb.log({'Epoch':epoch_num},step=batch_num)
        
            #Grad Analysis
            second_train_DG.Epoch_init(True)
            mean_grad_activity, mean_grad_var = fim.Grad_Div_Ensemble_Method(second_train_DG,model)
            wandb.log({"mean_grad_activity":mean_grad_activity,"mean_grad_var":mean_grad_var},step=batch_num)

            #FIM Analysis
            second_train_DG.Epoch_init(True)
            FIM_trace, FIM_trace_var = fim.FIM_trace_with_var(second_train_DG,second_train_DG.num_classes,model) #return the approximate trace of the FIM
            Test_FIM_trace, Test_FIM_trace_var = fim.FIM_trace_with_var(test_DG,second_train_DG.num_classes,model)
            wandb.log({'Approx_Trace_FIM': FIM_trace,"Test_Trace_FIM":Test_FIM_trace,'FIM_var': FIM_trace_var,"Test_FIM_var":Test_FIM_trace_var},step=batch_num)

            optimizer.learning_rate = FIM_lr_schedule()
            print('Learning rate: ',optimizer.learning_rate)
            FIM_lr_schedule.update(FIM_trace)


        #Early stopping
        # if test_acc_metric.result() > early_stop_max:
        #    early_stop_max = test_acc_metric.result()
        #    early_stop_count = 0
        # else:
        #    early_stop_count += 1
        #    if early_stop_count > config['early_stop']:
        #        break


    #wandb.log({'Images_used_hist':wandb.Histogram(train_DG.data_used)})

    #Finish - clear keras backend
    tf.keras.backend.clear_session()
    print('Finished')


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    wandb.login()
    main()