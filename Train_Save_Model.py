import supporting_models as sm
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os

model_name = 'EfficientNetV2B0_pretrained'
ds_name = 'cifar10'
disabled = False

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'val_accuracy'
		},
    'parameters': {
        'batch_size': {'values': [32,64,128,256]},
        'lr': {'values': [0.1,0.01,0.001,0.0001]},
        'momentum': {'values': [0,0.5,0.99]},
     }
}

os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
if disabled:
    os.environ['WANDB_DISABLED'] = 'true'
wandb.login()


def main():
    wandb.init(project='k_diversity',entity='adamdowse')
    #get dataset
    train_ds, train_info = tfds.load(ds_name, split='train', shuffle_files=True,as_supervised=True,with_info=True)
    test_ds = tfds.load(ds_name, split='test', shuffle_files=True,as_supervised=True)

    #preprocess
    data_shape = train_info.features['image'].shape
    data_classes = train_info.features['label'].num_classes
    train_ds = train_ds.batch(wandb.config.batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=data_classes)))
    test_ds = test_ds.batch(wandb.config.batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=data_classes)))

    #get model
    model = sm.select_model(model_name,data_classes,data_shape)
    model.build(data_shape+(1,))
    model.summary()

    #compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.lr, momentum=wandb.config.momentum)#
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    #Setup logs and records
    logging_callback = WandbCallback(log_freq=1,save_model=True,save_code=False)

    #Train model
    model.fit(train_ds,epochs=20,validation_data=test_ds,callbacks=[logging_callback])

sweep_id = wandb.sweep(sweep=sweep_configuration, project='k_diversity')
wandb.agent(sweep_id, function=main, count=10)
#main()


