import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import numpy as np
import matplotlib.pyplot as plt
import supporting_models as sm
import os
import tensorflow_datasets as tfds


#init vars
batch_size = 32
learning_rate = 0.001
momentum = 0
epochs = 20

model_name = 'EfficientNetV2B0_pretrained'
ds_name = 'cifar10'

os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
wandb.login()
wandb.init(project='Mixup_loss',entity='adamdowse')

#get dataset
train_ds, train_info = tfds.load(ds_name, split='train', shuffle_files=True,as_supervised=True,with_info=True)
test_ds = tfds.load(ds_name, split='test', shuffle_files=True,as_supervised=True)

#preprocess
data_shape = train_info.features['image'].shape
print(data_shape)
print(data_shape+(1,))
data_classes = train_info.features['label'].num_classes
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=data_classes)))
test_ds = test_ds.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=data_classes)))

#no augmentation

#get model
model = sm.select_model(model_name,data_classes,data_shape)
model.build(data_shape+(1,))
model.summary()

#compile model
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_func_nored = tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

#Setup logs and records
logging_callback = WandbCallback(log_freq=1,save_model=False,save_code=False)

#select images 3 of the same class
print("|--Selecting images for analysis")
images = []
labels = []

wanted_class = [tf.one_hot(6, depth=data_classes),tf.one_hot(7, depth=data_classes),tf.one_hot(8, depth=data_classes)]
train_ds_np = tfds.as_numpy(train_ds)
count = 0
for data in train_ds_np:  
    if np.array_equal(data[1],wanted_class[count]):
        count += 1
        images.append(data[0])
        labels.append(data[1])
        
        if count > 2:
            wandb.log({'OG_images': [wandb.Image(images[0],caption='OG_image_a'),wandb.Image(images[1],caption='OG_image_b'),wandb.Image(images[2],caption='OG_image_c')]})
            break 

print("|--Building Dataset of Mixed Images")
mixed_images = []
mixed_labels = []
c = 0
#create dataset of mixup images
for lam_1 in np.linspace(1,0,101):
    for lam_2 in np.linspace(0,1,101):
        if lam_1 + lam_2 <= 1:
            c += 1
            lam_3 = 1 - lam_1 - lam_2
            mixed_image = np.add(np.add(lam_1*images[0],lam_2*images[1]), lam_3*images[2]).astype(int)
            mixed_label = np.add(np.add(lam_1*labels[0],lam_2*labels[1]), lam_3*labels[2])
            #mixed_label = wanted_class

            mixed_images.append(mixed_image)
            mixed_labels.append(mixed_label)

            #save interpolated images
            #a to c
            if lam_1 in [0,0.25,0.5,0.75,1] and lam_2 == 0:
                wandb.log({'Interpolated_images_a_to_c': [wandb.Image(mixed_image,caption='a'+str(lam_1)+'_b'+str(lam_2)+'_c'+str(lam_3))]})
            #b to c
            if lam_1 == 0 and lam_2 in [0,0.25,0.5,0.75,1]:
                wandb.log({'Interpolated_images_b_to_c': [wandb.Image(mixed_image,caption='a'+str(lam_1)+'_b'+str(lam_2)+'_c'+str(lam_3))]})
            #a to b
            if lam_1 in [0,0.25,0.5,0.75,1] and lam_3 == 0:
                wandb.log({'Interpolated_images_a_to_b': [wandb.Image(mixed_image,caption='a'+str(lam_1)+'_b'+str(lam_2)+'_c'+str(lam_3))]})

print(c)
mixup_ds = tf.data.Dataset.from_tensor_slices((mixed_images,mixed_labels)).batch(32)

print("|--Training with full dataset and then evaluating on mixup dataset")
#model.fit(train_ds.batch(batch_size).shuffle(1000),epochs=4,validation_data=test_ds,callbacks=[logging_callback])
for e in range(epochs):
    #Train model
    model.fit(train_ds.batch(batch_size).shuffle(1000),epochs=1,validation_data=test_ds,callbacks=[logging_callback])

    loss_landscapes = np.zeros(c)

    #get loss landscape
    for i, data in enumerate(mixup_ds):
        preds = model(data[0])
        loss = loss_func_nored(data[1],preds)
        loss_landscapes[i*32:(i*32)+loss.shape[0]] = loss.numpy()


    final_loss_landscape = np.zeros((101,101))
    ind = np.tril_indices(101)
    final_loss_landscape[ind] = loss_landscapes

    #plot loss landscape
    wandb.log({'Loss_landscape': [wandb.Image(plt.imshow(final_loss_landscape),caption=str(e)+'Loss_Landscape')]})

    



