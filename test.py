import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import supporting_models as sm

#Load the dataset
ds_name = 'cifar10'
ds, ds_info = tfds.load(ds_name, split='train', shuffle_files=False, with_info=True,as_supervised=True)

img_size = ds_info.features['image'].shape
num_classes = ds_info.features['label'].num_classes

ds = ds.batch(32)
ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))


#Create the model
model = sm.select_model('Simple_CNN',num_classes,img_size)
model.build(img_size+(1,))
loss_func_no = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
loss_func = keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(optimizer='SGD',loss=loss_func)

model.fit(ds,epochs=1)

@tf.function
def loss_step(x,y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_func_no(y,pred)
    return loss

#model.compile(optimizer='SGD',loss=loss_func_no)

losses = np.array([])
for x,y in ds:
    losses = np.append(losses,loss_step(x,y).numpy())

print(losses.shape)


