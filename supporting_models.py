import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers
import math



def Simple_CNN(num_classes,in_shape,REG):
    model = tf.keras.Sequential([
        layers.Conv2D(32,(3,3), activation='relu',input_shape=in_shape,kernel_regularizer=tf.keras.regularizers.l2(REG)),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(100,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(REG),name="penultimate_layer"),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model

def Simple_CNN_Multi_Output(num_classes,in_shape,REG): 
    inp = keras.Input(in_shape)
    x = layers.Conv2D(192,(3,3),activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(96,(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(96,(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(32,(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    
    x = layers.Flatten()(x)
    a = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(num_classes)(a)
    x = layers.Softmax()(x)
    return keras.Model(inp,[x,a])




def All_CNN_noBN(num_classes,in_shape):
    #cnn arch used in crittical learnning point paper
    inp = keras.Input(in_shape)
    x = layers.Conv2D(96,(3,3), activation='relu')(inp)
    x = layers.Conv2D(96,(3,3), activation='relu')(x)
    x = layers.MaxPool2D((3,3),strides=2)(x)
    x = layers.Conv2D(192,(3,3), activation='relu')(x)
    x = layers.Conv2D(192,(3,3), activation='relu')(x)
    x = layers.MaxPool2D((3,3),strides=2)(x)
    x = layers.Conv2D(192,(1,1), activation='relu')(x)
    x = layers.Conv2D(10,(1,1), activation='relu')(x)
    a = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes,name="last_layer")(a)
    x = layers.Softmax()(x)
    return keras.Model(inp,[x,a])

def FullyConnected(num_classes,in_shape):
    inp = keras.Input(in_shape)
    x = layers.Flatten()(inp)
    x = layers.Dense(1000,activation='relu')(x)
    x = layers.Dense(500,activation='relu')(x)
    x = layers.Dense(250,activation='relu')(x)
    a = layers.Dense(30,activation='relu')(x)
    x = layers.Dense(num_classes,name="last_layer")(a)
    x = layers.Softmax()(x)
    return keras.Model(inp,[x,a])


def AlexNet (num_classes,in_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(96, 11, strides=4, activation='relu',input_shape=in_shape),
        layers.BatchNormalization(),

        layers.MaxPool2D(2, strides=2),
        
        layers.Conv2D(256,11,strides=1,activation='relu',padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
        layers.BatchNormalization(),
    
        layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
        layers.BatchNormalization(),

        layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
        layers.BatchNormalization(),

        layers.MaxPooling2D(2, strides=(2, 2)),

        layers.Flatten(),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(4096, activation='relu',name="penultimate_layer"),
        layers.Dropout(0.5),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model

def EfficientNetV2B0_pretrained(num_classes,in_shape):
    model = tf.keras.Sequential([
        tf.keras.applications.EfficientNetV2B0(include_top=False,weights='imagenet',input_shape=in_shape,classifier_activation='None'),
        layers.Flatten(),
        layers.Dense(100,activation='relu',name="penultimate_layer"),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model

def ResNet50(num_classes,in_shape):
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=in_shape,
        pooling=None,
        classes=num_classes)
    return model

def ResNet101(num_classes,in_shape):
    model =  tf.keras.applications.resnet.ResNet101(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=in_shape,
        pooling=None,
        classes=num_classes)
    return model

def ResNet18(num_classes,in_shape,REG):
    inputs = keras.Input(shape=in_shape)
    outputs = build_resnet(inputs, [2, 2, 2, 2], num_classes,REG)
    model = keras.Model(inputs, outputs)
    return model
    


def build_resnet(x,vars,num_classes,REG=0):
    kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

    def conv3x3(x, out_planes, stride=1, name=None):
        x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
        return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=name)(x)

    def basic_block(x, planes, stride=1, downsample=None, name=None):
        identity = x

        out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
        out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
        out = layers.ReLU(name=f'{name}.relu1')(out)

        out = conv3x3(out, planes, name=f'{name}.conv2')
        out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

        if downsample is not None:
            for layer in downsample:
                identity = layer(identity)

        out = layers.Add(name=f'{name}.add')([identity, out])
        out = layers.ReLU(name=f'{name}.relu2')(out)

        return out

    def make_layer(x, planes, blocks, stride=1, name=None):
        downsample = None
        inplanes = x.shape[3]
        if stride != 1 or inplanes != planes:
            downsample = [
                layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=f'{name}.0.downsample.0'),
                layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
            ]

        x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
        for i in range(1, blocks):
            x = basic_block(x, planes, name=f'{name}.{i}')

        return x

    def resnet(x, blocks_per_layer, num_classes):
        x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
        x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name='conv1')(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
        x = layers.ReLU(name='relu1')(x)
        x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
        x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

        x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
        x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
        x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
        x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

        

        x = layers.GlobalAveragePooling2D(name='avgpool')(x)
        a = layers.Flatten()(x)
        initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
        x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
        x = layers.Softmax(name='softmax')(x)
        return x,a
    return resnet(x, vars,num_classes)

def CIFAR10_ViT(num_classes,in_shape):
    image_size = 72
    patch_size = 6
    num_patches = (image_size//patch_size)**2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim,]
    transformer_layers = 8
    mlp_head_units = [2048,1024]

    return ViT(num_classes,in_shape,image_size,patch_size,num_patches,projection_dim,num_heads,transformer_units,transformer_layers,mlp_head_units)

def ViT(num_classes,in_shape,image_size,patch_size,num_patches,projection_dim,num_heads,transformer_units,transformer_layers,mlp_head_units):

    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size,image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation"
    )
    #data_augmentaion.layers[0].adapt(x_train) #THIS NEEDS FIXING

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size
        
        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images = images,
                sizes  = [1,self.patch_size, self.patch_size, 1],
                strides= [1,self.patch_size, self.patch_size, 1],
                rates  = [1,1,1,1],
                padding= "VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

    class PatchEncoder(layers.Layer):
        def __init__(self,num_patches,projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded
    
    inputs = layers.Input(shape=in_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches,projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1,x1)

        x2 = layers.Add()([attention_output,encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3,x2])
    
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    sm = layers.Softmax()(logits)
    return keras.Model(inputs=inputs,outputs=sm)




def select_model(model_name,num_classes,img_shape,REG=0,getLLactivations=False):
    if model_name == 'CIFAR10_ViT':
        return CIFAR10_ViT(num_classes,img_shape)
    if model_name == 'Simple_CNN':
        return Simple_CNN(num_classes,img_shape,REG)
    if model_name == 'Simple_CNN_Multi_Output':
        return Simple_CNN_Multi_Output(num_classes,img_shape,REG)
    if model_name == 'AlexNet':
        return AlexNet(num_classes,img_shape)
    if model_name == 'EfficientNetV2B0_pretrained':
        return EfficientNetV2B0_pretrained(num_classes,img_shape)
    if model_name == 'ResNet50':
        return ResNet50(num_classes,img_shape)
    if model_name == 'ResNet101':
        return ResNet101(num_classes,img_shape)
    if model_name == 'ResNet18':
        return ResNet18(num_classes,img_shape,REG)
    if model_name == 'All_CNN_noBN':
        return All_CNN_noBN(num_classes,img_shape)
    if model_name == 'FullyConnected':
        return FullyConnected(num_classes,img_shape)