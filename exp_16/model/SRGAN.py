import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg19 import VGG19

class Generator():
    def __init__(self):
        pass

    def gene_base_block(self, x):
        '''
        skip conection을 가지고 있는
        간단한 Convoultion block
        Conv2D(#ofC, f, s, p)
        '''
        self.out = layers.Conv2D(64, 3, 1, "same")(x)
        self.out = layers.BatchNormalization()(self.out)
        self.out = layers.PReLU(shared_axes=[1, 2])(self.out)
        self.out = layers.Conv2D(64, 3, 1, 'same')(self.out)
        self.out = layers.BatchNormalization()(self.out)
        return layers.Add()([x, self.out])

    def upsample_block(self, x):
        self.out = layers.Conv2D(256, 3, 1, "same")(x)
        self.out = layers.Lambda(lambda x:tf.nn.depth_to_space(x,2))(self.out)
        return layers.PReLU(shared_axes=[1,2])(self.out)

    def get_generator(self, input_shape=(None, None, 3)):
        self.inputs =Input(input_shape)

        self.out = layers.Conv2D(64, 9 ,1, "same")(self.inputs)
        self.out = self.residual = layers.PReLU(shared_axes=[1,2])(self.out)
        for _ in range(5):
            self.out = self.gene_base_block(self.out)

        self.out = layers.Conv2D(64, 3, 1, 'same')(self.out)
        self.out = layers.BatchNormalization()(self.out)
        self.out = layers.Add()([self.residual, self.out])

        for _ in range(2):
            self.out = self.upsample_block(self.out)

        self.out = layers.Conv2D(3, 9, 1, 'same', activation='tanh')(self.out)
        return Model(self.inputs, self.out)

class Discriminator():
    def __init__(self):
        pass

    def disc_base_block(self, x, n_filters=128):
        self.out = layers.Conv2D(n_filters, 3, 1, "same")(x)
        self.out = layers.BatchNormalization()(self.out)
        self.out = layers.LeakyReLU()(self.out)
        self.out = layers.Conv2D(n_filters, 3, 2, 'same')(self.out)
        self.out = layers.BatchNormalization()(self.out)
        return layers.LeakyReLU()(self.out)

    def get_discriminator(self, input_shape=(None, None, 3)):
        self.inputs = Input(input_shape)

        self.out = layers.Conv2D(64, 3, 1,  'same')(self.inputs)
        self.out = layers.LeakyReLU()(self.out)
        self.out = layers.Conv2D(64, 3, 2, 'same')(self.out)
        self.out = layers.BatchNormalization()(self.out)
        self.out = layers.LeakyReLU()(self.out)

        for n_filters in [128,256,512]:
            self. out = self.disc_base_block(self.out, n_filters)

        self.out = layers.Dense(1024)(self.out)
        self.out = layers.LeakyReLU()(self.out)
        self.out = layers.Dense(1, activation="sigmoid")(self.out)
        return Model(self.inputs, self.out)