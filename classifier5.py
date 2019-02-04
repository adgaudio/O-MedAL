from keras.layers import Dense,Activation,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D,Conv2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg16
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_v3


class Classifier:
    """Class that wraps a CNN classifier."""

    def __init__(self, architecture='inception', n_features=1024, weights='imagenet', input_tensor=None):
        self.architecture = architecture
        self.n_features = n_features

        m, preprocessing_fn = self._get_architecture(architecture)
        self.base_model = m(weights=None, include_top=False, input_tensor=input_tensor)

        x = GlobalAveragePooling2D()(self.base_model.output)
        x = Dense(512,kernel_regularizer=l2(0.000001))(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128,kernel_regularizer=l2(0.000001))(x)
        x = Activation('relu')(x)    
        x = Dense(1)(x)
        out = Activation('sigmoid')(x)

        self.model = Model(inputs=self.base_model.input, outputs=out)

    def prepare_to_finetune(self, init_lr):
        """Set transfered layers untrainable and compile model."""
        for layer in self.model.layers:
            layer.trainable = True

        opt = Adam(lr=init_lr)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

        return self.model

    def prepare_to_init(self, fine_lr):
        """Set transfered layers trainable and compile model."""
        for layer in self.model.layers[:132]:
            layer.trainable = False

        for layer in self.model.layers[132:]:
            layer.trainable = True

        opt = Adam(lr=fine_lr)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

        return self.model

    def _get_architecture(self, architecture):
        arch_low = architecture.lower()
        if arch_low == 'inception':
            return InceptionV3, preprocess_input_v3
        if arch_low == 'vgg16':
            return VGG16, preprocess_input_vgg16
        if arch_low == 'vgg19':
            return VGG19, preprocess_input_vgg19
        if arch_low == 'resnet':
            return ResNet50, preprocess_input_resnet

        raise ValueError("Unknown architecture. Known architectures are 'inception', 'vgg16', 'vgg19' and 'resnet' but got {0}.".format(architecture))
