from scipy.misc import imread, imresize
import numpy as np

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Merge
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.layers.wrappers import TimeDistributed

from keras.layers.core import Layer
import keras.backend as K

from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D, splittensor_r


class MaxLayer(Layer):
    def __init__(self, **kwargs):
        super(MaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = (input_shape[0], input_shape[2])
        self.input_dim = input_shape

    def call(self, x, mask=None):
        return K.max(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return self.output_dim
    
class Average(Layer):
    def __init__(self, **kwargs):
        super(Average, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = (input_shape[0], input_shape[2])
        self.input_dim = input_shape

    def call(self, x, mask=None):
        return K.mean(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return self.output_dim

def AlexNet(weights_path=None, inputs=None, id='1', trainable=True, finetune=False, last_layer=None):
    if inputs == None:
        inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name=id+'conv_1',trainable=trainable)(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name=id+"convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name=id+'conv_2_'+str(i+1),trainable=trainable)(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name=id+"conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name=id+'conv_3',trainable=trainable)(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name=id+'conv_4_'+str(i+1),trainable=trainable)(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name=id+"conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name=id+'conv_5_'+str(i+1),trainable=trainable)(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name=id+"conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name=id+"convpool_5",trainable=trainable)(conv_5)
    
    dense_1 = Flatten(name=id+"flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name=id+'dense_1',trainable=trainable)(dense_1)
    dense_2 = Dense(4096,name=id+'dense_4',trainable=True)(dense_1)
    dense_2_act = Activation('relu')(dense_2)


    dense_2_alex = Dense(4096 ,name=id+'dense_4',trainable=True,W_regularizer=l2(0))(dense_1)
    dense_2_alex_act = Activation('relu',name=id+'dense_4_embedding')(dense_2_alex)
    dense_3_alex = Dense(1000,name=id+'dense_3_100',activation='sigmoid',trainable=trainable)(dense_2_alex_act)


    if last_layer == None:
        last_layer = Dense(8,name=id+'dense_3_100',activation='softmax',trainable=True,W_regularizer=l2(0))
        dense_3 = last_layer(dense_2_alex_act)
    else:
        dense_3 = last_layer(dense_2_alex_act)

    if finetune:
        model_classification = Model(input=inputs, output=dense_3)
        output = dense_2
    else:
        model_classification = Model(input=inputs, output=dense_3_alex)
        output = dense_2_alex

    if weights_path:
        model_classification.load_weights(weights_path)#, by_name=True)
        print "model loaded"
        #model.save_weights('../../data/weights/alexnet_weights_finetune.h5')
        
    return dense_3, dense_2_alex_act, last_layer

def MAlexNet(weights_path=None, inputs=None, id='', trainable=True):
    conv_1 = TimeDistributed(Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name=id+'conv_1',trainable=trainable))(inputs)

    conv_2 = TimeDistributed(MaxPooling2D((3, 3), strides=(2,2)))(conv_1)
    conv_2 = TimeDistributed(crosschannelnormalization(name=id+"convpool_1"))(conv_2)
    conv_2 = TimeDistributed(ZeroPadding2D((2,2)))(conv_2)
    conv_2 = merge([
        TimeDistributed(Convolution2D(128,5,5,activation="relu",name=id+'conv_2_'+str(i+1),trainable=trainable))(
            splittensor_r(axis=2,ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=2,name=id+"conv_2")

    conv_3 = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(conv_2)
    conv_3 = TimeDistributed(crosschannelnormalization())(conv_3)
    conv_3 = TimeDistributed(ZeroPadding2D((1,1)))(conv_3)
    conv_3 = TimeDistributed(Convolution2D(384,3,3,activation='relu',name=id+'conv_3',trainable=trainable))(conv_3)

    conv_4 = TimeDistributed(ZeroPadding2D((1,1)))(conv_3)
    conv_4 = merge([
        TimeDistributed(Convolution2D(192,3,3,activation="relu",name=id+'conv_4_'+str(i+1),trainable=trainable))(
            splittensor_r(axis=2,ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=2,name=id+"conv_4")

    conv_5 = TimeDistributed(ZeroPadding2D((1,1)))(conv_4)
    conv_5 = merge([
        TimeDistributed(Convolution2D(128,3,3,activation="relu",name=id+'conv_5_'+str(i+1),trainable=trainable))(
            splittensor_r(axis=2,ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=2,name=id+"conv_5")

    dense_1 = TimeDistributed(MaxPooling2D((3, 3), strides=(2,2),name=id+"convpool_5",trainable=trainable))(conv_5)

    dense_1 = TimeDistributed(Flatten(name=id+"flatten"))(dense_1)
    dense_1 = TimeDistributed(Dense(4096, activation='relu',name=id+'dense_1',trainable=trainable))(dense_1)

    dense_2 = TimeDistributed(Dense(4096, name=id+'dense_4',trainable=True))(dense_1)
    dense_2_act = Activation('relu')(dense_2)

    dense_2_alex = TimeDistributed(Dense(4096 , activation='relu',trainable=True))(dense_1)
    dense_3_alex = TimeDistributed(Dense(1000,name=id+'dense_3_100',activation='sigmoid',trainable=trainable))(dense_2_alex)

    #dense_2_avg = Average(name=id+'dense_4_embedding')(dense_2_alex)
    dense_2_avg = MaxLayer(name=id+'dense_4_embedding')(dense_2_alex)

    dense_3 = Dense(8, activation='softmax', name=id+'dense_3_100',trainable=True)(dense_2_avg)
    model_classification = Model(input=inputs, output=dense_3_alex)

    if weights_path:
        model_classification.load_weights(weights_path)#, by_name=True)
        print "model loaded"
        #model.save_weights('../../data/weights/alexnet_weights_finetune.h5')

    return dense_3, dense_2_avg

def L2Distance(inputs):
    '''
    inputs: [inputs_1, inputs_2]
    return:
        L2 Distance between inputs_1 and inputs_2
    '''
    return K.mean(K.square(inputs[0]-inputs[1]), axis=1)


def L2Distance_001(inputs):
    '''
    inputs: [inputs_1, inputs_2]
    return:
        L2 Distance between inputs_1 and inputs_2
    '''
    return K.mean(K.square(inputs[0]-0.001*inputs[1]), axis=1)

def cos_distance(inputs):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true, y_pred = inputs[0], inputs[1]
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)

def BiAlex(weights_path_1='../../data/alexnet_weights.h5', weights_path_2='../../data/alexnet_weights.h5', finetune=True):
    inputs_1 = Input(shape=(3,227,227), name='inputs_1')
    inputs_2 = Input(shape=(3,227,227), name='inputs_2')
    supervision_1, embedding_1, last_layer = AlexNet(weights_path=weights_path_1, inputs=inputs_1, id='1', trainable=False, finetune=finetune, last_layer=None)
    supervision_2, embedding_2, last_layer2 = AlexNet(weights_path=weights_path_2, inputs=inputs_2, id='2', trainable=False, finetune=finetune, last_layer=None)
    distance = Merge(mode=L2Distance, output_shape=[1], name='distanceMerge')([embedding_1, embedding_2])
    #distance = Activation('sigmoid')(distance)

    model = Model(input=[inputs_1, inputs_2], output=[distance, supervision_1, supervision_2])
    model_img = Model(input=inputs_1, output=supervision_1)
    model_gel = Model(input=inputs_2, output=supervision_2)
    return model, model_img, model_gel


def MBiAlex(weights_path_1='../../data/alexnet_weights.h5', weights_path_2='../../data/alexnet_weights.h5',finetune=False):
    inputs_1 = Input(shape=(3,227,227),name='inputs_1')
    inputs_2 = Input(shape=(3,3,227,227),name='inputs_2')
    supervision_1, embedding_1, last_layer = AlexNet(weights_path=weights_path_1, inputs=inputs_1, id='1', trainable=False, finetune=finetune)
    supervision_2, embedding_2 = MAlexNet(weights_path=weights_path_2, inputs=inputs_2, id='2', trainable=False)
    distance = Merge(mode=L2Distance, output_shape=[1], name='distanceMerge')([embedding_1, embedding_2])
    #distance = Activation('sigmoid')(distance)

    model = Model(input=[inputs_1, inputs_2], output=[distance, supervision_1, supervision_2])
    model_img = Model(input=inputs_1, output=supervision_1)
    model_gel = Model(input=inputs_2, output=supervision_2)
    return model, model_img, model_gel


if __name__ == "__main__":
    img = imresize(imread('laska.png', mode='RGB'), (227, 227)).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img = img.transpose((2, 0, 1))
    img = np.array([img]*64)

    # Test pretrained model
    model = BiAlex('../../data/weights/alexnet_weights.h5')
    #model = AlexNet('../../data/weights/alexnet_weights.h5')
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    import time
    st = time.time()
    out = model.predict([img,img-1])
    print time.time()-st
    print out
    st = time.time()
    out = model.predict([img,img+1])
    print time.time()-st
    print out

    print model.layers

    import numpy as np
    x = np.zeros((64,3,227,227))
    y = np.zeros((64,1))
    print 'begin'
    model.fit([img,img+1],y,batch_size=128,verbose=1)

