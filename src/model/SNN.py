from scipy.misc import imread, imresize
import numpy as np

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Merge
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD

from keras.layers.core import Layer
import keras.backend as K

from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D

def AlexNet(weights_path=None, inputs1=None, inputs2=None, id='', trainable=True):

    conv_1_op = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name=id+'conv_1',trainable=trainable)



    conv_1 = conv_1_op(inputs1)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name=id+"convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)

    conv_2_op_a = Convolution2D(128,5,5,activation="relu",name=id+'conv_2_1',trainable=trainable)
    conv_2_op_b = Convolution2D(128,5,5,activation="relu",name=id+'conv_2_2',trainable=trainable)
    conv_2 = merge([
            conv_2_op_a(splittensor(ratio_split=2,id_split=0)(conv_2)),
            conv_2_op_b(splittensor(ratio_split=2,id_split=1)(conv_2))],
            mode='concat',concat_axis=1,name=id+"conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)

    conv_3_op = Convolution2D(384,3,3,activation='relu',name=id+'conv_3',trainable=trainable)
    conv_3 = conv_3_op(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)

    conv_4_op_a = Convolution2D(192,3,3,activation="relu",name=id+'conv_4_1',trainable=trainable)
    conv_4_op_b = Convolution2D(192,3,3,activation="relu",name=id+'conv_4_2',trainable=trainable)

    conv_4 = merge([
        conv_4_op_a(splittensor(ratio_split=2,id_split=0)(conv_4)),
        conv_4_op_b(splittensor(ratio_split=2,id_split=1)(conv_4))],
        mode='concat',concat_axis=1,name=id+"conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)

    conv_5_op_a = Convolution2D(128,3,3,activation="relu",name=id+'conv_5_1',trainable=trainable)
    conv_5_op_b = Convolution2D(128,3,3,activation="relu",name=id+'conv_5_2',trainable=trainable)
    conv_5 = merge([
        conv_5_op_a(splittensor(ratio_split=2,id_split=0)(conv_5)),
        conv_5_op_b(splittensor(ratio_split=2,id_split=1)(conv_5))],
        mode='concat',concat_axis=1,name=id+"conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name=id+"convpool_5",trainable=trainable)(conv_5)
    dense_1 = Flatten(name=id+"flatten")(dense_1)

    dense_1_op = Dense(4096, activation='relu',name=id+'dense_1',trainable=trainable)
    dense_1_a = dense_1_op(dense_1)

    dense_2_op = Dense(4096 , activation='relu', name=id+'dense_2_gel',trainable=True)
    dense_2 = dense_2_op(dense_1_a)

    dense_3_op = Dense(1000,name=id+'dense_3_100',activation='sigmoid',trainable=trainable)
    dense_3 = dense_3_op(dense_2)


    model = Model(input=inputs1, output=dense_3)

    if weights_path:
        model.load_weights(weights_path, by_name=True)
        print "model loaded"
        #model.save_weights('../../data/weights/alexnet_weights_finetune.h5')


    super_op = Dense(8, name=id+'supervision', activation='softmax')
    dense_4_1 = dense_2
    supervision_1 = super_op(dense_4_1)



    # FOR INPUTS2
    id = '2'

    conv_1 = conv_1_op(inputs2)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name=id+"convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)

    conv_2 = merge([
            conv_2_op_a(splittensor(ratio_split=2,id_split=0)(conv_2)),
            conv_2_op_b(splittensor(ratio_split=2,id_split=1)(conv_2))],
            mode='concat',concat_axis=1,name=id+"conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)

    conv_3 = conv_3_op(conv_3)
    conv_4 = ZeroPadding2D((1,1))(conv_3)

    conv_4 = merge([
        conv_4_op_a(splittensor(ratio_split=2,id_split=0)(conv_4)),
        conv_4_op_b(splittensor(ratio_split=2,id_split=1)(conv_4))],
        mode='concat',concat_axis=1,name=id+"conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        conv_5_op_a(splittensor(ratio_split=2,id_split=0)(conv_5)),
        conv_5_op_b(splittensor(ratio_split=2,id_split=1)(conv_5))],
        mode='concat',concat_axis=1,name=id+"conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name=id+"convpool_5")(conv_5)
    dense_1 = Flatten(name=id+"flatten")(dense_1)
    dense_1_1 = dense_1_op(dense_1)
    dense_2 = dense_2_op(dense_1_1)

    
    dense_4_2 = dense_2
    super_op = Dense(8, name=id+'supervision', activation='softmax')
    supervision_2 = super_op(dense_4_2)

    single_model = Model(input=inputs1, output=dense_4_1)
    return dense_4_1, dense_4_2, supervision_1, supervision_2


def L2Distance(inputs):
    '''
    inputs: [inputs_1, inputs_2]
    return:
        L2 Distance between inputs_1 and inputs_2
    '''
    return K.mean(K.square(inputs[0]-inputs[1]), axis=1)

def cos_distance(inputs):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true, y_pred = inputs[0], inputs[1]
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)

def SNN(weights_path='../../data/alexnet_weights.h5'):
    inputs_1 = Input(shape=(3,227,227),name='inputs_1')
    inputs_2 = Input(shape=(3,227,227),name='inputs_2')
    embedding_1, embedding_2, supervision_1, supervision_2 = AlexNet(weights_path=weights_path, inputs1=inputs_1, inputs2=inputs_2, id='', trainable=False)
    distance = Merge(mode=L2Distance, output_shape=[1], name='distanceMerge')([embedding_1, embedding_2])
    #distance = Activation('sigmoid')(distance)

    model = Model(input=[inputs_1, inputs_2], output=[distance, supervision_1, supervision_2])
    return model


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
    out = model.predict([img,img-1]) # note: the model has three outputs
    print time.time()-st
    print out
    st = time.time()
    out = model.predict([img,img+1]) # note: the model has three outputs
    print time.time()-st
    print out

    print model.layers

    import numpy as np
    x = np.zeros((64,3,227,227))
    y = np.zeros((64,1))
    print 'begin'
    model.fit([img,img+1],y,batch_size=128,verbose=1)

