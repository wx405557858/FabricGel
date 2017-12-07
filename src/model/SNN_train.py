#!/usr/bin/env python
# encoding: utf-8
# File Name: alexnet_train.py
# Author: Shaoxiong Wang
# Create Time: 2016/10/05 22:33
# TODO:

from SNN import SNN
from datalayer import FabricDataset
import numpy as np
import time
import argparse
import keras
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from collections import Counter
import keras.backend as K
import scipy.io

def evaluate():
    pass

data = FabricDataset(given_type='kinect_depth_new', split_by='all', nb_cluster=8)
isGel = False
#data = FabricDataset(given_type='canon', split_by='shop')

gel_depth = False
n = data.num_test_example(gel_depth=gel_depth)
test_samples = next(data.generate_arrays_from_file(n, train=False, SNN=True, gel=isGel))


def get_embedding(model, data, logdir, gpu, test):
    if test:
        imgs = data.test_img
        gels = data.test_gel
        save_fn = '_test'
    else:
        imgs = data.train_img
        gels = data.train_gel
        save_fn = '_train'

    img_fns = []
    gel_fns = []
    for i in imgs:
        img_fns = img_fns + imgs[i]
    for i in gels:
        for date in gels[i]:
            gel_fns.append(gels[i][date][-2])

    img_embeddings = []
    gel_embeddings = []
    with tf.device('/gpu:%d'%(gpu)):
        get_3rd_layer_output = K.function([model.get_layer("inputs_1").input, K.learning_phase()],
                                    [model.get_layer('dense_2_gel').get_output_at(0)])
        for fn in img_fns:
            frame = data.imread(fn, mode='RGB')
            frame = data._transpose(frame)
            embedding = get_3rd_layer_output([[frame], 0])
            img_embeddings.append(embedding)
    scipy.io.savemat(logdir+'/embedding'+save_fn+'.mat', {'img_embeddings':img_embeddings, 'img_fns':img_fns})

    return img_embeddings, img_fns


def save_prediction(model, data, logdir, epoch, out_f, gpu):
    get_embedding(model, data, logdir, gpu, False)
    get_embedding(model, data, logdir, gpu, True)
    #exit(0)

    pred = []
    true = []
    details = []
    # n = 500
    batch_size = 128
    count = 0
    print "N=%d"%n
    inputs, label, detail = test_samples
    img, gel = inputs

    with tf.device('/gpu:%d'%(gpu)):
        pred_y = model.predict([img, gel], batch_size=batch_size, verbose=1)
    # print img.shape, gel.shape, detail
    for k in range(len(detail)):
        pred.append(pred_y[0][k])
        true.append(label[0][k][0])
        details.append(detail[k])
    print "predict OK"
    
    top5 = 0
    top10 = 0
    top20 = 0
    rank = 0
    result = []
    samples = 0
    for i in range(len(pred)+1):
        # print true[i*100],
        if i > 0 and (i == len(pred) or details[i][1] != details[i-1][1]):
            samples += 1
            result = sorted(result, key=lambda d:d[1])
            if samples < 3:
                print result
            # print result
            for j in range(len(result)):
                if result[j][0] == 0:
                    break
            if j < 1:
                top5 += 1
            if j < 3:
                top10 += 1
            if j < 5:
                top20 += 1
            print j,
            rank += j
            result = []
        if i == len(pred): break
        result.append([true[i], pred[i]])
    print >>out_f, "rank", rank * 1.0 / samples
    print >>out_f, "top 5", top5 * 1.0 / samples
    print >>out_f, "top 10", top10 * 1.0 / samples
    print >>out_f, "top 20", top20 * 1.0 / samples

    print "rank", rank * 1.0 / samples
    print "top 1", top5 * 1.0 / samples
    print "top 3", top10 * 1.0 / samples
    print "top 5", top20 * 1.0 / samples

    np.savez(logdir+'/pred_'+str(epoch)+'.npz', pred=pred, true=true, details=details)
    return top5 * 1.0 / samples

def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 2
    Q = 100
    label = K.transpose(y)
    return K.mean((1 - label) * d + label * K.maximum(margin - d, 0))

def main(logdir='default', number=None, cat='_softness_avg', gpu='0', weight_fn='../../data/weights/alexnet_weights.h5', test=False):


    if gpu:
        gpu = int(gpu)

    try:
        logdir = "../result/logs/" + logdir + cat
        import os
        os.mkdir(logdir)

    except:
        print logdir, "already exist"

    if not weight_fn:
        weight_fn='../../data/weights/alexnet_weights.h5'

    print logdir
    import shutil
    shutil.copy('alexnet.py', logdir+'/'+'alexnet.py')
    shutil.copy('alexnet_train.py', logdir+'/'+'alexnet_train.py')
    shutil.copy('datalayer.py', logdir+'/'+'datalayer.py')


    st = time.time()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    print gpu
    with tf.device('/gpu:%d'%(gpu)):
        weights_fn = '../../data/weights/alexnet_weights.h5'
        model = SNN(weights_fn)
        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(loss=[contrastive_loss, 'categorical_crossentropy','categorical_crossentropy'],
                optimizer=opt, metrics=['accuracy'],
                    loss_weights=[1, 0.001, 0.001])

        if test == 'test':

            get_acc(model, data_test, logdir, -1)
            return

        losses = []
        accs = []
        batch_size = 128
        max_acc = 0
        output = open(logdir+'/log.txt','w')
        sample_method = data.generate_arrays_from_file(128,train=True,SNN=True,gel=isGel)
        #model.load_weights('../result/logs/SNN_branch_all_shop/weights0.536.hdf5')
        for epoch in range(100):
            top1 = save_prediction(model, data, logdir, epoch, output, gpu)
            if top1 > max_acc:
                max_acc = top1
                model.save(logdir+'/weights%.3lf.hdf5'%(top1))
                shutil.copy(logdir+'/embedding_train.mat', logdir+'/embedding_train_%.3lf.mat'%max_acc)
                shutil.copy(logdir+'/embedding_test.mat', logdir+'/embedding_test_%.3lf.mat'%max_acc)
                
            model.fit_generator(sample_method,
                    samples_per_epoch=12800, nb_epoch=1)
            print "epoch ", epoch, "time ", time.time()-st
            print >>output, "epoch ", epoch, "time ", time.time()-st
            model.save(logdir+"/weights.hdf5")
            # save_prediction(model, data, logdir, epoch, output, gpu)


if __name__ == "__main__":
    description = """
    Training Alexnet model for fabric joint embedding
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-l','--logdir',
            metavar='logdir',
            help='path to store evaluation')
    parser.add_argument('-c','--cat',
            metavar='cat',
            help='category')
    parser.add_argument('-g','--gpu',
            metavar='gpu',
            help='gpu')
    parser.add_argument('-w','--weight_fn',
            metavar='weight_fn',
            help='weights')
    parser.add_argument('-t','--test',
            metavar='test',
            help='test')
    args = parser.parse_args()
    kwargs = dict(args._get_kwargs())
    main(**kwargs)
