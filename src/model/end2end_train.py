#!/usr/bin/env python
# encoding: utf-8
# File Name: alexnet_train.py
# Author: Shaoxiong Wang
# Create Time: 2016/10/05 22:33
# TODO:

from end2end import BiAlex, MAlexNet, MBiAlex
from datalayer import FabricDataset
from keras.layers import Input
import numpy as np
import time
import argparse
import keras
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from collections import Counter
import keras.backend as K
from scipy.misc import imread
import scipy.io
import random
import rank_metrics
from score import get_score

margin = 1
isRegr = False


nb_cluster = 8

data = FabricDataset(given_type='kinect_depth_new', split_by='all', nb_cluster=nb_cluster)
#data = FabricDataset(given_type='canon_resize', split_by='all', nb_cluster=nb_cluster)
#data = FabricDataset(given_type='kinect_depth_new', split_by='home', nb_cluster=nb_cluster)
# data = FabricDataset(given_type='kinect_depth', split_by='all', nb_cluster=nb_cluster)

#data = FabricDataset(given_type='kinect_depth_all', split_by='all', nb_cluster=nb_cluster)
#data = FabricDataset(given_type='kinect_depth', split_by='shop', nb_cluster=nb_cluster)
#data = FabricDataset(given_type='kinect_depth_new', split_by='shop', nb_cluster=nb_cluster)
#data = FabricDataset(given_type='canon', split_by='shop', nb_cluster=nb_cluster)

gel_depth = True
multi_press = 3
n = data.num_test_example(gel_depth=gel_depth)
test_samples = next(data.generate_arrays_from_file(n, train=False, gel_depth=gel_depth, E2E=True,multi_press=multi_press,regr=isRegr))


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def get_gel_id(fn):
    return int(fn.split('/')[-1][1:5])

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
                                    [model.get_layer('1dense_4_embedding').output])
        get_4rd_layer_output = K.function([model.get_layer("inputs_2").input, K.learning_phase()],
                                    [model.get_layer('2dense_4_embedding').output])
        for fn in img_fns:
            frame = data.imread(fn, mode='RGB')
            frame = data._transpose(frame)
            embedding = get_3rd_layer_output([[frame], 0])
            img_embeddings.append(embedding)
        combines = []
        for fn in gel_fns:
            frame = data.imread(fn)
            frame = data._transpose(frame)
            count_2 = 0
            for fn_2 in random.sample(gel_fns,len(gel_fns)):
                if get_gel_id(fn) != get_gel_id(fn_2):
                    continue
                count_2 += 1
                frame_2 = data.imread(fn_2)
                frame_2 = data._transpose(frame_2)
                count_3 = 0
                for fn_3 in random.sample(gel_fns,len(gel_fns)):
                    if get_gel_id(fn) != get_gel_id(fn_3):
                        continue
                    count_3 += 1
                    frame_3 = data.imread(fn_3)
                    frame_3 = data._transpose(frame_3)
                    if multi_press == 3:
                        embedding = get_4rd_layer_output([np.array([[frame,frame_2,frame_3]],dtype='float'), 0])
                        combines.append(', '.join([fn, fn_2, fn_3]))
                    elif multi_press == 1:
                        embedding = get_4rd_layer_output([np.array([frame],dtype='float'), 0])
                        combines.append(fn)
                    gel_embeddings.append(embedding)
                    if count_3 >= 1:
                        break
                if count_2 >= 1:
                    break
    scipy.io.savemat(logdir+'/embedding'+save_fn+'.mat', {'img_embeddings':img_embeddings, 'gel':gel_embeddings, 'img_fns':img_fns, 'gel_fns':combines})

    return img_embeddings, img_fns, gel_embeddings, gel_fns


def evaluate(model, data, logdir, epoch, out_f, gpu):
    get_embedding(model, data, logdir, gpu, test=False)
    img_embeddings, img_fns, gel_embeddings, gel_fns = get_embedding(model, data, logdir, gpu, test=True)
    precision = get_score(img_embeddings, img_fns, gel_embeddings, gel_fns)
    return precision

    nb_img = len(img_embeddings)
    nb_gel = len(gel_embeddings)
    distance_matrix = np.zeros((nb_gel, nb_img))
    img_embeddings = np.array(img_embeddings)
    gel_embeddings = np.array(gel_embeddings)
    dim_embedding = img_embeddings.shape[-1]
    img_embeddings = img_embeddings.reshape((nb_img, dim_embedding))
    gel_embeddings = gel_embeddings.reshape((nb_gel, dim_embedding))

    scores = []
    for i in range(nb_gel):
        distance_matrix[i,:] = np.mean(np.square(img_embeddings - gel_embeddings[i,:]), axis=1).T

        r = []
        for j in range(nb_img):
            if (get_gel_id(img_fns[j]) == get_gel_id(gel_fns[i])):
                r.append(1)
            else:
                r.append(0)
        d = distance_matrix[i, :].tolist()
        a = zip(d, r)
        a = sorted(a, key=lambda d:d[0])
        r = [x[1] for x in a]
        ndcg = [rank_metrics.ndcg_at_k(r, k) for k in [10, 20, 30]]
        precision = [rank_metrics.precision_at_k(r, k) for k in [10, 20, 30]]
        scores.append(ndcg + precision)

    scores = np.array(scores)
    scores = np.mean(scores, axis=0)
    print "ndcg & precision", scores
    print >>out_f, "ndcg & precision", scores
    #return scores[0]

def save_prediction(model, data, logdir, epoch, out_f, gpu):
    pred = []
    true = []
    details = []
    # n = 500
    n = data.num_test_example()
    batch_size = 256
    print "N=%d"%n
    inputs, label, detail = test_samples
    img, gel = inputs

    with tf.device('/gpu:%d'%(gpu)):
        pred_y = model.predict([img, gel], batch_size=batch_size, verbose=1)

    print len(pred_y[0])
    print len(label[0])
    
    for k in range(len(detail)):
        pred.append(pred_y[0][k])
        true.append(label[0][k][0])
        details.append(detail[k])
    print "predict OK"
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    top30 = 0
    top50 = 0
    rank = 0
    result = []
    samples = 0
    for i in range(len(pred)+1):
        # print true[i*100],
        if i > 0 and (i == len(pred) or (
            (gel_depth and details[i][1] != details[i-1][1])
            or (not gel_depth and details[i][0] != details[i-1][0]))):
            samples += 1
            result = sorted(result, key=lambda d:d[1])
            if samples < 2:
                print result
            # print result
            for j in range(len(result)):
                if result[j][0] == 0:
                    break
            if j < 1:
                top1 += 1
            if j < 3:
                top3 += 1
            if j < 5:
                top5 += 1
            if j < 10:
                top10 += 1
            if j < 30:
                top30 += 1
            if j < 50:
                top50 += 1
            rank += j
            result = []
        if i == len(pred): break
        result.append([true[i], pred[i]])
    print >>out_f, "rank", rank * 1.0 / samples
    print >>out_f, "top 1", top1 * 1.0 / samples
    print >>out_f, "top 3", top3 * 1.0 / samples
    print >>out_f, "top 5", top5 * 1.0 / samples
    print >>out_f, "top 10", top10 * 1.0 / samples
    print >>out_f, "top 30", top30 * 1.0 / samples
    print >>out_f, "top 50", top50 * 1.0 / samples

    print "rank", rank * 1.0 / samples
    print "top 1", top1 * 1.0 / samples,
    print "top 3", top3 * 1.0 / samples,
    print "top 5", top5 * 1.0 / samples
    # print "top 10", top10 * 1.0 / samples
    # print "top 30", top30 * 1.0 / samples
    # print "top 50", top50 * 1.0 / samples

    #np.savez(logdir+'/pred_'+str(epoch)+'.npz', pred=pred, true=true, details=details)
    return (top1 * 1.0) / samples

def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    global margin
    Q = 4
    print "margin =", margin

    label = K.transpose(y)
    p = K.transpose(d)
    return K.mean((1 - label) * p + label * K.maximum(margin - p, 0))
    # return K.mean((1 - label) * Q * K.square(d) + label * K.exp(-2.77 / Q * d))

def multi_contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    global margin
    Q = 4
    print "margin =", margin

    label = K.transpose(y)
    p = K.transpose(d)
    return K.mean(K.cast(K.equal(label, 0), 'float') * p +
            K.cast(K.equal(label, 1), 'float') * ((K.maximum(margin - p, 0)))+
                K.cast(K.equal(label, 2), 'float') * K.maximum(margin - p, 0))
    # return K.mean((1 - label) * Q * K.square(d) + label * K.exp(-2.77 / Q * d))


def main(logdir='default', number=None, cat='_softness_avg', gpu='0', weight_fn='../../data/weights/alexnet_weights.h5', test=False, given_margin=1):

    if given_margin:
        global margin
        margin = eval(given_margin)
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
    shutil.copy('end2end.py', logdir+'/'+'end2end.py')
    shutil.copy('end2end_train.py', logdir+'/'+'end2end_train.py')
    shutil.copy('datalayer.py', logdir+'/'+'datalayer.py')


    st = time.time()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    print gpu
    with tf.device('/gpu:%d'%(gpu)):
        weights_fn = '../../data/weights/alexnet_weights.h5'

        if multi_press > 1:
            model, model_img, model_gel = MBiAlex(weights_fn, weights_fn, finetune=False)
        else:
            model, model_img, model_gel = BiAlex(weights_fn, weights_fn, finetune=False)

        # model.compile(loss=my_loss, optimizer='adam')
        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        supervision_loss = 'categorical_crossentropy'
        if isRegr:
            supervision_loss = 'mean_squared_error'

        model.compile(loss=[multi_contrastive_loss,
            supervision_loss,
            supervision_loss],
            optimizer=opt,
            metrics=['mean_absolute_error','accuracy'],
            loss_weights=[1, 1, 1]
            #loss_weights=[1, 1, 0.001]
            #loss_weights=[1, 0.1, 0.1]
            )

        max_acc = 0
        output = open(logdir+'/log.txt','w')

        sample_method = data.generate_arrays_from_file(128,train=True,E2E=True,multi_press=multi_press,regr=isRegr)

        f_loss = open(logdir+'/loss.txt', 'w')
        f_precision = open(logdir+'/precision.txt', 'w')
        history = LossHistory()

        for epoch in range(50):

            x, y = next(data.generate_arrays_from_file(10,train=True,SNN=True,gel=True,E2E=True,multi_press=multi_press,regr=isRegr))
            pred = model.predict(x, batch_size=128)
            print pred[0]

            #top1 = save_prediction(model, data, logdir, epoch, output, gpu)
            precision = evaluate(model, data, logdir, epoch, output, gpu)
            top1 = precision[0]

            shutil.copy(logdir+'/embedding_train.mat', logdir+'/embedding_train_%d_%s.mat'%(epoch, str(precision)))
            shutil.copy(logdir+'/embedding_test.mat', logdir+'/embedding_test_%d_%s.mat'%(epoch,str(precision)))

            model.fit_generator(sample_method,
                     samples_per_epoch=32800, nb_epoch=1, nb_worker=8, callbacks=[history])
            print "epoch ", epoch, "time ", time.time()-st
            print >>output, "epoch ", epoch, "time ", time.time()-st
            
            model.save(logdir+"/weights.hdf5")

            print >>f_loss, history.losses
            print >>f_precision, precision

        f_loss.close()
        f_precision.close()


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
    parser.add_argument('-m','--given_margin',
            metavar='given_margin',
            help='margin')
    args = parser.parse_args()
    kwargs = dict(args._get_kwargs())
    main(**kwargs)
