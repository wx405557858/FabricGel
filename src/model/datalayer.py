#!/usr/bin/env python
# encoding: utf-8
# File Name: datalayer.py
# Author: Shaoxiong Wang
# Create Time: 2016/10/06 17:32
# TODO:

import cv2
from scipy.misc import imread, imresize
import numpy as np
import random
import sys
from preprocess import Split
from pprint import pprint
from load_labels import physical
import copy
from physicals import physicals
import threading
category = "_softness_avg_reg"

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class FabricDataset():
    def __init__(self, given_type='kinect_depth', split_by='home', train=True, nb_cluster=6):
        self.train = train
        self.given_type = given_type
        self.split_by = split_by

        split = Split()
        if split_by == 'home':
            self.train_img, self.test_img, self.train_gel, self.test_gel = split.split_home(given_type)
        elif split_by == 'shop':
            self.train_img, self.test_img, self.train_gel, self.test_gel = split.split_shop(given_type)
        else:
            self.train_img, self.test_img, self.train_gel, self.test_gel = split.split_all(given_type)


        random.seed(0)

        if given_type == 'kinect_depth':
            self.crop_x = 60
            self.crop_y = 100
            self.width = 227
        elif given_type == 'canon':
            self.crop_x = 250
            self.crop_y = 1100
            self.width = 3000

        self.load_all_images()

        labels = []
        if nb_cluster == 6:
            f = open('../../data/label/classes.txt','r')
        else:
            f = open('../../data/label/classes_%d.txt'%nb_cluster,'r')
        for line in f.readlines():
            labels.append(int(line.rstrip()))
        self.labels = labels

        self.it = 0
        self.nb_cluster = nb_cluster
        random.seed(0)
        pass

    def num_example(self):
        return 100

    def num_test_example(self,gel_depth=True):
        if gel_depth:
            count = 0
            for id in self.test_gel:
                count += len(self.test_gel[id])
            count *= 10#len(self.test_img)
        else:
            count = 0
            for id in self.test_img:
                count += len(self.test_img[id])
            count *= 10#len(self.test_gel)
        return count

    def num_class(self):
        return 100



    def crop(self, frame):
        '''
        crop the center part of image and resize to 227 * 227
        '''
        # frame[:, :, 0] = frame[:,:,0] - 123.68
        # frame[:, :, 1] = frame[:,:,1] - 103.939
        # frame[:, :, 2] = frame[:,:,2] - 116.779
        width, height = frame.shape[0], frame.shape[1]
        if width == 720:
            frame = frame[:,120:120+720]
        else:
            frame = frame[self.crop_x:self.crop_x+self.width, self.crop_y:self.crop_y+self.width]
        # print frame.shape
        if self.width != 227:
            frame = imresize(frame, (227,227))
        return frame


    def next_classification_sample(self, imgs, loop=False):
        '''
        yield next training sample

        return:
            (img, y, detail)
            img : the rgb/depth image of a fabric
        '''
        labels = self.labels
        while True:
            for id in imgs:
                for fn_img in imgs[id]:
                    frame_img = self.imread(fn_img, mode='RGB')

                    detail = fn_img
                    # y = np.zeros((self.num_example()))
                    # y[id-1] = 1
                    y = np.zeros((self.nb_cluster))
                    y[labels[id-1]] = 1
                    yield (frame_img, y, detail)
            if not loop:
                break


    def next_classification_sample_gel(self, imgs, loop=False, multi_press=1):
        '''
        yield next training sample

        return:
            (img, y, detail)
            img : the rgb/depth image of a fabric
        '''
        labels = self.labels
        while True:
            for id in imgs:
                for date in imgs[id]:
                    # for fn_img in imgs[id]:
                    fn_img = imgs[id][date][-2]
                    frame_img = self.imread(fn_img, mode='RGB')
                    detail = fn_img
                    y = np.zeros((self.nb_cluster))
                    y[labels[id-1]] = 1

                    if multi_press > 1:
                        frame_imgs = [frame_img]
                        details = [fn_img]
                        ys = [y]
                        for k in range(multi_press-1):
                            date_k = random.choice(imgs[id].keys())
                            fn_img_k = imgs[id][date_k][-2]
                            frame_img_k = self.imread(fn_img_k, mode='RGB')
                            frame_imgs.append(frame_img_k)
                            details.append(fn_img_k)
                            ys.append(y)
                        frame_img = frame_imgs
                        detail = details
                        #y = ys

                    # y = np.zeros((self.num_example()))
                    # y[id-1] = 1
                    yield (frame_img, y, detail)
            if not loop:
                break

    def next_train_sample(self, negative_rato=0.8):
        '''
        yield next training sample

        return:
            (img, gel, y, detail)
            img : the rgb/depth image of a fabric
            gel : the gelsight image of a fabric
        '''
        positive_number = 0
        negative_number = 0
        while True:
            keys = self.train_gel.keys()
            random.shuffle(keys)
            for id in keys:
                for fn_img in self.train_img[id]:
                    for id_gel in self.train_gel[id]:
                        fn_gel = self.train_gel[id][id_gel][-2]

                        frame_img = self.imread(fn_img, mode='RGB')
                        frame_gel = self.imread(fn_gel)

                        detail = [fn_img, fn_gel]
                        positive_number += 1

                        yield (frame_img, frame_gel, 0, detail)

                        while negative_number < (positive_number + negative_number) * negative_rato:
                            # randomly sample negative samples
                            negative_id = random.choice(self.train_img.keys())
                            negative_sample = random.choice(self.train_img[negative_id])
                            # print negative_sample

                            frame_img = self.imread(negative_sample, mode='RGB')

                            detail = [negative_sample, fn_gel]
                            negative_number += 1

                            yield [frame_img, frame_gel, 1, detail]


    def next_train_sample_SNN(self, negative_rato=0.8, gel=False):
        '''
        yield next training sample

        return:
            (img, gel, y, detail)
            img : the rgb/depth image of a fabric
            gel : the gelsight image of a fabric
        '''
        positive_number = 0
        negative_number = 0

        if gel:
            set2 = self.train_gel
        else:
            set2 = self.train_img
        while True:
            for id in self.train_img:
                for fn_img in set2[id]:
                    if gel:
                        fn_img = set2[id][fn_img][-2]
                        frame_img = self.imread(fn_img)
                    else:
                        frame_img = self.imread(fn_img, mode='RGB')

                    for fn_img_2 in set2[id]:
                        if gel:
                            fn_img_2 = set2[id][fn_img_2][-2]
                            frame_gel = self.imread(fn_img_2)
                        else:
                            frame_gel = self.imread(fn_img_2, mode='RGB')

                        detail = [fn_img, fn_img_2]
                        positive_number += 1
                        yield (frame_img, frame_gel, 0, detail)

                        while negative_number < (positive_number + negative_number) * negative_rato:
                            # randomly sample negative samples
                            negative_id = random.choice(self.train_img.keys())
                            while negative_id == id:
                                negative_id = random.choice(self.train_img.keys())

                            if gel:
                                negative_sample = random.choice(self.train_gel[negative_id].keys())
                                negative_sample = self.train_gel[negative_id][negative_sample][-2]
                                frame_img = self.imread(negative_sample)
                            else:
                                negative_sample = random.choice(self.train_img[negative_id])
                                frame_img = self.imread(negative_sample, mode='RGB')

                            # print negative_sample


                            detail = [negative_sample, fn_img_2]
                            negative_number += 1
                            '''
                            physical1 = physical[id-1]
                            physical2 = physical[negative_id-1]
                            d = np.sqrt(np.mean((physical1 - physical2)**2))
                            if d < 0.2:
                                l = 0
                            else:
                                l = 1
                            '''

                            yield [frame_img, frame_gel, 1, detail]


    def next_train_sample_E2E(self, negative_rato=0.8, multi_press=1, regr=False):
        '''
        yield next training sample

        return:
            (img, gel, y, detail)
            img : the rgb/depth image of a fabric
            gel : the gelsight image of a fabric
        '''
        labels = self.labels
        positive_number = 0
        negative_number = 0
        while True:
            keys = self.train_gel.keys()
            random.shuffle(keys)
            for id in keys:
                for gel_date in [random.choice(self.train_gel[id].keys())]:
                    for fn_img in [random.choice(self.train_img[id])]:
                        fn_gel = self.train_gel[id][gel_date][-2]

                        frame_img = self.imread(fn_img, mode='RGB')
                        frame_gel = self.imread(fn_gel)

                        positive_number += 1

                        if regr == False:
                            y_img = np.zeros((self.nb_cluster))
                            y_img[labels[id-1]] = 1

                            y_gel = np.zeros((self.nb_cluster))
                            y_gel[labels[id-1]] = 1
                        else:
                            y_img = physicals[id-1]
                            y_gel = physicals[id-1]


                        detail = [fn_img, fn_gel]
                        # sample multi-press of gelsight
                        if multi_press > 1:
                            frame_imgs = [frame_gel]
                            ys_gel = [y_gel]
                            for k in range(multi_press-1):
                                date_k = random.choice(self.train_gel[id].keys())
                                fn_img_k = self.train_gel[id][date_k][-2]
                                frame_img_k = self.imread(fn_img_k)
                                frame_imgs.append(frame_img_k)
                                ys_gel.append(y_gel)
                            frame_gel = frame_imgs
                            #y_gel = ys_gel

                        yield (frame_img, frame_gel, [0, y_img, y_gel], detail)

                        while negative_number < (positive_number + negative_number) * negative_rato:
                            # randomly sample negative samples
                            negative_id = random.choice(self.train_img.keys())
                            negative_sample = random.choice(self.train_img[negative_id])
                            # print negative_sample

                            frame_img = self.imread(negative_sample, mode='RGB')

                            if regr == False:
                                y_img = np.zeros((self.nb_cluster))
                                y_img[labels[negative_id-1]] = 1
                            else:
                                y_img = physicals[negative_id-1]


                            detail = [negative_sample, fn_gel]

                            negative_number += 1
                            # print "NEG", frame_gel[0].shape

                            if negative_id == id:
                                d = 0
                            elif labels[negative_id-1] == labels[id-1]:
                                d = 1
                            else:
                                d = 2
                            yield [frame_img, frame_gel, [d, y_img, y_gel], detail]





    def next_test_sample(self, gel_depth=True):
        '''
        yield next test sample

        return:
            (img, gel, y, detail)
            img : the rgb/depth image of a fabric
            gel : the gelsight image of a fabric
        '''
        if gel_depth:
            while True:
                # for each fabric id
                for id in self.test_gel:
                    # for each gelsight press
                    for id_gel in self.test_gel[id]:
                        # load the gelsight image
                        fn_gel = self.test_gel[id][id_gel][-2]
                        frame_gel = self.imread(fn_gel)

                        # for each candidate fabric id

                        candidate = self.test_img.keys()
                        del candidate[candidate.index(id)]
                        candidate = random.sample(candidate, 9)
                        candidate.append(id)
                        random.shuffle(candidate)

                        for id_img in candidate:
                        #for id_img in self.test_img:
                            # random sample an image
                            fn_img = random.choice(self.test_img[id_img])

                            frame_img = self.imread(fn_img, mode='RGB')

                            detail = [fn_img, fn_gel]

                            # decide the label
                            if id == id_img:
                                label = 0
                            else:
                                label = 1

                            yield (frame_img, frame_gel, label, detail)
        else:
            while True:
                # for each fabric id
                for id in self.test_img:
                    # for each gelsight press
                    for fn_img in self.test_img[id]:
                        # load the gelsight image
                        frame_img = self.imread(fn_img, mode='RGB')

                        candidate = self.test_gel.keys()
                        del candidate[candidate.index(id)]
                        candidate = random.sample(candidate, 9)
                        candidate.append(id)
                        random.shuffle(candidate)

                        # for each candidate fabric id
                        for id_gel in candidate:
                        #for id_gel in self.test_gel:
                            # random sample an image
                            date_gel = random.choice(self.test_gel[id_gel].keys())
                            fn_gel = self.test_gel[id_gel][date_gel][-2]

                            frame_gel = self.imread(fn_gel)

                            detail = [fn_img, fn_gel]

                            # decide the label
                            if id == id_gel:
                                label = 0
                            else:
                                label = 1

                            yield (frame_img, frame_gel, label, detail)



    def next_test_sample_E2E(self, gel_depth=True, multi_press=1):
        '''
        yield next test sample

        return:
            (img, gel, y, detail)
            img : the rgb/depth image of a fabric
            gel : the gelsight image of a fabric
        '''
        labels = self.labels
        if gel_depth:
            while True:
                # for each fabric id
                keys = self.test_gel.keys()
                random.shuffle(keys)
                for id in keys:
                    # for each gelsight press
                    for id_gel in self.test_gel[id]:
                        # load the gelsight image
                        fn_gel = self.test_gel[id][id_gel][-2]
                        frame_gel = self.imread(fn_gel)

                        candidate = self.test_img.keys()
                        del candidate[candidate.index(id)]
                        candidate = random.sample(candidate, 9)
                        candidate.append(id)
                        random.shuffle(candidate)

                        y_gel = np.zeros((self.nb_cluster))
                        y_gel[labels[id-1]] = 1

                        # sample multi-press of gelsight
                        if multi_press > 1:
                            frame_imgs = [frame_gel]
                            ys_gel = [y_gel]
                            for k in range(multi_press-1):
                                date_k = random.choice(self.test_gel[id].keys())
                                fn_img_k = self.test_gel[id][date_k][-2]
                                frame_img_k = self.imread(fn_img_k)
                                frame_imgs.append(frame_img_k)
                                ys_gel.append(y_gel)
                            frame_gel = frame_imgs
                            #y_gel = ys_gel

                        # for each candidate fabric id
                        for id_img in candidate:
                        #for id_img in self.test_img:
                            # random sample an image
                            fn_img = random.choice(self.test_img[id_img])

                            frame_img = self.imread(fn_img, mode='RGB')

                            detail = [fn_img, fn_gel]

                            # decide the label
                            if id == id_img:
                                label = 0
                            else:
                                label = 1


                            y_img = np.zeros((self.nb_cluster))
                            y_img[labels[id_img-1]] = 1



                            yield (frame_img, frame_gel, [label, y_img, y_gel], detail)
        else:
            while True:
                # for each fabric id
                keys = self.test_img.keys()
                random.shuffle(keys)
                for id in keys:
                    # for each gelsight press
                    for fn_img in self.test_img[id]:
                        # load the gelsight image
                        frame_img = self.imread(fn_img, mode='RGB')


                        candidate = self.test_gel.keys()
                        del candidate[candidate.index(id)]
                        candidate = random.sample(candidate, 9)
                        candidate.append(id)
                        random.shuffle(candidate)

                        y_img = np.zeros((self.nb_cluster))
                        y_img[labels[id-1]] = 1


                        # for each candidate fabric id
                        for id_gel in candidate:
                        #for id_gel in self.test_gel:
                            # random sample an image
                            date_gel = random.choice(self.test_gel[id_gel].keys())
                            fn_gel = self.test_gel[id_gel][date_gel][-2]

                            frame_gel = self.imread(fn_gel)

                            detail = [fn_img, fn_gel]


                            y_gel = np.zeros((self.nb_cluster))
                            y_gel[labels[id_gel-1]] = 1



                            # sample multi-press of gelsight
                            if multi_press > 1:
                                frame_imgs = [frame_gel]
                                ys_gel = [y_gel]
                                for k in range(multi_press-1):
                                    date_k = random.choice(self.train_gel[id_gel].keys())
                                    fn_img_k = self.train_gel[id_gel][date_k][-2]
                                    frame_img_k = self.imread(fn_img_k)
                                    frame_imgs.append(frame_img_k)
                                    ys_gel.append(y_gel)
                                frame_gel = frame_imgs
                                #y_gel = ys_gel

                            # decide the label
                            if id == id_gel:
                                label = 0
                            else:
                                label = 1

                            yield (frame_img, frame_gel, [label, y_img, y_gel], detail)


    def next_test_sample_SNN(self,gel=False):
        '''
        yield next test sample

        return:
            (img, gel, y, detail)
            img : the rgb/depth image of a fabric
            gel : the gelsight image of a fabric
        '''
        while True:
            # for each fabric id
            for id in self.test_img:
                if gel:
                    set2 = self.test_gel[id]
                else:
                    set2 = range(len(self.test_img[id]))
                # for each gelsight press
                for id_gel in set2:
                    # load the gelsight image
                    if gel:
                        fn_gel = self.test_gel[id][id_gel][-2]
                        frame_gel = self.imread(fn_gel)
                    else:
                        fn_gel = self.test_img[id][id_gel]
                        frame_gel = self.imread(fn_gel, mode='RGB')

                    candidate = self.test_gel.keys()
                    del candidate[candidate.index(id)]
                    candidate = random.sample(candidate, 9)
                    candidate.append(id)
                    random.shuffle(candidate)

                    # for each candidate fabric id
                    for id_img in candidate:
                        # random sample an image
                        # fn_img = random.choice(self.test_img[id_img])

                        # frame_img = self.imread(fn_img, mode='RGB')

                        if gel:
                            fn_img = random.choice(self.test_gel[id_img].keys())
                            fn_img = self.test_gel[id_img][fn_img][-2]
                            frame_img = self.imread(fn_img)
                        else:
                            fn_img = random.choice(self.test_img[id_img])
                            frame_img = self.imread(fn_img, mode='RGB')

                        detail = [fn_img, fn_gel]

                        # decide the label
                        if id == id_img:
                            label = 0
                        else:
                            label = 1

                        yield (frame_img, frame_gel, label, detail)

    def _transpose(self, img):
        frame = img
        # frame[:, :, 0] -= 123.68
        # frame[:, :, 1] -= 116.779
        # frame[:, :, 2] -= 103.939
        # img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        frame = frame.transpose((2, 0, 1))
        return frame

    def load_all_images(self, lstm=False):
        self.imageset = {}
        if lstm:
            frame_list = [1, 3, 5, 7, -2]
        else:
            frame_list = [-2]
        for id in self.train_img:
            for fn in self.train_img[id]:
                frame = imread(fn, mode='RGB')
                self.imageset[fn] = frame

            for date in self.train_gel[id]:
                for fid in frame_list:
                    fn = self.train_gel[id][date][fid]
                    frame = imread(fn)
                    self.imageset[fn] = frame

        for id in self.test_img:
            for fn in self.test_img[id]:
                frame = imread(fn, mode='RGB')
                self.imageset[fn] = frame
            for date in self.test_gel[id]:
                for fid in frame_list:
                    fn = self.test_gel[id][date][fid]
                    frame = imread(fn)
                    self.imageset[fn] = frame

    def imread(self, fn, mode=''):
        img = self.imageset[fn]
        #noise = np.random.uniform(-1,1,size=img.shape)
        #img = img + noise
        if self.train and random.random() > 0.8:
            d = random.randint(180, 227)
            x = random.randint(0, 227-d)
            y = random.randint(0, 227-d)
            img = imresize(img[x:x+d, y:y+d], (227,227))
        return img

    @threadsafe_generator
    def generate_arrays_from_file(self, batch_size, train=True, SNN=False, gel=False, gel_depth=True, E2E=False, multi_press=1, regr=False):
        imgs, gels, labels, details = [], [], [], []
        if E2E:
            labels = [[], [], []]
        count = 0
        if train:
            if E2E:
                sample_method = self.next_train_sample_E2E(0.5, multi_press=multi_press, regr=regr)
            elif SNN:
                sample_method = self.next_train_sample_SNN(0.5,gel=gel)
            else:
                sample_method = self.next_train_sample(0.5)
        else:
            if E2E:
                sample_method = self.next_test_sample_E2E(gel_depth=gel_depth, multi_press=multi_press)
            elif SNN:
                sample_method = self.next_test_sample_SNN(gel=gel)
            else:
                sample_method = self.next_test_sample(gel_depth=gel_depth)


        while True:
            # x, y, z, p = self.next_batch(batch_size)
            for sample in sample_method:
                count += 1

                img, gel, label, detail = sample
                img = self._transpose(img)
                # gel = self._transpose(gel)


                if type(gel) == list:
                    if gel[0].shape[0] != 3:
                        for i in range(len(gel)):
                            gel[i] = self._transpose(gel[i])
                else:
                    gel = self._transpose(gel)

                imgs.append(img)
                gels.append(gel)
                details.append(detail)
                if E2E:
                    labels[0].append([label[0]])
                    labels[1].append(label[1])
                    labels[2].append(label[2])
                else:
                    labels.append([label])

                if count >= batch_size:
                    # pprint(imgs)
                    imgs = np.array(imgs)
                    gels = np.array(gels)
                    if E2E:
                        for i in range(3):
                            labels[i] = np.array(labels[i])
                    else:
                        labels = np.array(labels)
                    if train:
                        yield ([imgs, gels], labels)#, details)
                    else:
                        yield ([imgs, gels], labels, details)

                    imgs, gels, labels, details = [], [], [], []
                    if E2E:
                        labels = [[], [], []]
                    count = 0

    @threadsafe_generator
    def generate_classification(self, batch_size, train=True, gel=False, multi_press=1):
        imgs, labels, details = [], [], []
        count = 0
        while True:

            if gel:
                if train:
                    sample_method = self.next_classification_sample_gel(self.train_gel, multi_press=multi_press)
                else:
                    sample_method = self.next_classification_sample_gel(self.test_gel, multi_press=multi_press)
            else:
                if train:
                    sample_method = self.next_classification_sample(self.train_img)
                else:
                    sample_method = self.next_classification_sample(self.test_img)

            # x, y, z, p = self.next_batch(batch_size)
            for sample in sample_method:
                count += 1

                img, label, detail = sample
                if type(img) == list:
                    for i in range(len(img)):
                        img[i] = self._transpose(img[i])
                else:
                    img = self._transpose(img)

                imgs.append(img)
                labels.append(label)
                details.append(detail)
                if batch_size != -1 and count >= batch_size:
                    # pprint(imgs)
                    imgs = np.array(imgs)
                    labels = np.array(labels)
                    if train:
                        yield (imgs, labels)
                    else:
                        yield (imgs, labels, details)

                    imgs, labels, details = [], [], []
                    count = 0

            if batch_size == -1:
                imgs = np.array(imgs)
                labels = np.array(labels)
                if train:
                    yield (imgs, labels)
                else:
                    yield (imgs, labels, details)
                imgs, labels, details = [], [], []
                count = 0

def main():
    dataset = FabricDataset(given_type='canon', split_by='home', train=True)

    pprint(next(dataset.generate_arrays_from_file(3)))
    pprint(next(dataset.generate_arrays_from_file(3)))


if __name__ == "__main__":
    main()
    pass
