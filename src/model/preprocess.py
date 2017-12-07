#!/usr/bin/env python
# encoding: utf-8
# File Name: preprocess.py
# Author: Shaoxiong Wang
# Create Time: 2016/10/06 19:28
# TODO:

import os, re
from sys import getsizeof
from pprint import pprint
import random


class Split(object):
    def __init__(self):
        print 'init'
        pass


    def load_kinect(self, fn=None, old=False):
    # def load_kinect(self, fn='../../data/Recording_siyuan/kinect_depth/'):
        '''
        load kinect files into dict.

        return:
            fns:dict - {file_id : [fn_1, fn_2, ..., fn_k]}
        '''
        root, dirs, files = next(os.walk(fn))

        fns = {}
        for f in files:
            if f[0] != 'F':
                continue

            id = int(f.split('_')[0][1:]) #F0001 -> id=1
            if old and id > 100:
                continue
            fn = os.path.join(root, f)

            fns.setdefault(id, [])
            #if len(fns[id]) < 8: fns[id].append(fn)
            fns[id].append(fn)
        #for id in fns:
        #    fns[id] = fns[id][8:]

        return fns

    def load_gelsight(self, fn='../../data/resize/image_fold', given_type='fold', all=False):
    #def load_gelsight(self, fn='../../data/resize/image_flat', given_type='flat', all=False):
    # def load_gelsight(self, fn='../../data/resize/image_rand2', given_type='rand2', all=False):
    # def load_gelsight(self, fn='../../data/Recording_siyuan/image', given_type='fold'):
        '''
        load gelsight files into dict.

        return:
            fns:dict - {file_id : {date : [frame_1, frame_2, ..., frame_k]}}
        '''
        root, dirs, files = next(os.walk(fn))
        # set into dict
        fns = {}
        for f in files:
            if f[0] != 'F':
                continue

            fn = os.path.join(root, f)
            id, type, number, date = re.findall('F(\d*) ?_(.*)_(\d*)_(.*).jpg', f)[0]
            id = int(id)

            # select specific type
            # if type != given_type:
                # continue

            fns.setdefault(id, {})
            fns[id].setdefault(date, [])
            fns[id][date].append(fn)

        if all:
            # r2,d2,f2 = next(os.walk('../../data/resize/image_fold_new'))
            r2,d2,f2 = next(os.walk('../../data/resize/image_fold'))
            for f in f2:
                if f[0] != 'F':
                    continue

                fn = os.path.join(r2, f)
                id, type, number, date = re.findall('F(\d*) ?_(.*)_(\d*)_(.*).jpg', f)[0]
                id = int(id)

                # select specific type
                # if type != given_type:
                    # continue

                fns.setdefault(id, {})
                fns[id].setdefault(date, [])
                fns[id][date].append(fn)




        # sort filenames by order
        for id in fns:
            for date in fns[id]:
                fns[id][date] = sorted(fns[id][date], key=lambda d:int(re.findall('F(\d*) ?_(.*)_(\d*)_(.*).jpg', d)[0][2]))

        return fns


    def split_home(self, type='kinect_depth', ratio=0.8):
        '''
        split each fabric's images into 4:1 for training and testing
        split each fabric's gelsight into 4:1 for training and testing

        parameters:
            type:
                kinect_depth or kinect_rgb or canon
            ratio:
                ratio of training and testing

        return:
            train_img, test_img, train_gel, test_gel
        '''

        imgs = self.load_kinect('../../data/resize/%s' % type)
        gels = self.load_gelsight('../../data/resize/image_fold', given_type='fold')

        train_img, test_img, train_gel, test_gel = {}, {}, {}, {}

        test_id = [87, 71, 35, 3, 16, 75, 30, 70, 56, 24, 43, 44, 7, 39, 52, 42, 78, 97, 107]
        train_id = [x for x in imgs.keys() if x not in test_id and x != 107]
        # split for images
        for id in imgs:
            if id not in train_id: continue
            train_number = int(len(imgs[id]) * ratio)
            train_img[id] = imgs[id][:train_number]
            test_img[id] = imgs[id][train_number:]

        # split for gelsight
        for id in gels:
            if id not in train_id:continue
            train_number = int(len(gels[id]) * ratio)

            train_gel[id] = {}
            test_gel[id] = {}

            i = 0
            for date in gels[id]:
                i += 1
                if i <= train_number:
                    train_gel[id][date] = gels[id][date]
                else:
                    test_gel[id][date] = gels[id][date]

        return (train_img, test_img, train_gel, test_gel)


    def split_shop(self, type='kinect_depth', ratio=0.8):
        '''
        split fabric into 4:1 for training and testing

        parameters:
            type:
                kinect_depth or kinect_rgb or canon
            ratio:
                ratio of training and testing

        return:
            train_img, test_img, train_gel, test_gel
        '''

        imgs = self.load_kinect('../../data/resize/%s' % type, old=False)
        gels = self.load_gelsight('../../data/resize/image_fold')

        train_img, test_img, train_gel, test_gel = {}, {}, {}, {}

        # select training id
        random.seed(0)
        # train_id = random.sample(imgs.keys(), int(len(imgs.keys())*ratio))

        # test_id = [84, 2, 35, 7, 9, 43, 12, 78, 15, 80, 17, 75, 83, 20, 99, 55, 57, 92, 93, 63]
        test_id = [87, 71, 35, 3, 16, 75, 30, 70, 56, 24, 43, 44, 7, 39, 52, 42, 78, 97, 107]
        train_id = [x for x in imgs.keys() if x not in test_id and x != 107]

        # split for images
        for id in imgs:
            if id in train_id:
                train_img[id] = imgs[id]
            else:
                test_img[id] = imgs[id]

        # split for gelsight
        for id in gels:
            if id in train_id:
                train_gel[id] = gels[id]
            else:
                test_gel[id] = gels[id]

        return (train_img, test_img, train_gel, test_gel)

    def c2gtransform(self, color):
        gels = {}
        for id in color:
            gels[id] = {}
            for i, img in enumerate(color[id]):
                gels[id][i] = [img] * 3
        return gels

    def split_all(self, type='kinect_depth_all', ratio=0.8):
        '''
        split fabric into 4:1 for training and testing

        parameters:
            type:
                kinect_depth or kinect_rgb or canon
            ratio:
                ratio of training and testing

        return:
            train_img, test_img, train_gel, test_gel
        '''

        imgs = self.load_kinect('../../data/resize/%s' % type)
        # if type != 'kinect_depth_all':
            # all = False
        #gels = self.load_gelsight('../../data/resize/image_fold_wz',all=False)
        #gels = self.load_gelsight('../../data/resize/image_flat',all=False)
        # gels = self.load_gelsight('../../data/resize/image_rand2',all=True)
        colors = self.load_kinect('../../data/resize/canon_resize')
        gels = self.c2gtransform(colors)

        train_img, test_img, train_gel, test_gel = {}, {}, {}, {}

        # select training id
        # train_id = random.sample(imgs.keys(), int(len(imgs.keys())*ratio))
        # train_id = range(1, 101)

        random.seed(0)
        # train_id = random.sample(imgs.keys(), int(len(imgs.keys())*ratio))

        # test_id = [84, 2, 35, 7, 9, 43, 12, 78, 15, 80, 17, 75, 83, 20, 99, 55, 57, 92, 93, 63]
        test_id = [87, 71, 35, 3, 16, 75, 30, 70, 56, 24, 43, 44, 7, 39, 52, 42, 78, 97, 107]
        train_id = [x for x in imgs.keys() if x not in test_id and x != 107]

        # split for images
        for id in imgs:
            if id in train_id:
                train_img[id] = imgs[id]
            else:
                test_img[id] = imgs[id]

        # split for gelsight
        for id in gels:
            if id in train_id:
                train_gel[id] = gels[id]
            else:
                '''
                test_gel[id] = {}
                for date in gels[id]:
                    fn = gels[id][date][-1]
                    id_, type, number, date = re.findall('F(\d*) ?_(.*)_(\d*)_(.*).jpg', fn)[0]
                    if type == 'fold':
                        test_gel[id][date] = gels[id][date]
                '''

                test_gel[id] = gels[id]


        tmp_train_gel = {}
        tmp_test_gel = {}
        # split for gelsight
        for id in train_gel:
            # if id not in train_id:continue
            train_number = int(len(train_gel[id]) * 0.8)

            tmp_train_gel[id] = {}
            tmp_test_gel[id] = {}

            i = 0
            for date in train_gel[id]:
                i += 1
                if i <= train_number:
                    tmp_train_gel[id][date] = train_gel[id][date]
                else:
                    tmp_test_gel[id][date] = train_gel[id][date]
        # train_gel = tmp_train_gel
        # train_gel = tmp_test_gel

        return (train_img, test_img, train_gel, test_gel)



if __name__ == '__main__':
    split = Split()
    ## test load kinect
    # print split.load_kinect('../../data/canon')
    train_img, test_img, train_gel, test_gel = split.split_all('kinect_depth_new')


    print train_img[train_img.keys()[0]]
    print train_gel[train_gel.keys()[0]]
    # print train_gel
    # print train_gel.keys()

    exit(0)
    for i in range(100):
        if i in train_gel:
            print len(train_img[i])

    print '*'*100
    for i in range(100,120):
        if i in train_gel:
            print len(train_img[i])


    ## test load gelsight
    # print getsizeof(split.load_gelsight())

    ## test split by home situation
    # train_img, test_img, train_gel, test_gel = split.split_shop('kinect_depth')
    # pprint(train_gel)
