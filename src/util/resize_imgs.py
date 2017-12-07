from preprocess import Split
from pprint import pprint
from scipy.misc import imread, imresize, imsave
import multiprocessing
import numpy as np

def resize_canon(frame):
    return imresize(frame[250:250+3000, 1100:1100+3000], (227, 227))


def resize_depth(frame):
    # img = frame[60:60+227,100:100+227].astype('float32')
    img = frame[69:69+227,137:137+227].astype('float32')
    img = (img - 800) / (1300 - 800) * 255
    img[img<0] = 0
    img[img>255] = 255
    img = np.array(img)
    return img

def resize_gel(frame):
    return imresize(frame[:,120:120+720], (227, 227))

def worker(fn):
    frame = imread(fn)
    # frame = resize_canon(frame)
    frame = resize_depth(frame)
    # frame = resize_gel(frame)
    # resize_fn = '/'.join(fn.split('/')[:-2] + ['resize'] + fn.split('/')[-2:])
    # resize_fn = '/'.join(fn.split('/')[:-2] + ['resize'] + ['image_fold'] + fn.split('/')[-1:])
    resize_fn = '/'.join(fn.split('/')[:-3] + ['resize'] + ['image_flat'] + fn.split('/')[-1:])
    try:
        imsave(resize_fn, frame)
    except Exception, e:
        print e
    print resize_fn

def main():
    split = Split()
    # dn = '../../data/canon'
    #dn = '../../data/Recording_siyuan/kinect_depth/'
    #imgs = split.load_kinect(dn)

    imgs = split.load_gelsight()
    # print imgs

    processes_no = 28
    pool = multiprocessing.Pool(processes=processes_no)

    for id in imgs:
        for date in imgs[id]:
            for fn in imgs[id][date]:
                pool.apply_async(worker, (fn, ))

        #for fn in imgs[id]:
        #    pool.apply_async(worker, (fn, ))
            # print resize_fn

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
