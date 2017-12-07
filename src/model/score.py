from scipy.io import loadmat
import numpy as np
import random

def get_precision(img, gel, img_id, gel_id, img_fns=None, gel_fns=None):
    random.seed(0)
    nb_times = 50

    new_img = []
    new_gel = []
    new_img_id = []
    new_gel_id = []
    new_img_fns = []
    new_gel_fns = []
    for i,id in enumerate(img_id):
        if id != 107:
            new_img.append(img[i])
            new_img_id.append(id)
            new_img_fns.append(img_fns[i])

    for i,id in enumerate(gel_id):
        if id != 107:
            new_gel.append(gel[i])
            new_gel_id.append(id)
            new_gel_fns.append(gel_fns[i])
    gel_fns = new_gel_fns
    img_fns = new_img_fns

    img = np.array(new_img)
    img_id = new_img_id
    gel = np.array(new_gel)
    gel_id = new_gel_id


    nb_img = len(img_id)
    nb_gel = len(gel_id)


    distance_matrix = np.zeros((nb_gel, nb_img))

    for i in range(nb_gel):
        distance_matrix[i,:] = np.sqrt(np.sum(np.square(img - gel[i,:]), axis=1)).T

    img_id_set = list(set(img_id))

    ks = range(1, 4)
    precison = [0 for i in range(4)]


    for i in range(nb_gel):
        for times in range(nb_times):
            choices = []
            correct = random.choice([x for x in range(nb_img) if x != i and img_id[x] == gel_id[i]])
            candidate = [img_id[x] for x in range(nb_img)]
            candidate = list(set(candidate))
            candidate = random.sample(candidate, 9)
            # print candidate
            for c in candidate:
                s = [x for x in range(nb_img) if img_id[x] == c]
                j = random.choice(s)
                choices.append(j)

            a = [(x, img_id[x] == gel_id[i], img_fns[x], distance_matrix[i, x]) for x in [correct] + choices]
            sort_a = sorted(a, key=lambda d:d[-1])
            for k in ks:
                if True in [x[1] for x in sort_a[:k]]:
                    precison[k] += 1

    precison = [np.mean(x) * 1.0 / nb_gel / nb_times for x in precison[1:]]
    
    print precison[0],"\t",precison[2],"\t",

    return [precison[0], precison[2]]


def get_score(img_embeddings, img_fns, gel_embeddings, gel_fns):
    
    nb_img = len(img_embeddings)
    img_embeddings = np.array(img_embeddings)
    dim_embedding = img_embeddings.shape[-1]
    img_embeddings = img_embeddings.reshape((nb_img, dim_embedding))
    img_id = [int(x.split('/')[-1][1:5]) for x in img_fns]

    nb_gel = len(gel_embeddings)
    gel_embeddings = np.array(gel_embeddings)
    gel_embeddings = gel_embeddings.reshape((nb_gel, dim_embedding))
    gel_id = [int(x.split('/')[-1][1:5]) for x in gel_fns]

    # print gel_id.index(35)
    
    'Gel to Color'
    g2c = get_precision(img_embeddings, gel_embeddings, img_id, gel_id, img_fns, gel_fns)
    
    
    'Gel to Depth'
    #g2d = get_precision(depth_embeddings, gel_embeddings, depth_id, gel_id, depth_fns, gel_fns)

    
    'Color to Gel'
    c2g = get_precision(gel_embeddings, img_embeddings, gel_id, img_id, gel_fns, img_fns)
    
    
    'Color to Depth'
    #c2d = get_precision(depth_embeddings, img_embeddings, depth_id, img_id, depth_fns, img_fns)
    
    
    'Depth to Gel'
    #d2g = get_precision(gel_embeddings, depth_embeddings, gel_id, depth_id, gel_fns, depth_fns)
    
    
    'Depth to Color'
    #d2c = get_precision(img_embeddings, depth_embeddings, img_id, depth_id, img_fns, depth_fns)
    
    
    'Color to Color'
    #c2c = get_precision(img_embeddings, img_embeddings, img_id, img_id, img_fns, img_fns)
    
    
    'Depth to Depth'
    #d2d = get_precision(depth_embeddings, depth_embeddings, depth_id, depth_id, depth_fns, depth_fns)
    
    
    'Gel to Gel'
    #g2g = get_precision(gel_embeddings, gel_embeddings, gel_id, gel_id, gel_fns, gel_fns)
    
    
    'Depth to Gel'
    #get_precision(gel_embeddings, img_embeddings, gel_id, img_id)
    #get_precision(gel_embeddings, img_embeddings, gel_id, img_id, gel_fns, img_fns)

    'Depth to Depth'
    #get_precision(img_embeddings, img_embeddings, img_id, img_id)

    'Gel to Gel'
    #get_precision(gel_embeddings, gel_embeddings, gel_id, gel_id)
    return [g2c, c2g]



if __name__ == '__main__':
    fn = '../result/logs/c2d_branch_shop/embedding_test.mat'#multi

    data = loadmat(fn)
    img_fns = data['img_fns']
    img_embeddings = data['img_embeddings']

    gel_fns = data['gel_fns']
    gel_embeddings = data['gel']

    get_score(img_embeddings, img_fns, gel_embeddings, gel_fns)