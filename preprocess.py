import cPickle as pkl

import re

import numpy
from scipy.sparse import csr_matrix

from collections import OrderedDict

base_path='/data/lisatmp2/xukelvin/flickr30k/'

def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])


def main():

    # load the file names
    fdict = OrderedDict()
    ifdict = OrderedDict()
    with open(base_path+'imagelist.txt', 'r') as f:
        for idx, line in enumerate(f):
            line = re.sub(r'\/.*\/','',line).strip()
            line = line.split('.')[0]
            fdict[line] = idx
            ifdict[idx] = line
            print '', idx, ':', line

    # load captions
    captions = OrderedDict()
    with open(base_path+'results_20130124.token', 'r') as f:
        for idx, line in enumerate(f):
            [fname, cap] = line.strip().split('\t')
            [fname, cid] = fname.split('#')
            fname = fname.split('.')[0]
            if fname in captions:
                captions[fname].append(cap)
            else:
                captions[fname] = [cap]

    # build dictionary
    print 'Building dicionary...',
    wordcount = OrderedDict()
    for kk, vv in captions.iteritems():
        for cc in vv:
            words = cc.split()
            for w in words:
                if w not in wordcount:
                    wordcount[w] = 0
                wordcount[w] += 1
    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2 # 0: <eos>, 1: <unk>

    with open('dictionary.pkl', 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)
    print 'Done'


    # load the splits
    def _load_split(name):
        split = []
        with open(base_path+name, 'r') as f:
            for idx, line in enumerate(f):
                split.append(line.strip().split('_')[0])
        return split
                
    train_f = _load_split('flkr8k_splits/Flickr_8k.trainImages.txt')
    test_f = _load_split('flkr8k_splits/Flickr_8k.testImages.txt')
    dev_f = _load_split('flkr8k_splits/Flickr_8k.devImages.txt')

    # load features
    ## final feature map
    features_sp = load_sparse_csr(base_path+'align/features_hidden5_4_conv.npz')
    
    def _build_data(flist):
        data_img = [None] * len(flist)
        data_cap = []
        for idx, fname in enumerate(flist):
            #feature = numpy.array(features_sp[fdict[fname],:].todense()).reshape([14,14,512])
            # save a sparse matrix
            feature = features_sp[fdict[fname],:]
            data_img[idx] = feature
            for cc in captions[fname]:
                data_cap.append((cc, idx))

        return data_cap, data_img

    print 'Processing Train...',
    data_cap, data_img = _build_data(train_f)
    with open('flicker_8k_align.train.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
    print 'Done'

    print 'Processing Test...',
    data_cap, data_img = _build_data(test_f)
    with open('flicker_8k_align.test.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
    print 'Done'

    print 'Processing Dev...',
    data_cap, data_img = _build_data(dev_f)
    with open('flicker_8k_align.dev.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
    print 'Done'



if __name__ == '__main__':
    main()
