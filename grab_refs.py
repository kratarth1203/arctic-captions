import cPickle as pkl

import re

import numpy
from scipy.sparse import csr_matrix

from collections import OrderedDict

base_path='/data/lisatmp2/xukelvin/flickr30k/'
split_path='/data/lisatmp3/chokyun/flickr30k/'

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

    # load the splits
    def _load_split(name):
        split = []
        with open(split_path+name, 'r') as f:
            for idx, line in enumerate(f):
                ''' Flickr8k
                split.append(line.strip().split('_')[0])
                '''
                split.append(line.strip().split('.')[0])
        return split
                
    ''' Flickr8k
    train_f = _load_split('flickr8k_splits/Flickr_8k.trainImages.txt')
    test_f = _load_split('flickr8k_splits/Flickr_8k.testImages.txt')
    dev_f = _load_split('flickr8k_splits/Flickr_8k.devImages.txt')
    '''
    train_f = _load_split('flickr30k_splits/flickr30k_train.txt')
    test_f = _load_split('flickr30k_splits/flickr30k_test.txt')
    dev_f = _load_split('flickr30k_splits/flickr30k_val.txt')

    def _generate_caps(flist, name):
        data_caps = list()
        for idx in xrange(5):
            data_caps.append(list())

        for fname in flist:
            for cidx, cc in enumerate(captions[fname]):
                data_caps[cidx].append(cc)

        for idx in xrange(5):
            with open('%s%d'%(name,idx), 'w') as f:
                print >>f, "\n".join(data_caps[idx])

        return 

    print 'Processing Train...',
    _generate_caps(train_f, 'flicker_30k_referece.train')
    print 'Done'

    print 'Processing Test...',
    _generate_caps(test_f, 'flicker_30k_referece.test')
    print 'Done'

    print 'Processing Dev...',
    _generate_caps(dev_f, 'flicker_30k_referece.dev')
    print 'Done'

if __name__ == '__main__':
    main()

