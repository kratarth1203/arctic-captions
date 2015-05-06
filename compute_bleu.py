import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import argparse

import numpy
import cPickle as pkl

import math
from nltk.align.bleu import BLEU

from capgen import get_dataset

"""
Following two functions are lifted from 
https://github.com/karpathy/neuraltalk/blob/master/eval_sentence_predictions.py
"""
def BLEUscore(candidate, references, weights):
    p_ns = [BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1)]
    if all([x > 0 for x in p_ns]):
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns))
        bp = 1 #BLEU.brevity_penalty(candidate, references)
        return bp * math.exp(s)
    else: # this is bad
        return 0

def evalCandidate(candidate, references):
    """ 
    candidate is a single list of words, references is a list of lists of words
    written by humans.
    """
    b1 = BLEUscore(candidate, references, [1.0])
    b2 = BLEUscore(candidate, references, [0.5, 0.5])
    b3 = BLEUscore(candidate, references, [1/3.0, 1/3.0, 1/3.0])
    return [b1,b2,b3]


def main(dataset, basename):

    load_data, prepare_data = get_dataset(dataset)

    train, valid, test, worddict = load_data()

    def _compute_bleu(caps, refs):
        ridx = 0
        all_bleu_scores = []
        for c in caps:
            rr = []
            for ii in xrange(5): # five groundtruth captions
                rr.append([w for w in refs[ridx][0].strip().split()])
                ridx += 1
            all_bleu_scores.append(evalCandidate(c.strip().split(), rr))
        bleu_averages = [sum(x[i] for x in all_bleu_scores)*1.0/len(all_bleu_scores) for i in xrange(3)]
        return bleu_averages

    print 'Computing BLEU scores for Development Set...',
    with open(basename+'.dev.txt', 'r') as f:
        dev_bleu = _compute_bleu(f, valid[0])
    print 'Done'
    print 'Development BLEU-1:', dev_bleu[0], ' BLEU-2:', dev_bleu[1], ' BLEU-3:', dev_bleu[2]

    print 'Computing BLEU scores for Test Set...',
    with open(basename+'.test.txt', 'r') as f:
        test_bleu = _compute_bleu(f, test[0])
    print 'Done'
    print 'Test BLEU-1:', test_bleu[0], ' BLEU-2:', test_bleu[1], ' BLEU-3:', test_bleu[2]

if __name__ == "__main__":
    #print 'DO NOT USE THIS SCRIPT! USE multi-bleu.perl'
    #exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='flickr8k')
    parser.add_argument('basename', type=str)

    args = parser.parse_args()

    main(args.d, args.basename)

