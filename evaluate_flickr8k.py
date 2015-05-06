import numpy
import sys
import argparse

from capgen import train

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    trainerr, validerr, testerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        ctx_dim=params['ctx_dim'][0],
                                        dim=params['dim'][0],
                                        n_layers_att=params['n-layers-att'][0],
                                        n_layers_out=params['n-layers-out'][0],
                                        n_layers_lstm=params['n-layers-lstm'][0],
                                        n_layers_init=params['n-layers-init'][0],
                                        n_words=params['n-words'][0],
                                        lstm_encoder=params['lstm-encoder'][0],
                                        decay_c=params['decay-c'][0],
                                        alpha_c=params['alpha-c'][0],
                                        prev2out=True,
                                        ctx2out=True,
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        selector=params['selector'][0],
                                        maxlen=100,
                                        batch_size=64,
                                        valid_batch_size=64,
                                        validFreq=-1,
                                        dispFreq=1,
                                        saveFreq=-1,
                                        sampleFreq=250,
                                        dataset='flickr8k', 
                                        use_dropout=params['use-dropout'][0],
                                        use_dropout_lstm=params['use-dropout-lstm'][0],
                                        )
    print "Final validation error {}".format(-validerr.mean())
    return validerr

if __name__ == '__main__':
    options = {
        'model': ['model_lstm.npz'],
        'ctx_dim': [512],
        'dim_word': [500],
        'dim': [1200],
        'n-layers-att': [2],
        'n-layers-out': [1],
        'n-layers-lstm': [1],
        'n-layers-init': [1],
        'lstm-encoder': [False],
        'n-words': [23463], #[8184]
        'optimizer': ['rmsprop'],
        'decay-c': [0.], 
        'alpha-c': [0.25], 
        'use-dropout': [True],
        'use-dropout-lstm': [False],
        'learning-rate': [0.01],
        'selector': [True],
        'reload': [False]}

    if len(sys.argv) > 1:
        options.update(eval("{%s}"%sys.argv[1]))

    main(0, options)

