import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import fastai

from engine.trainer import Trainer
from engine.data_loader import DataLoader

from engine.models.awkward import awkwardClassifier

def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--min_vocab_freq', type=int, default=1)
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--n_epochs', type=int, default=80)

    p.add_argument('--word_vec_size', type=int, default= 512)
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=64)
    
    p.add_argument('--awkward', action='store_true')
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--nhid', type=int, default=768)
    p.add_argument('--nlayers', type=int, default=2) # transformer layers
    p.add_argument('--cos', type=int, default=2) # transformer layers

    config = p.parse_args()

    return config


def main(config):
    loaders = DataLoader(train_fn=config.train_fn,
                         batch_size=config.batch_size,
                         min_freq=config.min_vocab_freq,
                         max_vocab=config.max_vocab_size,
                         device=config.gpu_id
                         )

    print(
        '|train| =', len(loaders.train_loader.dataset),
        '|valid| =', len(loaders.valid_loader.dataset),
    )
    
    vocab1_size = len(loaders.text1.vocab)
    vocab2_size = len(loaders.text2.vocab)
    n_classes = len(loaders.label.vocab)
    
    print(loaders.label.vocab)
    
    print('|vocab1| =', vocab1_size, '|vocab2| =', vocab2_size, '|classes| =', n_classes)

    if config.awkward:
        model = awkwardClassifier(
            ntoken = vocab1_size,
            ntoken2 = vocab2_size,
            ninp = config.word_vec_size,
            n_classes=n_classes,
            nhead = config.nhead, 
            nhid = config.nhid, 
            nlayers = config.nlayers,
            cos = config.cos,
            dropout=config.dropout,
        )
    
        optimizer = optim.Adam(model.parameters())
        #crit = nn.BCEWithLogitsLoss()
        #crit = nn.MSELoss() # MSELossFlat loss 사용시 !pip install --user fastai 설치 필요
               
        
        crit = nn.NLLLoss() # nn.LogSoftmax(dim=-1)와 조합이 되는 loss function()
        
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)
            
        awkward_trainer = Trainer(config)
        awkward_model = awkward_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader
           ,loaders.valid_loader
        )
    
#     if config.rnn is False and config.cnn is False:
#         raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')


    torch.save({
        'awkward': awkward_model.state_dict() if config.awkward else None,
        'config': config,
        'vocab1': loaders.text1.vocab,
        'vocab2': loaders.text2.vocab,
        'classes': loaders.label.vocab,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)

    