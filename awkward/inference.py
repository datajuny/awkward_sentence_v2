
from torchtext import data
from engine.models.awkward import awkwardClassifier
from konlpy.tag import Okt

import torch
import argparse

path = "./"
#model_name = "model_gpu_nsp_cosine_v7.pth"
#model_name = "model_gpu_okt_layer2_seq64_v8.pth"
model_name = "model_gpu_okt_layer2_v9.pth"

use_eos=False
n_classes = 2

saved_data = torch.load(path + model_name, map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu')

config = saved_data['config']
awkward_best = saved_data['awkward']
vocab1 = saved_data['vocab1']
vocab2 = saved_data['vocab2']
classes = saved_data['classes']

vocab1_size = len(vocab1)
vocab2_size = len(vocab2)

okt = Okt()

def infer(x1, x2):
    
    def define_field():

#         return (
#             data.Field(sequential = True,
#                                   use_vocab = True,
#                                   batch_first = True,
#                                   include_lengths = False,
#                                   eos_token='<EOS>' if use_eos else None,
#                                   pad_first=True
#                                   ),
#             data.Field(sequential = True,
#                                   use_vocab = True,
#                                   batch_first = True,
#                                   include_lengths = False,
#                                   eos_token='<EOS>' if use_eos else None,
#                                   pad_first=True
#                                   ),        
#             data.Field(sequential = False,
#                                  preprocessing = lambda x: int(x),
#                                      use_vocab = True,
#                                     init_token = None,
#                                      unk_token = None
#                                   )
#         )

        return (data.Field(sequential = True,
                              tokenize=okt.morphs, # data가 이미 tokenizing 되어 있음.
                              use_vocab = True,
                              batch_first = True,
                              include_lengths = False,
                              eos_token='<EOS>' if use_eos else None,
                              pad_first=True
                              ),
                 data.Field(sequential = True,
                              tokenize=okt.morphs,
                              use_vocab = True,
                              batch_first = True,
                              include_lengths = False,
                              eos_token='<EOS>' if use_eos else None,
                              pad_first=True
                              ),
        # sequential : If False,no tokenization is applied
        data.Field(sequential = False,
                             preprocessing = lambda x: int(x),
                                 use_vocab = True,
                                init_token = None,
                                 unk_token = None
                               )
    )
    

    text1_field, text2_field, label_field = define_field()
    
    text1_field.vocab = vocab1
    text2_field.vocab = vocab2 
    label_field.vocab = classes
    
    lines1 = []
    lines2 = []
    
    lines1 += [x1.strip().split(' ')]
    lines2 += [x2.strip().split(' ')]
    
    with torch.no_grad():
        x1 = text1_field.numericalize(
        text1_field.pad(lines1),
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        )

        x2 = text2_field.numericalize(
        text2_field.pad(lines2),
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        )

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
        
        ensemble = []
        model.load_state_dict(awkward_best)
        ensemble += [model]
        
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)

        model.eval()
        
        y_hat = []
        
        def pad_to_maxseq_to_batch(batch, max_length, device=-1):
    
            if batch.size(1) >= max_length:
                batch = batch[:, :max_length]

            else:
                pad_num = max_length - batch.size(1)
                pad = torch.ones(batch.size(0), pad_num, device=device) # pad 값이 vocab index 1이라는 가정.
                batch = torch.cat([pad.long(), batch], dim=-1)

            return batch
        
        
    x1 = pad_to_maxseq_to_batch(x1, config.max_length, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    x2 = pad_to_maxseq_to_batch(x2, config.max_length, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    print(model(x1, x2)[0])    
    return torch.argmax(model(x1, x2)[0], dim=-1)
