
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class awkwardClassifier(nn.Module):

    def __init__(self, 
                 ntoken, # 단어수
                 ntoken2,
                 ninp,   # 임베딩 차원
                 n_classes,
                 nhead, 
                 nhid, 
                 nlayers,
                 use_batch_norm=False,
                 dropout=0.5):

        super(awkwardClassifier, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        
        self.ninp = ninp
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=1)
        self.encoder2 = nn.Embedding(ntoken2, ninp, padding_idx=1)
        
        ## 첫번째 문장
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.decoder = nn.Linear(ninp, ntoken)

        ## 두번째 문장
        self.pos_encoder2 = PositionalEncoding(ninp, dropout)
        encoder_layers2 = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, nlayers)
        #self.decoder2 = nn.Linear(ninp, ntoken)
        
        ## 문장간 유사도
        #self.con_li = nn.Linear(ninp, n_classes)
        self.con_li = nn.Linear(64, n_classes) # 시퀀스 길이 x 클래스 갯수
        self.activation = nn.LogSoftmax(dim=-1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder2.weight.data.uniform_(-initrange, initrange)
        self.con_li.weight.data.uniform_(-initrange, initrange)

        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)
        
        #self.decoder2.bias.data.zero_()
        #self.decoder2.weight.data.uniform_(-initrange, initrange)
        #self.result.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, src2):
        
        ## 첫번째 문장
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        #output = self.decoder(output)
        
        ## 두번째 문장
        if self.src_mask is None or self.src_mask.size(0) != len(src2):
            device = src2.device
            mask = self._generate_square_subsequent_mask(len(src2)).to(device)
            self.src_mask = mask

        src2 = self.encoder2(src2) * math.sqrt(self.ninp)
        src2 = self.pos_encoder2(src2)
        output2 = self.transformer_encoder2(src2, self.src_mask)
        # output2 = self.decoder2(output2)

        #final_outs = torch.cat([output, output2], dim = 2)
        ## <1차 성공>
        #  final_outs = torch.add(output, output2)
        #  output = self.con_li(final_outs)

        # y = self.activation(output[:, -1])
        ## </1차 성공>
    
        ## <2차 시도>
        #final_outs = torch.exp(-torch.sum(torch.abs(output - output2), dim=2)) # 성공 (7, 64) -> 모두 0 아니면 1로
        final_outs = -torch.sum(torch.abs(output - output2), dim=2) # 성공 (7, 64)
        output = self.con_li(final_outs)
        y = self.activation(output)
        
        return y
        #return torch.argmax(y, dim=1)

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
