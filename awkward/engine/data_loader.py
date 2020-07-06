from torchtext import data

from torchtext import data
from konlpy.tag import Okt 
okt = Okt()

class DataLoader(object):
    '''
    Data loader class to load text file using torchtext library.
    '''

    def __init__(
        self, train_fn,
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        use_eos=False,
        shuffle=True
    ):
        '''
        DataLoader initialization.
        :param train_fn: Train-set filename
        :param batch_size: Batchify data fot certain batch size.
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: Maximum vocabulary size
        :param min_freq: Minimum frequency for loaded word.
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super().__init__()

        # Define field of the input file.
        # The input file consists of two fields.
        self.text1 = data.Field(sequential = True,
                              tokenize=okt.morphs, # data가 이미 tokenizing 되어 있음.
                              use_vocab = True,
                              batch_first = True,
                              include_lengths = False,
                              eos_token='<EOS>' if use_eos else None,
                              pad_first=True
                              )
        
        self.text2 = data.Field(sequential = True,
                              tokenize=okt.morphs,
                              use_vocab = True,
                              batch_first = True,
                              include_lengths = False,
                              eos_token='<EOS>' if use_eos else None,
                              pad_first=True
                              )
        
        # sequential : If False,no tokenization is applied
        self.label = data.Field(sequential = False,
                             preprocessing = lambda x: int(x),
                                 use_vocab = True,
                                init_token = None,
                                 unk_token = None
                               )

        # Those defined two columns will be delimited by TAB.
        # Thus, we use TabularDataset to load two columns in the input file.
        # We would have two separate input file: train_fn, valid_fn
        # Files consist of two columns: label field and text field.
        train, valid = data.TabularDataset(
            path=train_fn,
            format='tsv', 
            fields=[
                ('text1', self.text1),
                ('text2', self.text2),
                ('label', self.label),
            ],
            skip_header = True
        ).split(split_ratio=(1 - valid_ratio))

        # Those loaded dataset would be feeded into each iterator:
        # train iterator and valid iterator.
        # We sort input sentences by length, to group similar lengths.
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            sort_key=lambda x: len(x.text1),
            sort_within_batch=True,
        )

        # At last, we make a vocabulary for label and text field.
        # It is making mapping table between words and indice.
        self.label.build_vocab(train)
        self.text1.build_vocab(train, max_size=max_vocab, min_freq=min_freq)
        self.text2.build_vocab(train, max_size=max_vocab, min_freq=min_freq)