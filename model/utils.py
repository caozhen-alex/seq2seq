import torch
import spacy
import re
import os
import numpy as np

INIT = 1e-2
INF = 1e18

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

url = re.compile('(<url>.*</url>)')
spacy_en = spacy.load('en')


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]


def sequence_mean(sequence, seq_lens, dim):
    # inputs: sequence: Tensor(batch_size, time_step, hidden_size)
    #         seq_lens: list(batch_size,)
    #         dim: int
    # output: seq_mean: Tensor(batch_size, hidden_size)
    assert sequence.size(0) == len(seq_lens)
    seq_sum = torch.sum(sequence, dim=dim, keepdim=False)  # if dim=1, ---> Tensor(batch_size, hidden_size)
    seq_mean = torch.stack([s / l for s, l in zip(seq_sum, seq_lens)], dim=0)
    return seq_mean


def len_mask(seq_lens, max_len):  # for attention calculation
    # inputs: seq_lens: list(batch_size,);
    #         max_len: int(the length of the longest seq in the batch)
    # output: mask: Tensor(batch_size, max_len) with value 1 at seq place and value 0 at other place.
    batch_size = len(seq_lens)
    mask = torch.ByteTensor(batch_size, max_len).cuda()
    mask.fill_(0)  # mask: Tensor(batch_size, max_len) with value 0
    # The previous two line can be replaced by:
    # mask = torch.zeros(batch_size, max_len).byte().cuda()
    for i, l in enumerate(seq_lens):
        mask[i, :l].fill_(1)
    return mask


def cat_mask(seq_lens, cat_size):  # for concatenation
    # inputs: seq_lens: list(batch_size,);
    #         cat_size: Tensor size, list like
    # output: mask: Tensor(cat_size) with value 1 at seq place and value 0 at other place.
    assert len(seq_lens) == cat_size[0]
    mask = torch.zeros(cat_size).cuda()
    for i, l in enumerate(seq_lens):
        mask[i, :l, :].fill_(1)
    return mask


def load_word_embeddings(fname, vocab_size, embed_size, word2index):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.uniform(-0.01, 0.01, [vocab_size, embed_size])
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                word2vec[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    word2vec[word2index[PAD], :] = 0
    return word2vec


def sentence_clip(sentences, lens):
    max_len = max(lens)
    sentences = sentences[:, 0:max_len].contiguous()  # contiguous(): makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch.
    return sentences
