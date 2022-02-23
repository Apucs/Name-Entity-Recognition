import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset


import gensim
from collections import Counter
from torchtext.vocab import Vocab


def word2vec():
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    vector_size = w2v_model.vector_size

    #print(w2v_model.vector_size)

    word_freq = {word: w2v_model.wv.vocab[word].count for word in w2v_model.wv.vocab}
    word_counter = Counter(word_freq)

    #print(len(word_counter))

    vocab = Vocab(word_counter, min_freq=3)

    #print(len(vocab))

    return w2v_model, vector_size, word_counter, vocab


class Corpus(object):

  def __init__(self, input_folder, train_data, val_data, test_data, min_word_freq, batch_size, w2v=0):
    
    #Listing two fields
    self.word_field = Field(lower=True)
    self.tag_field = Field(unk_token=None)
    
    # creating dataset from given files using torchtext "SequenceTaggingDataset"
    self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
        
        path=input_folder,
        train=train_data,
        validation=val_data,
        test=test_data,
        fields=(("word", self.word_field), ("tag", self.tag_field))
    
    )
    if w2v==1:
        self.wv_model, self.embedding_dim, word_counter, self.word_field.vocab = word2vec()

        print("Word2Vec Embedding dim:", self.embedding_dim)

        vectors = []
        for word, idx in self.word_field.vocab.stoi.items():
            if word in self.wv_model.wv.vocab.keys():
                vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
            else:
                vectors.append(torch.zeros(self.embedding_dim))
        self.word_field.vocab.set_vectors(
            stoi=self.word_field.vocab.stoi,
            vectors=vectors,
            dim=self.embedding_dim
        )
        print("vector size:",  len(vectors))
    else:
        self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)

    self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.tag_field.build_vocab(self.train_dataset.tag)

    #creating iterator for each batch
    self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
        datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
        batch_size=batch_size
    )
    
    self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
    self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
