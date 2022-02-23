from ast import arg
from tkinter import HIDDEN
from models.model import BiLSTM

from models.model_w2v import BiLSTM_W2V
from models.NER import NER
from build_dataloader import corpus
import argparse
import torch
from torch import nn
from torch.optim import Adam

parser = argparse.ArgumentParser()

parser.add_argument('--emd', type=int, default=256)

parser.add_argument('--hidden', type=int, default=64)

parser.add_argument('--lstm', type=int, default=2)

parser.add_argument('--em_drop', type=float, default=0.25)

parser.add_argument('--lstm_drop', type=float, default=0.25)

parser.add_argument('--fc_drop', type=float, default=0.25)

parser.add_argument('--epoch', type=int, default=10)

parser.add_argument('--w2v', type=int, default=0)




args = parser.parse_args()


EMBEDDING_DIM = args.emd
HIDDEN_DIM = args.hidden
LSTM_LAYER = args.lstm
EMB_DROPOUT = args.em_drop
LSTM_DROPOUT = args.lstm_drop
FC_DROPOUT = args.fc_drop
EPOCH = args.epoch
W2V = args.w2v


MODEL=None


if W2V==0:

    bilstm = BiLSTM(
        input_dim=len(corpus.word_field.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(corpus.tag_field.vocab),
        lstm_layers=LSTM_LAYER,
        emb_dropout=EMB_DROPOUT,
        lstm_dropout=LSTM_DROPOUT,
        fc_dropout=FC_DROPOUT,
        word_pad_idx=corpus.word_pad_idx,
    )
    bilstm.init_weights()
    bilstm.init_embeddings(word_pad_idx=corpus.word_pad_idx)
    
    MODEL = bilstm

    print(f"The bilstm model has {MODEL.count_parameters():,} trainable parameters.")
    print(MODEL)   

else:

    bilstm = BiLSTM_W2V(
        input_dim=len(corpus.word_field.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(corpus.tag_field.vocab),
        lstm_layers=LSTM_LAYER,
        emb_dropout=EMB_DROPOUT,
        lstm_dropout=LSTM_DROPOUT,
        fc_dropout=FC_DROPOUT,
        word_pad_idx=corpus.word_pad_idx,
    )

    bilstm.init_weights()

    bilstm.init_embeddings(
        word_pad_idx=corpus.word_pad_idx,
        pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
        freeze=True)

    MODEL = bilstm

    print(f"The Wrod2Vec bilstm model has {MODEL.count_parameters():,} trainable parameters.")
    print(MODEL)


extraction_model = NER(
  model=MODEL,
  data=corpus,
  optimizer_cls=Adam,
  loss_fn_cls=nn.CrossEntropyLoss
)

def main():
    extraction_model.train(EPOCH)

if __name__=='__main__':
    main()

