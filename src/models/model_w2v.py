import torch
from torch import nn

class BiLSTM_W2V(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, lstm_layers,
                emb_dropout, lstm_dropout, fc_dropout, word_pad_idx):
        super().__init__()
        
        
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim, 
            embedding_dim=embedding_dim, 
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout
        )

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  

    def forward(self, sentence):

        
        embedding_out = self.emb_dropout(self.embedding(sentence))    #output:(sen_length*batch_size*embedding_dim)

        lstm_out, _ = self.lstm(embedding_out)  #output: (sen_length*batch_size*(hidden_dim * 2))

        ner_out = self.fc(self.fc_dropout(lstm_out))  #output: (sen_length*batch_size*output_dim)
        return ner_out

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)



    def init_embeddings(self, word_pad_idx, pretrained=None, freeze=True):
        # initialize embedding for padding as zero
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(pretrained),
                padding_idx=word_pad_idx,
                freeze=freeze
            )


    def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)