from data import config
from train import bilstm
from models.NER import NER
from build_dataloader import corpus, test_iter

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


checkpoint_path = config.CHECKPOINT3

def accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != corpus.tag_pad_idx).nonzero()  # prepare masking for paddings
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def inference(checkpoint_path, iterator):
    epoch_loss = 0
    epoch_acc = 0


    model = torch.jit.load(checkpoint_path)


    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in iterator:

            text = batch.word
            
            true_tags = batch.tag

            pred_tags = model(text)

            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            
            top_tags = pred_tags.argmax(-1)

            predicted_tags = [corpus.tag_field.vocab.itos[t.item()] for t in top_tags]
            
            true_tags = true_tags.view(-1)
            
            batch_loss = loss_fn(pred_tags, true_tags)
            batch_acc = accuracy(pred_tags, true_tags)
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    test_loss, test_acc = inference(checkpoint_path,  test_iter)
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")

if __name__=='__main__':
    main()