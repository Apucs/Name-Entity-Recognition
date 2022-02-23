import matplotlib
import torch
from torch import nn
from torch.optim import Adam

from tqdm import tqdm
from spacy.lang.en import English
import time
import pandas as pd
import matplotlib.pyplot as plt

class NER(object):

  def __init__(self, model, data, optimizer_cls, loss_fn_cls):
    self.model = model
    self.data = data
    self.optimizer = optimizer_cls(model.parameters())
    self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)

  def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

  def plot_graph(self,X_acc, X_loss, epoch, type):
    if type=="train":
      plt.plot(epoch, X_acc, label = "accuracy")
      plt.plot(epoch, X_loss, label = "loss")
      plt.xlabel('No of epoch(train)')
      plt.legend()
      plt.show()

    elif type=="val":
      plt.plot(epoch, X_acc, label = "accuracy")
      plt.plot(epoch, X_loss, label = "loss")
      plt.xlabel('No of epoch(validation)')
      plt.legend()
      plt.show()




  def accuracy(self, preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  #prepare masking for paddings
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

  def epoch(self):
      epoch_loss = 0
      epoch_acc = 0
      self.model.train()
      for batch in tqdm(self.data.train_iter):
        # text = [sent len, batch size]
        text = batch.word
        # tags = [sent len, batch size]
        true_tags = batch.tag
        self.optimizer.zero_grad()
        pred_tags = self.model(text)
        # flatten pred_tags to [sent len, batch size, output dim]
        pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
        # flatten true_tags to [sent len * batch size]
        true_tags = true_tags.view(-1)


        if true_tags.shape[0]!=pred_tags.shape[0]:
          continue
        
        else:
          batch_loss = self.loss_fn(pred_tags, true_tags)
          batch_acc = self.accuracy(pred_tags, true_tags)
          batch_loss.backward()
          self.optimizer.step()
          epoch_loss += batch_loss.item()
          epoch_acc += batch_acc.item()
      return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

  def evaluate(self, iterator):
      epoch_loss = 0
      epoch_acc = 0
      self.model.eval()
      with torch.no_grad():
          for batch in iterator:
              text = batch.word
              true_tags = batch.tag
              pred_tags = self.model(text)
              pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
              true_tags = true_tags.view(-1)
              batch_loss = self.loss_fn(pred_tags, true_tags)
              batch_acc = self.accuracy(pred_tags, true_tags)
              epoch_loss += batch_loss.item()
              epoch_acc += batch_acc.item()
      return epoch_loss / len(iterator), epoch_acc / len(iterator)

  # main training sequence
  def train(self, n_epochs):

    best_acc = 0
    summary = []
    t_loss=[]
    t_acc = []
    v_loss = []
    v_acc = []
    ep = []

    for i, epoch in enumerate(tqdm(range(n_epochs))):
        start_time = time.time()
        train_loss, train_acc = self.epoch()
        end_time = time.time()
        epoch_mins, epoch_secs = NER.epoch_time(start_time, end_time)
        print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")

        val_loss, val_acc = self.evaluate(self.data.val_iter)
        print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")

        test_loss, test_acc = self.evaluate(self.data.test_iter)
        print(f"\tTest Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")


        summary.append([train_acc*100, val_acc*100, test_acc*100, train_loss, val_loss, test_loss])
        t_acc.append(train_acc)
        t_loss.append(train_loss)
        v_acc.append(val_acc)
        v_loss.append(val_loss)
        ep.append(i+1)

        if test_acc>best_acc:
          best_acc = test_acc

          model_scripted = torch.jit.script(self.model) # Export to TorchScript
          model_scripted.save('checkpoint/model_scripted.pt') # Save

    print(t_acc, t_loss, v_acc, v_loss)
    print(ep)
    print(summary)


    df = pd.DataFrame(summary, columns = ["train accuracy", "val accuracy", "test accuracy", "train loss", "val loss", "test loss"])

    df.to_csv('output/result.csv', index=False)
    print(df.head(5))

    self.plot_graph(t_acc, t_loss, ep, "train")
    self.plot_graph(v_acc, v_loss, ep, "val")

    model_last = torch.jit.script(self.model) # Export to TorchScript
    model_last.save('checkpoint/model_last.pt')


    torch.save(self.model.state_dict(), "checkpoint/checkpoint.pth")


    torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, "checkpoint/checkpoint_saved.pth")

  


  