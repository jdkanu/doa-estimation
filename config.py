import torch.nn as nn
from model import CRNN, ConvNet, LSTM_FIRST, LSTM_FULL, LSTM_LAST
from dataset import generate_loaders

class Config():
  def __init__(self, **kwargs):
    self.data_folder = None
    self.num_threads = None
    self.learning_rate = None
    self.batch_size = None
    self.num_epochs = None
    self.test_to_all_ratio = None
    self.results_dir = None
    self.model = None
    self.loss_criterion = None
    self.lstm_output = None
    self.shuffle = None
    for key,value in kwargs.items():
      self.__dict__[key] = value
    print('\n'.join(["{}={}".format(p,self.__dict__[p]) for p in self.__dict__]))
    
    self.train_loader,self.val_loader,self.test_loader = generate_loaders(self)
    
  def get_loaders(self):
    return self.train_loader,self.val_loader,self.test_loader

  def all_lstm_frames(self):
    return self.lstm_output == LSTM_FULL and isinstance(self.model, CRNN)

class Dropouts():
    def __init__(self, input_dropout, conv_dropout, lstm_dropout):
        self.input_dropout = nn.Dropout(input_dropout)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)

