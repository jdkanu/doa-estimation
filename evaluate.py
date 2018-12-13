# -*- coding: utf-8 -*-
"""
Evaluation script for DOA estimation
"""

import argparse
import torch
from model import CRNN, ConvNet, LSTM_FIRST, LSTM_FULL, LSTM_LAST
from doa_math import DoaClasses,to_cartesian,to_class
from doa_stats import ToleranceScore,SNRCurve

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_tolerance_score(config):
  model = config.model
  _,_,test_loader = config.get_loaders()
  doa_classes = config.doa_classes
  all_lstm_frames = config.all_lstm_frames()

  tolerance_score = ToleranceScore(config.thresholds,doa_classes)
  
  CC = CX = XC = XX = 0
  with torch.no_grad():
    total = 0
    for i,(X,Y) in enumerate(test_loader):
      total += len(Y)
      X = X.float().to(device)
      Y = Y.to(device)
      Yhat = model(X)
      if doa_classes:
        if all_lstm_frames:
          Yhat = torch.sum(Yhat, 2)
          Y = Y[:, 0] # Can take the 0th b/c labels identical for frames
        _, Yhat = torch.max(Yhat, 1)
        Yhat = [to_cartesian(x,doa_classes) for x in Yhat]
      else:
        if all_lstm_frames:
          Yhat = torch.sum(Yhat, 1)/25
          Y = Y[:, 0]
      tolerance_score.update(Yhat,Y)

  return tolerance_score

def compute_SNR_curve(config):
  SNR_curve = None
  
  return SNR_curve

def compute_stats(config):
  tolerance_score = compute_tolerance_score(config)
  SNR_curve = compute_SNR_curve(config)

def inference_model(network,lstmout,out_format):
  if out_format == "cartesian":
    out_dim = 3
  elif out_format == "class":
    out_dim = len(doa_classes.classes)
  
  if network == "CNN":
    model = ConvNet(device, Dropouts(0,0,0), out_dim, doa_classes)
  elif network == "CRNN":
    model = CRNN(device, Dropouts(0,0,0), out_dim, doa_classes, lstmout)
  model.load_state_dict(torch.load(args.modelpath))
  model.eval()
  model.to(device)
  
  doa_classes = DoaClasses()
  
  return model,doa_classes

if __name__ == "__main__":
  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

  parser = argparse.ArgumentParser(prog='evaluate',\
                description="""Script to evaluate the DOA estimation system""")
  parser.add_argument("--data_dir", "-d", default="data", required=True,\
                help="Directory where data and labels are")
  parser.add_argument("--log_path", "-log", default=".", required=True,\
                help="Path to log results", type=str)
  parser.add_argument("--model_path", "-m", required=True,\
                help="Path to saved model")
  parser.add_argument("--network", "-n", required=True,\
                choices=["CNN", "CRNN"],\
                help="Specify network type", type=str)
  parser.add_argument("--lstm_out", "-lo", default=LSTM_FULL, required=True,\
                choices=[LSTM_FULL, LSTM_FIRST, LSTM_LAST],\
                help="Choose which LSTM output the model uses", type=str)
  parser.add_argument("--out_format", "-of", type=str,\
                choices=["reg", "class"], required=True,\
                help="Choose output format")
  args = parser.parse_args()

  model,doa_classes = inference_model(args.network, args.lstm_out, args.out_format)
  config = Config(data_folder=args.data_dir,\
                  model=model,\
                  doa_classes=doa_classes,\
                  lstm_output=args.lstmout,\
                  thresholds=[5,10,15])
  compute_stats(config)

