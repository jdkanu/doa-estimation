# -*- coding: utf-8 -*-

"""
Evaluation script for DOA estimation
"""

# interfacing with matlab on my machine required unsetting this env variable
import os
os.unsetenv('MKL_NUM_THREADS')

import argparse
import torch
from model import CRNN, ConvNet, LSTM_FIRST, LSTM_FULL, LSTM_LAST
from doa_math import DoaClasses,to_cartesian,to_class
from doa_stats import ToleranceScore
from graphics import plot_curve
from config import Config,Dropouts

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_tolerance_score(config):
  model = config.model
  _,_,test_loader = config.get_loaders()
  doa_classes = config.doa_classes
  use_all_lstm_frames = config.use_all_lstm_frames()

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
        if use_all_lstm_frames:
          Yhat = torch.sum(Yhat, 2)
          Y = Y[:, 0] # Can take the 0th b/c labels identical for frames
        _, Yhat = torch.max(Yhat, 1)
        Yhat = [to_cartesian(x,doa_classes) for x in Yhat]
      else:
        if use_all_lstm_frames:
          Yhat = torch.sum(Yhat, 1)/25
          Y = Y[:, 0]
      tolerance_score.update(Yhat,Y)
      
  return tolerance_score

def plot_SNR_curve(curve):
  plot_curve(curve)

def compute_stats(config):
  tolerance_score = compute_tolerance_score(config)

def inference_model(network,lstm_out,out_format,model_path):
  doa_classes = DoaClasses()
  if out_format == "cartesian":
    out_dim = 3
  elif out_format == "class":
    out_dim = len(doa_classes.classes)
  
  if network == "CNN":
    model = ConvNet(device, Dropouts(0,0,0), out_dim, doa_classes)
  elif network == "CRNN":
    model = CRNN(device, Dropouts(0,0,0), out_dim, doa_classes, lstm_out)
  model.load_state_dict(torch.load(model_path,map_location=device))
  model.eval()
  model.to(device)
  
  return model,doa_classes
  

def test_run(network,lstm_out,out_format,data_dirs,log_dirs,model_path):
  model,doa_classes = inference_model(network, lstm_out, out_format,model_path)
  data_dirs = data_dirs.split(',')
  log_dirs = log_dirs.split(',')
  tolerance_scores = []
  for data_dir,log_dir in zip(data_dirs,log_dirs):
    config = Config(data_folder=data_dir,\
                    model=model,\
                    doa_classes=doa_classes,\
                    lstm_output=lstm_out,\
                    results_dir=log_dir,\
                    thresholds=[5,10,15],\
                    batch_size=32)
    score = compute_tolerance_score(config)
    tolerance_scores.append(score)
    
  return tolerance_scores

def main():
  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

  parser = argparse.ArgumentParser(prog='evaluate',\
                description="""Script to evaluate the DOA estimation system""")
  parser.add_argument("--data_dirs", "-d", default="../data", required=True,\
                help="Directory where data and labels are", type=str)
  parser.add_argument("--log_dirs", "-log", default=".", required=True,\
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
  
  test_run(args.network,args.lstm_out,args.out_format,\
      args.data_dirs,args.log_dirs,args.model_path)

if __name__ == "__main__":
  main()
  
