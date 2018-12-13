from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

import torch
import torch.nn as nn
from train import doa_train
from config import TrainConfig, Dropouts
import argparse
import os
from model import CRNN, ConvNet
import time

from doa_math import DoaClasses

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ugly way of defining fixed parameters
savedir = '/playpen/zytang/hypertune'
inputdir = '/playpen/zytang/Ambisonic_houses_features'
batch_size = 512
modelname = 'CRNN'
dropout = 0
epochs = 30
outputformulation = "Class"
lstmout = 'Full'

def black_box_function(lr_pow):
    learning_rate = 10.0 ** lr_pow
    results_dir = os.path.join(savedir,
                               "results" + '_{}'.format(modelname) + '_{}'.format(args.outputformulation) + '_lr{}'.format(learning_rate) + '_bs{}'.format(
                                   batch_size) + '_drop{}'.format(dropout))
    print('writing results to {}'.format(results_dir))

    dropouts = Dropouts(dropout, dropout, dropout)
    doa_classes = None
    if outputformulation == "Reg":
        loss = nn.MSELoss(reduction='sum')
        output_dimension = 3
    elif outputformulation == "Class":
        loss = nn.CrossEntropyLoss()
        doa_classes = DoaClasses()
        output_dimension = len(doa_classes.classes)

    if modelname == "CNN":
        model_choice = ConvNet(device, dropouts, output_dimension, doa_classes).to(device)
    elif modelname == "CRNN":
        model_choice = CRNN(device, dropouts, output_dimension, doa_classes, lstmout).to(device)

    config = TrainConfig() \
        .set_data_folder(inputdir) \
        .set_learning_rate(learning_rate) \
        .set_batch_size(batch_size) \
        .set_num_epochs(epochs) \
        .set_test_to_all_ratio(0.1) \
        .set_results_dir(results_dir) \
        .set_model(model_choice) \
        .set_loss_criterion(loss) \
        .set_doa_classes(doa_classes) \
        .set_lstm_output(lstmout)
    # negative sign for minimization
    return -doa_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='hypertune',
                                     description="""Script to tune hyperparameters for deep learning""")
    parser.add_argument("--input", "-i", default="data", help="Directory where data and labels are", type=str)
    parser.add_argument("--logdir", "-l", default=None, help="Directory to write logfiles", type=str)
    parser.add_argument("--batchsize", "-b", type=int, default=None, help="Choose a batchsize, default to sweep")
    parser.add_argument("--probe", "-p", type=float, default=None, help="Choose a probe to start with")
    parser.add_argument("--outputformulation", "-of", type=str, choices=["Reg", "Class"], required=True, help="Choose output formulation")
    parser.add_argument("--model", "-m", type=str, choices=["CNN", "CRNN"], required=True, help="Choose network model")

    args = parser.parse_args()
    
    global inputdir
    inputdir = args.input
    global batch_size
    if args.batchsize:
        batch_size = args.batchsize
    global modelname
    if args.model:
        modelname = args.model
    global outputformulation
    outputformulation = args.outputformulation
    global savedir
    if args.logdir:
        savedir = args.logdir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    logpath = os.path.join(savedir, 'logs_bs{}_{}_{}.json'.format(batch_size, modelname, outputformulation))
    print('writing log file to {}'.format(logpath))

    # Bounded region of parameter space
    pbounds = {'lr_pow': (-10, -1)}


    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
    )

    if args.probe:
        optimizer.probe(
            params={"lr_pow": args.probe},
            lazy=True,
        )
    else:
        optimizer.probe(
            params={"lr_pow": -2},
            lazy=True,
        )
        optimizer.probe(
            params={"lr_pow": -4},
            lazy=True,
        )
        optimizer.probe(
            params={"lr_pow": -6},
            lazy=True,
        )
    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=0,
        n_iter=100,
    )

    print(optimizer.max)
