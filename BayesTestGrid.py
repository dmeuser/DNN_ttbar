import numpy as np
#  ~from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
#  ~from subprocess import call
#  ~from os.path import isfile
#  ~import sys
#  ~import struct
#  ~import datetime

#  ~import os
#  ~import pandas as pd
from BayesianOptimization.bayes_opt import BayesianOptimization
from BayesianOptimization import bayes_opt
from BayesianOptimization.bayes_opt.event import Events
from BayesianOptimization.bayes_opt.util import load_logs
import time
import random
import argparse
#  ~import /home/home4/institut_1b/nattland/BayesianOptimization

def fit_test(x,y):
    #  ~x=y
    #  ~time.sleep(random.randint(20, 30))
    time.sleep(random.randint(1, 5))
    return -x ** 2 - (y - 1) ** 2 + 1

def hyperOpt(kappa):
    pbounds = {'x': (2, 4), 'y': (-3, 3)}

    optimizer = BayesianOptimization(
        f=fit_test,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        #  ~random_state=30,
    )
    
    load_logs(optimizer, logs=["/home/home4/institut_1b/nattland/DNN_ttbar/HyperParameterResuls/HParaLogs/logTests/logSplit.json"]);
    logger = bayes_opt.logger.JSONLogger(path="HyperParameterResuls/HParaLogs/logTests/logSplit.json", reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    optimizer.maximize(init_points=1,n_iter=1, kappa=kappa)
    
    
    #  ~print(type(optimizer.res),len(optimizer.res))
    for i, res in enumerate(optimizer.res[-2:]):
        print("Iteration {}: {}".format(len(optimizer.res)-1+i, res))
    
    #  ~print(optimizer.max)




#############################################################

if __name__ == "__main__":
    #  ~for i in range(5):
        #  ~hyperOpt()
    parser = argparse.ArgumentParser(description="optimize parameters with given kappa value")
    parser.add_argument("kappaVal", metavar="K", type=float, help="kappa parameter for optimization")
    args=parser.parse_args()
    for i in range(10):
        hyperOpt(args.kappaVal)
