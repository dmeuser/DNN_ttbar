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
#  ~import /home/home4/institut_1b/nattland/BayesianOptimization

def fit_test(x,y):
    #  ~x=y
    time.sleep(random.randint(20, 30))
    return -x ** 2 - (y - 1) ** 2 + 1

def hyperOpt():
    pbounds = {'x': (2, 4), 'y': (-3, 3)}

    optimizer = BayesianOptimization(
        f=fit_test,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=30,
    )
    

    logger = bayes_opt.logger.JSONLogger(path="HyperParameterResuls/HParaLogs/logSplit.json", reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    load_logs(optimizer, logs=["/home/home4/institut_1b/nattland/DNN_ttbar/HyperParameterResuls/HParaLogs/logSplit.json"]);
    optimizer.maximize(init_points=0,n_iter=1)



    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)


#############################################################

if __name__ == "__main__":
    for i in range(5):
        hyperOpt()
