#  ~from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
#  ~from subprocess import call
#  ~from os.path import isfile
#  ~import sys
#  ~import struct
#  ~import datetime

#  ~import os
#  ~os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#  ~os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

#  ~from tensorflow.keras import Sequential
#  ~from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU, BatchNormalization
#  ~from tensorflow.keras.regularizers import l2
#  ~from tensorflow.keras.optimizers import SGD, Adam
#  ~from tensorflow.keras.constraints import NonNeg
#  ~from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#  ~from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
#  ~from tensorflow.keras.models import load_model

#  ~import uproot
#  ~import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#  ~import eli5
#  ~from eli5.sklearn import PermutationImportance
#  ~from sklearn.model_selection import train_test_split
#  ~from sklearn.preprocessing import MinMaxScaler,StandardScaler
#  ~from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
#  ~import shap
#  ~import seaborn as sns
#  ~import tensorflow as tf

def plotLoss(pathList):
    losses, steps = [], []
    for path in pathList:
        templosses, tempsteps = [], []
        with open(path) as f:
            for line in f.readlines()[1:]:
                tempsteps.append(line.split(",")[1])
                templosses.append(line.split(",")[2][:-1])
        losses.append(templosses)
        steps.append(tempsteps)
    steps = np.array(steps, dtype=int)
    losses = np.array(losses, dtype=float)
    plt.figure("Loss Comparison Zoom")
    plt.plot(steps[0], losses[0], color="red", ls="--", label="dropout: 0.3, $\lambda$: 0 (training)")
    plt.plot(steps[1], losses[1], color="red", ls="-", label="dropout: 0.3, $\lambda$: 0 (validation)")
    plt.plot(steps[6], losses[6], color="magenta", ls="--", label="dropout: 0.4, $\lambda$: 0 (training)")
    plt.plot(steps[7], losses[7], color="magenta", ls="-", label="dropout: 0.4, $\lambda$: 0 (validation)")
    plt.plot(steps[2], losses[2], color="blue", ls="--", label="dropout: 0.4, $\lambda$: 0.05 (training)")
    plt.plot(steps[3], losses[3], color="blue", ls="-", label="dropout: 0.4, $\lambda$: 0.05 (validation)")
    plt.plot(steps[4], losses[4], color="green", ls="--", label="dropout: 0.4, $\lambda$: 0.2 (training)")
    plt.plot(steps[5], losses[5], color="green", ls="-", label="dropout: 0.4, $\lambda$: 0.2 (validation)")
    #  ~plt.xlim(10,50)
    plt.ylim(19,100)
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    #  ~plt.show()
    plt.tight_layout()
    plt.savefig("LossPlots/Loss_Comp_Regu.pdf")
    
    
#############################################################

if __name__ == "__main__":
    pathList = ["LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1421genMETweighted_train-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1421genMETweighted_validation-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1509genMETweighted_train-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1509genMETweighted_validation-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1548genMETweighted_train-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1548genMETweighted_validation-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211025-1130genMETweighted_train-tag-epoch_loss.csv",
    "LossPlots/run-Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211025-1130genMETweighted_validation-tag-epoch_loss.csv"]
    plotLoss(pathList)

