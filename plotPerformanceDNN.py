from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
from subprocess import call
from os.path import isfile
import sys
import struct
import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#  ~os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

import scipy.spatial.distance as ssd
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
import shap
import seaborn as sns
import tensorflow as tf

def rmsFUNC(x):
    return np.sqrt(np.mean(np.square(x)))

def meanErr(x):
    return 2*np.std(x)/np.sqrt(len(x))


# function to derive different control performance plots based on trained model
def plot_Output(dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False):
    

    
    train_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/trainResults.pkl")

    
    val_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/valResults.pkl")
    # print(val_x)
    
    
    # Compare DNN and target - training sample
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    train_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/train_x_"+str(modelNr)+".pdf")
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    train_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/train_y_"+str(modelNr)+".pdf")
    
    
    # Compare DNN and target - validation sample
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    val_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/val_x_"+str(modelNr)+".pdf")
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    val_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/val_y_"+str(modelNr)+".pdf")
    
    # Compare DNN output between both samples
    min=train_x["DNN_1"].min()
    max=train_x["DNN_1"].max()
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    train_x["DNN_1"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x["DNN_1"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/DNNcomparison_x_"+str(modelNr)+".pdf")
    min=train_x["DNN_2"].min()
    max=train_x["DNN_2"].max()
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    train_x["DNN_2"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x["DNN_2"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/DNNcomparison_y_"+str(modelNr)+".pdf")
    
    # Compare target between both samples
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    train_x[target[0]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x[target[0]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/TARGETcomparison_x_"+str(modelNr)+".pdf")
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    train_x[target[1]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x[target[1]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/TARGETcomparison_y_.pdf")
    
    # Compare corrected MET to genMET X - validation sample
    val_x["genMET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x[target[0]]
    val_x["DNN_MET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x["DNN_1"]
    
    min=val_x["genMET_X"].min()
    max=val_x["genMET_X"].max()
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    val_x["genMET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_X")
    val_x["DNN_MET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_X")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_x_val_"+str(modelNr)+".pdf")
    
    # Compare corrected MET to genMET X - training sample
    train_x["genMET_X"]=train_x["PuppiMET*cos(PuppiMET_phi)"]-train_x[target[0]]
    train_x["DNN_MET_X"]=train_x["PuppiMET*cos(PuppiMET_phi)"]-train_x["DNN_1"]
    
    min=train_x["genMET_X"].min()
    max=train_x["genMET_X"].max()
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    train_x["genMET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_X")
    train_x["DNN_MET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_X")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_x_train_"+str(modelNr)+".pdf")
    
    # Compare corrected MET to genMET Y - validation sample
    val_x["genMET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x[target[1]]
    val_x["DNN_MET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x["DNN_2"]
    
    min=val_x["genMET_Y"].min()
    max=val_x["genMET_Y"].max()
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    val_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y")
    val_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_y_val_"+str(modelNr)+".pdf")
    
    # Compare corrected MET to genMET Y - training sample
    train_x["genMET_Y"]=train_x["PuppiMET*sin(PuppiMET_phi)"]-train_x[target[1]]
    train_x["DNN_MET_Y"]=train_x["PuppiMET*sin(PuppiMET_phi)"]-train_x["DNN_2"]
    
    min=train_x["genMET_Y"].min()
    max=train_x["genMET_Y"].max()
    plt.figure()
    train_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y")
    train_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y")
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_y_train_"+str(modelNr)+".pdf")
    
    # Compare corrected MET to genMET pT - validation sample
    val_x["genMET"]=np.sqrt(val_x["genMET_X"]**2+val_x["genMET_Y"]**2)
    val_x["DNN_MET"]=np.sqrt(val_x["DNN_MET_X"]**2+val_x["DNN_MET_Y"]**2)
    
    min_x=0
    max_x=400
    #  ~plt.figure()
    #  ~plt.suptitle("validation sample")
    #  ~val_x["DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="DNN", color="b")
    #  ~val_x["genMET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="gen", color="r")
    #  ~plt.xlabel("MET [GeV]")
    #  ~plt.ylabel("Normalized Counts")
    #  ~plt.ylim(0,0.013)
    #  ~plt.legend()
    #  ~plt.grid()
    #  ~plt.tight_layout()
    #  ~plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_val.pdf")
    subRatio = dict(height_ratios=[5,3])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=subRatio)
    fig.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    #  ~fig.suptitle("validation sample")
    countsDNN,bins = np.histogram(val_x["DNN_MET"],bins=500,range=(min_x,max_x))
    countsGen,bins = np.histogram(val_x["genMET"],bins=500,range=(min_x,max_x))
    js_dist = ssd.jensenshannon(countsDNN, countsGen)
    ax1.hist(val_x["DNN_MET"],alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="DNN", color="b")
    ax1.hist(val_x["genMET"],alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="gen", color="r")
    countsDNN, countsGen = np.array(countsDNN), np.array(countsGen)
    ratioErrs = countsDNN/countsGen*np.sqrt(countsDNN/np.square(countsDNN)+countsGen/np.square(countsGen))
    ax2.errorbar(0.5*(bins[1:]+bins[:-1]), countsDNN/countsGen, yerr=ratioErrs, color="b", alpha=0.5, maker=None, ls="none")
    ax2.set_xlabel("MET [GeV]")
    ax2.axhline(1, color="r", linewidth=1, alpha=0.5)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("Normalized Counts")
    ax2.set_ylabel("ratio")
    ax1.set_ylim(0,0.013)
    ax2.set_ylim(0.49,2.01)
    ax1.legend()
    ax1.grid()
    ax2.grid()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_val_"+str(modelNr)+".pdf")
    
    
    # Compare corrected MET to genMET pT - training sample
    train_x["genMET"]=np.sqrt(train_x["genMET_X"]**2+train_x["genMET_Y"]**2)
    train_x["DNN_MET"]=np.sqrt(train_x["DNN_MET_X"]**2+train_x["DNN_MET_Y"]**2)
    
    #  ~plt.figure()
    #  ~plt.suptitle("training sample")
    #  ~train_x["DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="DNN", color="b")
    #  ~train_x["genMET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="gen", color="r")
    #  ~plt.xlabel("MET [GeV]")
    #  ~plt.ylabel("Normalized Counts")
    #  ~plt.ylim(0,0.013)
    #  ~plt.legend()
    #  ~plt.grid()
    #  ~plt.tight_layout()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=subRatio)
    fig.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    countsDNN,bins = np.histogram(train_x["DNN_MET"],bins=500,range=(min_x,max_x))
    countsGen,bins = np.histogram(train_x["genMET"],bins=500,range=(min_x,max_x))
    ax1.hist(train_x["DNN_MET"],alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="DNN", color="b")
    ax1.hist(train_x["genMET"],alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="gen", color="r")
    countsDNN, countsGen = np.array(countsDNN), np.array(countsGen)
    ratioErrs = countsDNN/countsGen*np.sqrt(countsDNN/np.square(countsDNN)+countsGen/np.square(countsGen))
    ax2.errorbar(0.5*(bins[1:]+bins[:-1]), countsDNN/countsGen, yerr=ratioErrs, color="b", alpha=0.5, maker=None, ls="none")
    ax2.set_xlabel("MET [GeV]")
    ax2.axhline(1, color="r", linewidth=1, alpha=0.5)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("Normalized Counts")
    ax2.set_ylabel("ratio")
    ax1.set_ylim(0,0.013)
    ax2.set_ylim(0.49,2.01)
    ax1.legend()
    ax1.grid()
    ax2.grid()
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_train_"+str(modelNr)+".pdf")
    
    # Compare resolution for corrected MET and Puppi - validation
    split_str = ["tot",">=200", "<200"]
    for i,val_temp in enumerate([val_x, val_x[val_x["genMET"]>=200], val_x[val_x["genMET"]<200]]):
        val_temp["PuppiMET"]=np.sqrt(val_temp["PuppiMET*cos(PuppiMET_phi)"]**2+val_temp["PuppiMET*sin(PuppiMET_phi)"]**2)
        val_temp["genMET-PuppiMET"]=val_temp["genMET"]-val_temp["PuppiMET"]
        val_temp["genMET-DNN_MET"]=val_temp["genMET"]-val_temp["DNN_MET"]
        
        puppiMeanVal = "Mean Puppi: {:.2f}\nRMS Puppi: {:.1f}\nMean DNN: {:.2f}\nRMS DNN: {:.1f}".format(val_temp["genMET-PuppiMET"].mean(), rmsFUNC(val_temp["genMET-PuppiMET"]), val_temp["genMET-DNN_MET"].mean(), rmsFUNC(val_temp["genMET-DNN_MET"]))

        min_x=-150
        max_x=150
        if i>0: titleString = split_str[i]+" GeV"
        else: titleString = split_str[i]
        plt.figure()
        plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        plt.suptitle("validation sample"+split_str[i])
        val_temp["genMET-DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="DNN", color="b")
        val_temp["genMET-PuppiMET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="Puppi", color="r")
        plt.text(-148, 0.022, puppiMeanVal, bbox={"facecolor":"none", "pad":5})
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("genMET-recoMET [GeV]")
        plt.ylabel("Normalized Counts")
        plt.ylim(0,0.028)
        plt.grid()
        plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_val"+split_str[i]+"_"+str(modelNr)+".pdf")
        
        print(puppiMeanVal)
        
    for i,train_temp in enumerate([train_x, train_x[train_x["genMET"]>=200], train_x[train_x["genMET"]<200]]):
        # Compare resolution for corrected MET and Puppi - trainidation
        train_temp["PuppiMET"]=np.sqrt(train_temp["PuppiMET*cos(PuppiMET_phi)"]**2+train_temp["PuppiMET*sin(PuppiMET_phi)"]**2)
        train_temp["genMET-PuppiMET"]=train_temp["genMET"]-train_temp["PuppiMET"]
        train_temp["genMET-DNN_MET"]=train_temp["genMET"]-train_temp["DNN_MET"]
        puppiMeanTrain = "Mean Puppi: {:.2f}\nRMS Puppi: {:.1f}\nMean DNN: {:.2f}\nRMS DNN: {:.1f}".format(train_temp["genMET-PuppiMET"].mean(), rmsFUNC(train_temp["genMET-PuppiMET"]), train_temp["genMET-DNN_MET"].mean(), rmsFUNC(train_temp["genMET-DNN_MET"]))
        
        min_x=-150
        max_x=150
        if i>0: titleString = split_str[i]+" GeV"
        else: titleString = split_str[i]
        plt.figure()
        #  ~plt.suptitle("training sample; genMET range: "+split_str[i])
        plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        train_temp["genMET-DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="DNN", color="b")
        train_temp["genMET-PuppiMET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="Puppi", color="r")
        plt.text(-148, 0.022, puppiMeanTrain, bbox={"facecolor":"none", "pad":5})
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("genMET-recoMET [GeV]")
        plt.ylabel("Normalized Counts")
        plt.ylim(0,0.028)
        plt.grid()
        plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_train"+split_str[i]+"_"+str(modelNr)+".pdf")
        
        print(puppiMeanTrain)
    
    #  ~# Compare mean diff as a function of genMET
    min=train_x["genMET"].min()
    max=train_x["genMET"].max()
    
    binsMET = np.linspace(0,600,30)
    train_x["bin"] = np.digitize(train_x["genMET"],bins=binsMET)
    res_DNN_MET = train_x.groupby("bin")["genMET-DNN_MET"].agg(["mean",rmsFUNC,meanErr])
    res_PuppiMET = train_x.groupby("bin")["genMET-PuppiMET"].agg(["mean",rmsFUNC,meanErr])
    res_DNN_MET["metBins"] = binsMET
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    plt.errorbar(binsMET, res_DNN_MET["mean"], yerr=res_DNN_MET["meanErr"], color="b", label=None, ls="none", capsize=3)
    plt.step(binsMET, res_DNN_MET["mean"], color="b", linewidth=1, where="mid", label="DNN mean")
    plt.step(binsMET, res_DNN_MET["rmsFUNC"], color="b", linewidth=1, where="mid", linestyle="--", label="DNN rms")
    plt.errorbar(binsMET, res_PuppiMET["mean"], yerr=res_PuppiMET["meanErr"], color="r", label=None, ls="none", capsize=3)
    plt.step(binsMET, res_PuppiMET["mean"], color="r", linewidth=1, where="mid", label="Puppi mean")
    plt.step(binsMET, res_PuppiMET["rmsFUNC"], color="r", linewidth=1, where="mid", linestyle="--", label="Puppi rms")
    plt.ylabel("genMET-recoMET (GeV)")
    plt.xlabel("genMET (GeV)")
    plt.grid()
    plt.legend()
    plt.ylim(-40,80)
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/MeanDiff_vs_genMET_train_"+str(modelNr)+".pdf")
    
    binsMET = np.linspace(0,600,30)
    val_x["bin"] = np.digitize(val_x["genMET"],bins=binsMET)
    res_DNN_MET = val_x.groupby("bin")["genMET-DNN_MET"].agg(["mean",rmsFUNC,meanErr])
    res_PuppiMET = val_x.groupby("bin")["genMET-PuppiMET"].agg(["mean",rmsFUNC,meanErr])
    res_DNN_MET["metBins"] = binsMET
    plt.figure()
    plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
    plt.errorbar(binsMET, res_DNN_MET["mean"], yerr=res_DNN_MET["meanErr"], color="b", label=None, ls="none", capsize=3)
    plt.step(binsMET, res_DNN_MET["mean"], color="b", linewidth=1, where="mid", label="DNN mean")
    plt.step(binsMET, res_DNN_MET["rmsFUNC"], color="b", linewidth=1, where="mid", linestyle="--", label="DNN rms")
    plt.errorbar(binsMET, res_PuppiMET["mean"], yerr=res_PuppiMET["meanErr"], color="r", label=None, ls="none", capsize=3)
    plt.step(binsMET, res_PuppiMET["mean"], color="r", linewidth=1, where="mid", label="Puppi mean")
    plt.step(binsMET, res_PuppiMET["rmsFUNC"], color="r", linewidth=1, where="mid", linestyle="--", label="Puppi rms")
    plt.ylabel("genMET-recoMET (GeV)")
    plt.xlabel("genMET (GeV)")
    plt.grid()
    plt.legend()
    plt.ylim(-40,80)
    plt.tight_layout(pad=0.1)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/MeanDiff_vs_genMET_val_"+str(modelNr)+".pdf")
    
    #  ~bins=list(range(0,500,100))
    #  ~bins.append(genMET.max())
    
    
    #  ~plt.figure()
    #  ~train_x["genMET"].plot.hist(alpha=0.3, color="blue", bins=bins, density=True, weights=train_weights, label="genMET with weights")
    #  ~train_x["genMET"].plot.hist(alpha=0.3, color="red", bins=300, density=True, weights=train_weights, label="genMET with weights other binning")
    #  ~train_x["genMET"].plot.hist(alpha=0.3, color="green", bins=300, density=True, label="genMET without weights")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/genMET_test_train.pdf")
    
    #  ~plt.figure()
    #  ~val_x["genMET"].plot.hist(alpha=0.3, color="blue", bins=bins, density=True, weights=val_weights, label="genMET with weights")
    #  ~val_x["genMET"].plot.hist(alpha=0.3, color="red", bins=300, density=True, weights=val_weights, label="genMET with weights other binning")
    #  ~val_x["genMET"].plot.hist(alpha=0.3, color="green", bins=300, density=True, label="genMET without weights")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/genMET_test_val.pdf")
    
    #  ~plt.figure()
    #  ~plt.hist(train_x["genMET"], 300, density=True, weights=train_weights, label="genMET with different binning as weights")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/genMET_testBin2.pdf")

    
#############################################################

if __name__ == "__main__":
    # Define input data path
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v04/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"

    # Define Input Variables
    inputVars = ["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1421genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,1)GeV".format(0.3,0)
    #  ~modelNr = 16
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1017genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,100)GeV".format(0.3,0)
    #  ~modelNr = 13
    
    modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211012-0945"
    modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: none".format(0.1,0)
    modelNr = 1
    
    plot_Output(dataPath,inputVars,modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=False,genMETweighted=True)

