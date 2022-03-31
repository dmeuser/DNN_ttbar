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
from DNN_funcs import getInputArray_allBins_nomDistr, getMETarrays

#  ~def rmsFUNC(x):
    #  ~return np.sqrt(np.mean(np.square(x)))

def rmsFUNC(x):
    return np.sqrt(np.mean(np.square(x-np.mean(x))))

def meanErr(x):
    return 2*np.std(x)/np.sqrt(len(x))



def get_inputs(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    
    if not os.path.exists("outputComparison/2018/2D/"+modelPath.split("/")[-1]):    # create output folder for plots if not available
        os.makedirs("outputComparison/2018/2D/"+modelPath.split("/")[-1])
    
    # get inputs
    if genMETweighted:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights = getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
        
    else:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    model = load_model(modelPath+".h5")
    
    print(train_y.keys())
    
    if standardize:
        train_x_scaler = StandardScaler()
        train_x_std = pd.DataFrame(train_x_scaler.fit_transform(train_x.values), index=train_x.index, columns=train_x.columns)
        
        train_y_scaler = StandardScaler()
        train_y_std = pd.DataFrame(train_y_scaler.fit_transform(train_y.values), index=train_y.index, columns=train_y.columns)
        
        val_x_scaler = StandardScaler()
        val_x_std = pd.DataFrame(val_x_scaler.fit_transform(val_x.values), index=val_x.index, columns=val_x.columns)
        
        val_y_scaler = StandardScaler()
        val_y_std = pd.DataFrame(val_y_scaler.fit_transform(val_y.values), index=val_y.index, columns=val_y.columns)
    
    #  ~train_x=train_x.head(100000)    # could be used to only create plots for limited statistics (mainly for debugging)
    #  ~val_x=val_x.head(100000)
    
    # evaluate trained model
        y_hat_train = model.predict(train_x_std,use_multiprocessing=True)
        y_hat_val = model.predict(val_x_std,use_multiprocessing=True)
        y_hat_train = train_y_scaler.inverse_transform(y_hat_train)
        y_hat_val = val_y_scaler.inverse_transform(y_hat_val)
    else:
        y_hat_train = model.predict(train_x,use_multiprocessing=True)
        y_hat_val = model.predict(val_x,use_multiprocessing=True)
        y_hat_test = model.predict(test_x,use_multiprocessing=True)

    
    train_x["DNN_1"]=[row[0] for row in y_hat_train]
    train_x["DNN_2"]=[row[1] for row in y_hat_train]
    train_x[target[0]]=train_y[target[0]]
    train_x[target[1]]=train_y[target[1]]
    train_x["PuppiMET_xy_phi"]=train_metVals["PuppiMET_xy_phi"]
    train_x["genMET_phi"]=train_metVals["genMET_phi"]
    
    #  ~train_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/trainResults.pkl")
    
    val_x["DNN_1"]=[row[0] for row in y_hat_val]
    val_x["DNN_2"]=[row[1] for row in y_hat_val]
    val_x[target[0]]=val_y[target[0]]
    val_x[target[1]]=val_y[target[1]]
    val_x["PuppiMET_xy_phi"]=val_metVals["PuppiMET_xy_phi"]
    val_x["genMET_phi"]=val_metVals["genMET_phi"]
    
    #  ~val_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/valResults.pkl")
    
    test_x["DNN_1"]=[row[0] for row in y_hat_test]
    test_x["DNN_2"]=[row[1] for row in y_hat_test]
    test_x[target[0]]=test_y[target[0]]
    test_x[target[1]]=test_y[target[1]]
    test_x["PuppiMET_xy_phi"]=test_metVals["PuppiMET_xy_phi"]
    test_x["genMET_phi"]=test_metVals["genMET_phi"]
    
    #  ~test_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/testResults.pkl")
    return train_x, val_x, test_x


def print_targets(dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
     
    #  ~train_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/trainResults.pkl")
    #  ~val_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/valResults.pkl")
    #  ~test_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/testResults.pkl")
    train_x, val_x, test_x = get_inputs(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    binN=100
    sampleNames=["training sample", "validation sample", "test sample"]
    
    print("\n\nDNN NR. ",modelNr)
    print(title)
    #  ~print(train_x)
    
    #  ~return "haha"
    
    for i_sample,data_x in enumerate([train_x, val_x, test_x]):
        #  ~data_x["DNN_MET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x["DNN_1"]
        #  ~data_x["DNN_MET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x["DNN_2"]
        #  ~data_x["DNN_MET"]=np.sqrt(data_x["DNN_MET_X"]**2+data_x["DNN_MET_Y"]**2)
        #  ~data_x["genMET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x[target[0]]
        #  ~data_x["genMET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x[target[1]]
        #  ~data_x["genMET"]=np.sqrt(data_x["genMET_X"]**2+data_x["genMET_Y"]**2)
        
        y_pred = [data_x["DNN_1"], data_x["DNN_2"]]
        y_true = [data_x[target[0]], data_x[target[1]]]
        l = tf.keras.losses.LogCosh()
        logcosh = l(y_true, y_pred).numpy()
        
        #  ~met_MSE=np.mean((data_x["DNN_MET"]-data_x["genMET"])**2)
        #  ~metVec_mean=np.mean((data_x["DNN_MET_X"]-data_x["genMET_X"])**2+(data_x["DNN_MET_Y"]-data_x["genMET_Y"])**2)
        
        #  ~min_x, max_x = 0., 400.
        #  ~countsDNN,bins = np.histogram(np.clip(data_x["DNN_MET"],min_x+0.01,max_x-0.01),bins=binN,range=(min_x,max_x))
        #  ~countsGen,bins = np.histogram(np.clip(data_x["genMET"],min_x+0.01,max_x-0.01),bins=binN,range=(min_x,max_x))
        #  ~countsDNN, countsGen = np.array(countsDNN), np.array(countsGen)
        

        #  ~print("\n\n"+sampleNames[i_sample]+"\n")
        #  ~jensenShannon_met = ssd.jensenshannon(countsDNN, countsGen)
        #  ~chisquare_met = sum((countsDNN-countsGen)**2/countsGen)/binN
        
        print("loss (logcosh w/o regularisation term): {0:.5g}".format(logcosh))
        #  ~print("MSE met difference absolute: {0:.5g}".format(met_MSE))
        #  ~print("mean met difference vectorial: {0:.5g}".format(metVec_mean))
        #  ~print("Jensen-Shannon distance met: {0:.5g}".format(jensenShannon_met))
        #  ~print("chi square met distribution: {0:.5g}".format(chisquare_met))


def plot_Output(dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    
    #  ~train_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/trainResults.pkl")
    #  ~val_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/valResults.pkl")
    #  ~test_x=pd.read_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/testResults.pkl")
    
    train_x, val_x, test_x = get_inputs(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    sampleNames=["train", "val", "test"]
    #  ~sampleNames=["train", "val"]
    
    for i_sample,data_x in enumerate([train_x, val_x, test_x]):
    #  ~for i_sample,data_x in enumerate([train_x, val_x]):
        
        # Compare DNN and target - training sample
        plt.figure()
        plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        data_x[["PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)"]].plot.hist(alpha=0.5,bins=500,density=True)
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/"+sampleNames[i_sample]+"_Puppi_xy_"+str(modelNr)+".pdf")
        
        plt.figure()
        plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        data_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True)
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/"+sampleNames[i_sample]+"_x_"+str(modelNr)+".pdf")
        plt.figure()
        plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        data_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True)
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/"+sampleNames[i_sample]+"_y_"+str(modelNr)+".pdf")

        
        # Compare corrected MET to genMET X - training sample
        data_x["genMET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x[target[0]]
        data_x["DNN_MET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x["DNN_1"]
        
        min=data_x["genMET_X"].min()
        max=data_x["genMET_X"].max()
        plt.figure()
        plt.suptitle(title, fontsize=12, ha="left", x=0.1, y=0.99)
        data_x["genMET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_X")
        data_x["DNN_MET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_X")
        plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_x_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        
        # Compare corrected MET to genMET Y - training sample
        data_x["genMET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x[target[1]]
        data_x["DNN_MET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x["DNN_2"]
        
        min=data_x["genMET_Y"].min()
        max=data_x["genMET_Y"].max()
        plt.figure()
        data_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y")
        data_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y")
        plt.legend()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_y_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        
        # Compare corrected MET to genMET pT - training sample        
        min_x=0
        max_x=400
        subRatio = dict(height_ratios=[5,3])
        binN = 100
        
        data_x["genMET"]=np.sqrt(data_x["genMET_X"]**2+data_x["genMET_Y"]**2)
        data_x["DNN_MET"]=np.sqrt(data_x["DNN_MET_X"]**2+data_x["DNN_MET_Y"]**2)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=subRatio)
        fig.suptitle(title, fontsize=10, ha="left", x=0.1, y=0.99)
        countsDNN,bins = np.histogram(data_x["DNN_MET"],bins=binN,range=(min_x,max_x))
        countsGen,bins = np.histogram(data_x["genMET"],bins=binN,range=(min_x,max_x))
        ax1.hist(data_x["DNN_MET"],alpha=0.5,bins=binN,range=(min_x,max_x),density=True,label="DNN", color="b")
        ax1.hist(data_x["genMET"],alpha=0.5,bins=binN,range=(min_x,max_x),density=True,label="gen", color="r")
        countsDNN, countsGen = np.array(countsDNN), np.array(countsGen)
        ratioErrs = countsDNN/countsGen*np.sqrt(countsDNN/np.square(countsDNN)+countsGen/np.square(countsGen))
        ax2.errorbar(0.5*(bins[1:]+bins[:-1]), countsDNN/countsGen, yerr=ratioErrs, color="b", fmt=".", capsize=2, elinewidth=1, markersize=1.5, ls="none")
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
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        
        # Compare resolution for corrected MET and Puppi - validation
        split_str = ["tot",">=200 GeV", "<200 GeV"]
        #  ~data_x["dPhi_DNN_gen"] = np.pi + np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_x["Phi_gen"]
        #  ~data_x["dPhi_DNN"] = np.pi + np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])
        dPhi_DNN_gen_arr = np.array([(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_x["genMET_phi"]), (2*np.pi+np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_x["genMET_phi"])), (-2*np.pi+np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_x["genMET_phi"]))])
        data_x["dPhi_DNN_gen"] = dPhi_DNN_gen_arr.flatten()[np.arange(np.shape(dPhi_DNN_gen_arr)[1]) + np.abs(dPhi_DNN_gen_arr).argmin(axis=0)*dPhi_DNN_gen_arr.shape[1]]
        
        dPhi_Puppi_gen_arr = np.array([(data_x["PuppiMET_xy_phi"]-data_x["genMET_phi"]), (2*np.pi+(data_x["PuppiMET_xy_phi"]-data_x["genMET_phi"])), (-2*np.pi+(data_x["PuppiMET_xy_phi"]-data_x["genMET_phi"]))])
        data_x["dPhi_Puppi_gen"] = dPhi_Puppi_gen_arr.flatten()[np.arange(np.shape(dPhi_Puppi_gen_arr)[1]) + np.abs(dPhi_Puppi_gen_arr).argmin(axis=0)*dPhi_Puppi_gen_arr.shape[1]]
        #  ~data_x["dPhi_Puppi_gen"] = data_x["Phi_recPuppi"]-data_x["Phi_gen"]
        #  ~data_x["dPhi_Puppi_gen"] = data_x["Phi_gen"]
        
        for i,train_temp in enumerate([data_x, data_x[data_x["genMET"]>=200], data_x[data_x["genMET"]<200]]):
            # Compare resolution for corrected MET and Puppi - trainidation
            train_temp["PuppiMET_xy"]=np.sqrt(train_temp["PuppiMET_xy*cos(PuppiMET_xy_phi)"]**2+train_temp["PuppiMET_xy*sin(PuppiMET_xy_phi)"]**2)
            train_temp["genMET-PuppiMET_xy"]=train_temp["genMET"]-train_temp["PuppiMET_xy"]
            train_temp["genMET-DNN_MET"]=train_temp["genMET"]-train_temp["DNN_MET"]
            #  ~puppiMeanTrain = "Mean Puppi: {:.2f}\nStd Puppi: {:.2f}\nMean DNN: {:.2f}\nStd DNN: {:.2f}".format(train_temp["genMET-PuppiMET_xy"].mean(), rmsFUNC(train_temp["genMET-PuppiMET_xy"]), train_temp["genMET-DNN_MET"].mean(), rmsFUNC(train_temp["genMET-DNN_MET"]))
            plotStats = "Mean Puppi: {:.2f}\nStd Puppi: {:.2f}\nMean DNN: {:.2f}\nStd DNN: {:.2f}".format(train_temp["genMET-PuppiMET_xy"].mean(), train_temp["genMET-PuppiMET_xy"].std(), train_temp["genMET-DNN_MET"].mean(), train_temp["genMET-DNN_MET"].std())
            #  ~print("\n############\n\n")
            #  ~print(data_x["dPhi_DNN_gen"][:30], data_x["dPhi_Puppi_gen"][:30])
            #  ~print(puppiMeanTrain)
            min_x=-150
            max_x=150
            plt.figure()
            #  ~plt.suptitle("genMET "+split_str[i]+"\n"+title, fontsize=10, ha="left", x=0.1, y=0.99)
            train_temp["genMET-DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="DNN", color="b")
            train_temp["genMET-PuppiMET_xy"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="Puppi", color="r")
            plt.text(-148, 0.8*plt.gca().get_ylim()[1], plotStats, bbox={"facecolor":"none", "pad":5})
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel("genMET-recoMET [GeV]")
            plt.ylabel("Events")
            #  ~plt.grid()
            plt.legend()
            ax = plt.gca()
            ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
            ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom',fontname='Helvetica', style="italic", fontsize=10, color=(0.3,0.3,0.3))
            ax.text(1.,1.,r"$59.7\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
            plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_"+sampleNames[i_sample]+split_str[i]+"_"+str(modelNr)+".pdf")
            
            plt.clf()
            min_x = np.min(train_temp["dPhi_Puppi_gen"])
            max_x = np.max(train_temp["dPhi_Puppi_gen"])
            plotStats = "Mean Puppi: {:.2f}\nStd Puppi: {:.2f}\nMean DNN: {:.2f}\nStd DNN: {:.2f}".format(train_temp["dPhi_Puppi_gen"].mean(), train_temp["dPhi_Puppi_gen"].std(), train_temp["dPhi_DNN_gen"].mean(), train_temp["dPhi_DNN_gen"].std())
            plt.figure(figsize=(9,5))
            #  ~plt.suptitle("genMET "+split_str[i]+"\n"+title, fontsize=10, ha="left", x=0.1, y=0.99)
            train_temp["dPhi_DNN_gen"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="DNN", color="b")
            train_temp["dPhi_Puppi_gen"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=False,label="Puppi", color="r")
            plt.text(-3, 0.8*plt.gca().get_ylim()[1], plotStats, bbox={"facecolor":"none", "pad":5})
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel(r"$\Delta\Phi(genMET,\,recoMET)$")
            plt.ylabel("Events")
            #  ~plt.grid()
            plt.legend()
            #  ~plt.ylim(0,1)
            ax = plt.gca()
            ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
            ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom',fontname='Helvetica', style="italic", fontsize=10, color=(0.3,0.3,0.3))
            ax.text(1.,1.,r"$59.7\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)
            plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Resolution_dPhi_"+sampleNames[i_sample]+split_str[i]+"_"+str(modelNr)+".pdf")
            
            #  ~#print(puppiMeanTrain)
        
        #  ~# Compare mean diff as a function of genMET

        binsMET = np.linspace(0,400,21)
        data_x["bin"] = np.digitize(data_x["genMET"],bins=binsMET)
        res_DNN_MET = data_x.groupby("bin")["genMET-DNN_MET"].agg(["mean",rmsFUNC,meanErr])
        res_PuppiMET_xy = data_x.groupby("bin")["genMET-PuppiMET_xy"].agg(["mean",rmsFUNC,meanErr])
        res_DNN_MET["metBins"] = binsMET
        plt.figure()
        #  ~plt.suptitle(title, fontsize=10, ha="left", x=0.1, y=0.99)
        #  ~print("teste")
        plt.errorbar(binsMET, res_DNN_MET["mean"], yerr=res_DNN_MET["meanErr"], color="b", label=None, ls="none", capsize=3)
        plt.step(binsMET, res_DNN_MET["mean"], color="b", linewidth=1, where="mid", label="DNN mean")
        plt.step(binsMET, res_DNN_MET["rmsFUNC"], color="b", linewidth=1, where="mid", linestyle="--", label="DNN rms")
        plt.errorbar(binsMET, res_PuppiMET_xy["mean"], yerr=res_PuppiMET_xy["meanErr"], color="r", label=None, ls="none", capsize=3)
        plt.step(binsMET, res_PuppiMET_xy["mean"], color="r", linewidth=1, where="mid", label="Puppi mean")
        plt.step(binsMET, res_PuppiMET_xy["rmsFUNC"], color="r", linewidth=1, where="mid", linestyle="--", label="Puppi rms")
        plt.ylabel("genMET-recoMET (GeV)")
        plt.xlabel("genMET (GeV)")
        plt.grid()
        plt.legend()
        plt.ylim(-40,80)
        makeCMStitle()
        plt.tight_layout(pad=0.1)
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/MeanDiff_vs_genMET_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")

        
        plt.figure()
        #  ~data_x["genMET"].plot.hist(alpha=0.3, color="blue", bins=bins, density=True, weights=train_weights, label="genMET with weights")
        #  ~data_x["genMET"].plot.hist(alpha=0.3, color="red", bins=300, density=True, weights=train_weights, label="genMET with weights")
        data_x["genMET"].plot.hist(alpha=0.6, color="b", bins=300, density=True, label="genMET "+sampleNames[i_sample]+"ing w/o weights")
        plt.legend()
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/genMET_distr_"+sampleNames[i_sample]+"_"+str(modelNr)+".pdf")
        

        
def makeCMStitle():
    ax = plt.gca()
    ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    ax.text(1.,1.,r"$59.7\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)

def plot_Purity(dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    print("1")
    
    train_metVals, val_metVals, test_metVals = getMETarrays(dataPath,inputVars,modelPath,treeName,targetName,target,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    
    
    print("2")
    
    print(val_metVals.shape, test_metVals.shape)
    
    # binning in met and dphi
    metBins = np.array([0,40,80,120,160,230,400])
    dphiBins = np.array([0,0.7,1.4,3.15])
    
    
    for (data_metVals, sampName) in [(train_metVals, "train")]:
    #  ~for (data_metVals, sampName) in [(train_metVals, "train"), (val_metVals, "val"), (test_metVals, "test")]:
        
        # using last bin as overflow bin
        
        met_gen = np.clip(data_metVals["PtNuNu"], metBins[0], metBins[-1])
        dphi_gen = data_metVals["dPhi_PtNuNuNearLep"]
        
        met_reco_DNN = np.clip(data_metVals["DNN_MET"], metBins[0], metBins[-1])
        dphi_reco_DNN = data_metVals["DNN_dPhiMetNearLep"]

        met_reco_Puppi = np.clip(data_metVals["PuppiMET_xy"], metBins[0], metBins[-1])
        dphi_reco_Puppi = data_metVals["dPhi_PuppiNearLep"]
        
        for (met_reco, dphi_reco, dnnName) in [(met_reco_DNN, dphi_reco_DNN, "DNN",), (met_reco_Puppi, dphi_reco_Puppi, "Puppi")]:
            
            # creating histograms according to generated met/dphi (Gen), reconstructed met/dphi, and events, where both gen and reco are in the same bin
            
            histo2D_Gen, xedges, yedges = np.histogram2d(met_gen, dphi_gen, bins=[metBins, dphiBins])
            histo2D_Reco, xedges, yedges = np.histogram2d(met_reco, dphi_reco, bins=[metBins, dphiBins])
            histo2D_Both = np.copy(histo2D_Gen)
            
            nB = (len(metBins)-1)*(len(dphiBins)-1)
            histoResponse, edges = np.histogramdd(np.array([dphi_reco, met_reco, dphi_gen, met_gen]).T, bins=(dphiBins, metBins, dphiBins, metBins))
            histoResponse = histoResponse.reshape(len(metBins)-1,len(dphiBins)-1,-1)
            histoResponse = histoResponse.reshape(-1,nB)
            histoResponse = histoResponse / histoResponse.sum(axis=1)[:,np.newaxis]
            
            for i in range(len(metBins)-1):
                for j in range(len(dphiBins)-1):
                    temp1 = np.where((met_gen>metBins[i]) & (met_gen<=metBins[i+1]) & (dphi_gen>dphiBins[j]) & (dphi_gen<=dphiBins[j+1]), True, False)
                    temp2 = np.where((met_reco>metBins[i]) & (met_reco<=metBins[i+1]) & (dphi_reco>dphiBins[j]) & (dphi_reco<=dphiBins[j+1]), True, False)
                    histo2D_Both[i,j] = sum(np.where(temp1 & temp2, 1, 0))
                    
                    
            histo2D_Gen = histo2D_Gen.T
            histo2D_Reco = histo2D_Reco.T
            histo2D_Both = histo2D_Both.T
            
            print("{} Reco {} Last Bin Count: {}".format(dnnName, sampName, histo2D_Reco[-1,-1]))
            print("{} Both {} Last Bin Count: {}".format(dnnName, sampName, histo2D_Both[-1,-1]))
            
            # Plotting Purity
            
            Xbins, Ybins = np.meshgrid(metBins, dphiBins)
            metBinsC = (metBins[1:]+metBins[:-1])*0.5
            dphiBinsC = (dphiBins[1:]+dphiBins[:-1])*0.5
            
            fig, ax = plt.subplots(1,1)
            #  ~#fig.suptitle("{}, {} sample\n, {}".format(dnnName, sampName, title), fontsize=12, ha="left", x=0.1, y=0.99)
            mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Reco, vmin=0., vmax=1.0)
            for i,phiarr in enumerate(histo2D_Both/histo2D_Reco):
                for j,vali in enumerate(phiarr):
                    ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
                    tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Reco[i,j])
                    ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("purity")
            ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
            ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
            makeCMStitle()
            plt.savefig("outputComparison/2018/2D/{model}/{dnnName}_purity_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))
            
            # Plotting Stability
            
            fig, ax = plt.subplots(1,1)
            #  ~#fig.suptitle("{}, {} sample\n, {}".format(dnnName, sampName, title), fontsize=12, ha="left", x=0.1, y=0.99)
            mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Gen, vmin=0., vmax=1.0)
            for i,phiarr in enumerate(histo2D_Both/histo2D_Gen):
                for j,vali in enumerate(phiarr):
                    ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
                    tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Gen[i,j])
                    ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("stability")
            ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
            ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
            makeCMStitle()
            plt.savefig("outputComparison/2018/2D/{model}/{dnnName}_stability_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))
            
            
            # Plotting Response Matrix
            
            Xbins, Ybins = np.meshgrid(np.arange(nB+1),np.arange(nB+1))
            binsC = (np.arange(nB+1)[1:]+np.arange(nB+1)[:-1])*0.5
            
            fig, ax = plt.subplots(figsize=(9,5))
            mesh1 = ax.pcolormesh(Xbins, Ybins, histoResponse, vmin=0., vmax=1.)
            for i,row in enumerate(histoResponse):
                for j,vali in enumerate(row):
                    ax.text(binsC[j]-0.4, binsC[i]-0.25, "{:.2f}".format(vali)[1:], fontsize=10, color="red", fontweight="bold")
                    # tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Gen[i,j])
                    # ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("line normalized distribution")
            ax.set_ylabel(r"reco bin number")
            ax.set_xlabel(r"gen bin number")
            ax.axhline(6., color="g", linestyle="dashed")
            ax.axhline(12., color="g", linestyle="dashed")
            ax.axvline(6., color="g", linestyle="dashed")
            ax.axvline(12., color="g", linestyle="dashed")
            makeCMStitle()
            plt.tight_layout()
            plt.savefig("outputComparison/2018/2D/{model}/{dnnName}_Response_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))

    
#############################################################

if __name__ == "__main__":
    # Define input data path
    #  ~dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v06/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v07/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"
    #  ~dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v06/minTrees/100.0/Nominal/TTbar_diLepton_merged.root"

    # Define Input Variables
    #  ~inputVars1 = ["PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    #  ~inputVars = ["vecsum_pT_allJet*sin(HT_phi)","vecsum_pT_allJet*cos(HT_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)","PuppiMET_xy*cos(PuppiMET_xy_phi)","MET*sin(PFMET_phi)","MET*cos(PFMET_phi)","CaloMET*sin(CaloMET_phi)","CaloMET*cos(CaloMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","Jet1_pt*sin(Jet1_phi)","dPhiMETnearJet_Puppi","Lep2_pt*sin(Lep2_phi)","n_Interactions","dPhiMETnearJet","Lep1_pt*sin(Lep1_phi)", "Lep2_pt*cos(Lep2_phi)","dPhiLep1Jet1","dPhiLep1Lep2","Jet2_pt*cos(Jet2_phi)","vecsum_pT_l1l2_allJet","MT2","METunc_Puppi","dPhiMETfarJet_Puppi","dPhiMETfarJet","nJets","mLL","Jet2_eta","dPhiMETleadJet_Puppi","Lep1_eta","dPhiJet1Jet2","Jet1_eta","Jet1_pt*cos(Jet1_phi)","dPhiMETleadJet","Lep1_pt*cos(Lep1_phi)","Jet2_E","Lep1_flavor","MT","dPhiMETbJet_Puppi","Lep2_flavor","Lep1_E","Lep2_E","mjj","Lep2_eta","dPhiMETlead2Jet_Puppi","METsig","vecsum_pT_allJet","Jet2_pt*sin(Jet2_phi)","Jet1_E","dPhiMETbJet","MHT","dPhiMETlead2Jet","mass_l1l2_allJet","dPhiLep1bJet","ratio_vecsumpTlep_vecsumpTjet","looseLeptonVeto"]
    #  ~inputVars = ["METunc_Puppi","PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","CaloMET*sin(CaloMET_phi)","CaloMET*cos(CaloMET_phi)","vecsum_pT_allJet*sin(HT_phi)","vecsum_pT_allJet*cos(HT_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_pt*cos(Jet1_phi)","MT2","Lep2_pt*sin(Lep2_phi)","Lep2_pt*cos(Lep2_phi)","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_pt*cos(Lep1_phi)","n_Interactions","dPhiJet1Jet2","dPhiMETnearJet_Puppi","dPhiLep1Jet1","vecsum_pT_allJet","dPhiMETleadJet","nJets","MT","dPhiMETnearJet","Jet2_E","Jet1_eta","dPhiLep1Lep2","mLL","dPhiMETbJet_Puppi","dPhiMETfarJet","Lep2_flavor","vecsum_pT_l1l2_allJet","looseLeptonVeto","Lep2_eta","Lep1_eta","dPhiMETlead2Jet","dPhiMETlead2Jet_Puppi","dPhiMETbJet","dPhiLep1bJet","Lep1_E","Lep1_flavor","ratio_vecsumpTlep_vecsumpTjet","mjj","Lep2_E","dPhiMETfarJet_Puppi","mass_l1l2_allJet","METsig","dPhiMETleadJet_Puppi","Jet2_eta","MHT","Jet1_E","HT"]
    #  ~inputVars = ["METunc_Puppi", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "PuppiMET_xy*sin(PuppiMET_xy_phi)", "MET*cos(PFMET_phi)", "MET*sin(PFMET_phi)", "CaloMET*sin(CaloMET_phi)", "CaloMET*cos(CaloMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "Jet1_pt*sin(Jet1_phi)", "Jet1_pt*cos(Jet1_phi)", "MHT", "mass_l1l2_allJet", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "mjj", "n_Interactions", "MT2", "Lep2_pt*sin(Lep2_phi)", "dPhiMETleadJet_Puppi", "Lep2_pt*cos(Lep2_phi)", "HT", "dPhiMETleadJet", "dPhiLep1Jet1", "MT", "Lep1_pt*cos(Lep1_phi)", "vecsum_pT_allJet", "dPhiMETnearJet_Puppi", "vecsum_pT_l1l2_allJet", "nJets", "dPhiMETnearJet", "dPhiJet1Jet2", "Jet2_E", "Lep1_pt*sin(Lep1_phi)", "Jet1_E", "dPhiMETlead2Jet_Puppi", "dPhiLep1Lep2", "Lep1_E", "dPhiMETfarJet", "Jet2_eta", "dPhiMETbJet", "dPhiMETfarJet_Puppi", "mLL", "dPhiMETbJet_Puppi", "Lep2_flavor", "Lep2_E", "Jet1_eta", "Lep1_eta", "dPhiMETlead2Jet", "Lep1_flavor", "dPhiLep1bJet", "Lep2_eta", "METsig", "ratio_vecsumpTlep_vecsumpTjet", "looseLeptonVeto"]
    #  ~inputVars = ["METunc_Puppi", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "PuppiMET_xy*sin(PuppiMET_xy_phi)", "MET*cos(PFMET_phi)", "MET*sin(PFMET_phi)", "CaloMET*sin(CaloMET_phi)", "CaloMET*cos(CaloMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "Jet1_pt*sin(Jet1_phi)", "Jet1_pt*cos(Jet1_phi)", "MHT", "mass_l1l2_allJet", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "mjj", "n_Interactions", "MT2", "Lep2_pt*sin(Lep2_phi)", "dPhiMETleadJet_Puppi", "Lep2_pt*cos(Lep2_phi)", "HT", "dPhiMETleadJet", "dPhiLep1Jet1", "MT", "Lep1_pt*cos(Lep1_phi)", "vecsum_pT_allJet", "dPhiMETnearJet_Puppi", "vecsum_pT_l1l2_allJet", "nJets", "dPhiMETnearJet", "dPhiJet1Jet2", "Jet2_E", "Lep1_pt*sin(Lep1_phi)", "Jet1_E", "dPhiMETlead2Jet_Puppi", "dPhiLep1Lep2", "Lep1_E", "dPhiMETfarJet", "Jet2_eta", "dPhiMETbJet", "dPhiMETfarJet_Puppi", "mLL", "dPhiMETbJet_Puppi", "Lep2_flavor", "Lep2_E", "Jet1_eta", "Lep1_eta", "dPhiMETlead2Jet", "Lep1_flavor", "dPhiLep1bJet", "Lep2_eta", "METsig", "ratio_vecsumpTlep_vecsumpTjet", "looseLeptonVeto"]
    #  ~inputVars = ["METunc_Puppi", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "PuppiMET_xy*sin(PuppiMET_xy_phi)", "MET*cos(PFMET_phi)", "MET*sin(PFMET_phi)", "CaloMET", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "Jet1_pt*sin(Jet1_phi)", "Jet1_pt*cos(Jet1_phi)", "MHT", "mass_l1l2_allJet", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "mjj", "n_Interactions", "MT2", "Lep2_pt*sin(Lep2_phi)", "dPhiMETleadJet_Puppi", "Lep2_pt*cos(Lep2_phi)", "HT", "dPhiMETleadJet", "dPhiLep1Jet1", "MT", "Lep1_pt*cos(Lep1_phi)", "vecsum_pT_allJet", "dPhiMETnearJet_Puppi", "vecsum_pT_l1l2_allJet", "nJets", "dPhiMETnearJet", "dPhiJet1Jet2", "Jet2_E", "Lep1_pt*sin(Lep1_phi)", "Jet1_E", "dPhiMETlead2Jet_Puppi", "dPhiLep1Lep2", "Lep1_E", "dPhiMETfarJet", "Jet2_eta", "dPhiMETbJet", "dPhiMETfarJet_Puppi", "mLL", "dPhiMETbJet_Puppi", "Lep2_flavor", "Lep2_E", "Jet1_eta", "Lep1_eta", "dPhiMETlead2Jet", "Lep1_flavor", "dPhiLep1bJet", "Lep2_eta", "METsig", "ratio_vecsumpTlep_vecsumpTjet", "looseLeptonVeto"]
    
    #  ~inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET*sin(PFMET_phi)", "MET*cos(PFMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "MT2", "mjj", "nJets", "Jet1_E", "HT", "METunc_Puppi"]
    #  ~inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET*sin(PFMET_phi)", "MET*cos(PFMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "mjj", "Jet1_E"]
    
    inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "MT2", "mjj", "nJets", "Jet1_E", "HT", "METunc_Puppi"]
    
    #  ~print(inputVars[20:30])
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1421genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,1)GeV".format(0.3,0)
    #  ~modelNr = 16
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1017genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,100)GeV".format(0.3,0)
    #  ~modelNr = 13
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211012-0945"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: none".format(0.1,0)
    #  ~modelNr = 1
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211122-1118genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.29,0.00036)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.097075752974773, 0.2899393936036301, -8.01970544548232, 8.154158507965985, 2.402508434459325, 3.5646202688746493, 0.11396439396039146
    #  ~modelNr = 24
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211122-1150genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0
    #  ~modelNr = 25
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211122-1454genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.5,0.00069)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.524903792932268, 0.5, -7.271889484384392, 7.1970043195604125, 8.0, 4.648503137689418, 0.0
    #  ~modelNr = 26
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211125-1531genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV, SeLU".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 using SeLU
    #  ~modelNr = 27

    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211125-1629genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; onehot".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 onehot, without lepFlav
    #  ~modelNr = 28

    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211126-0958genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; onehot".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 onehot, with lepFlav
    #  ~modelNr = 29

    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211126-1158genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; onehot".format(0.43,0.0015)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.40951173838937, 0.4307905292558073, -6.478574338240861, 5.525847797310747, 7.743792064550437, 1.6496830299486087, 0.21534075102851583 
    #  ~modelNr = 30
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1106genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~modelNr = 31
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1106genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~modelNr = 32
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1202genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; MSE loss".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~modelNr = 33
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1431"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; MSE loss, no genMet weights".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~modelNr = 34
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1606genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~#fluctuation plotting 31, 35, 36
    #  ~modelNr = 35
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1637genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~#fluctuation plotting 31, 35, 36
    #  ~modelNr = 36

    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211130-1032genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; MSE loss".format(0.42,0.12)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.724648816372166, 0.42019722418191996, -1.9871840218385974, 7.288464822183116, 3.7077713293386814, 3.098255409797496, 0.10318236454640405
    #  ~modelNr = 37

    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211201-1704genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; standardize".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 standardize
    #  ~modelNr = 38

    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211203-1058genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; standardize v2".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 standardize
    #  ~modelNr = 39
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211210-1108"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,50)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample *min
    #  ~modelNr = 40
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211210-1335"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,25)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample *min
    #  ~modelNr = 41
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211210-1421"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,5)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample *min
    #  ~modelNr = 42
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211210-1613"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,5)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample *1/min
    #  ~modelNr = 43
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211213-1406"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,25)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample after trainsplit
    #  ~modelNr = 44
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211214-1022"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,25), earlystopping".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample after trainsplit, early stopping
    #  ~modelNr = 46
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211214-1516"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,25), earlystopping".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample after trainsplit, early stopping
    #  ~#47: training with oversampled events, 48: without copies, 481: in MeandiffvsgenMet rms(x-mean(x)) instead of rsm(x)
    #  ~modelNr = 481
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211220-1137"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; oversample (0,500,25), earlystopping".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample Gaussian Noise *0.001 after trainsplit, early stopping
    #  ~modelNr = 49
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211221-1219"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; oversample (0,500,25), GN, earlystopping".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 oversample Gaussian Noise corrected *0.001,s after trainsplit, early stopping
    #  ~modelNr = 50
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211222-1155"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; over/undersample (0,500,25)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 under/oversample, early stopping
    #  ~modelNr = 51
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220110-1240"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; undersample (0,500,25)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 undersample, early stopping
    #  ~modelNr = 52
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220110-1529"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; under/oversample (0,500,25)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 under/oversample, early stopping
    #  ~modelNr = 53
    
    #  ~modelPath0 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220110-1600genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMetRew (0,500,25)".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 genMetRe, early stopping
    #  ~modelNr = 54
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220201-1700genMETweighted" #wrong input nodes
    #  ~modelPath1 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220202-1115genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMetRew, 40 inputs".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 genMetRe, early stopping
    #  ~modelNr = 540
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220201-1745genMETweighted" #wrong input nodes
    #  ~modelPath2 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220202-1121genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMetRew, 30 inputs".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 genMetRe, early stopping
    #  ~modelNr = 530
    
    #  ~modelPath3 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220202-1131genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMetRew, 20 inputs".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 genMetRe, early stopping
    #  ~modelNr = 525
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220201-1751genMETweighted" #wrong input nodes
    #  ~modelPath4 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220202-1138genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMetRew, 20 inputs".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 genMetRe, early stopping
    #  ~modelNr = 520
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220202-0951genMETweighted" #wrong input nodes
    #  ~modelPath5 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220202-1146genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMetRew, 10 inputs".format(0.1,0.00046)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 genMetRe, early stopping
    #  ~modelNr = 510
    
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220201-1745genMETweighted" #wrong input nodes
    #  ~modelPath1 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220203-1214genMETweighted"
    #  ~modelPath2 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220203-1227genMETweighted"
    #  ~modelPath3 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220203-1233genMETweighted"
    #  ~modelPath4 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220203-1242genMETweighted"
    #  ~modelPath5 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220203-1304genMETweighted"

    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMetRew, 30 inputs".format(0.1,0.00046)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 genMetRe,  fluctuations
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220204-1344genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; batchsize=50000; genMETrew".format(0.35,0.05)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, np.log(50000), 6.0, 3.0, 0.2  genMETrew, early stopping, epochs 200
    #  ~modelNr = 55
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220207-1403genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMETrew; HT_x, HT_y corrected".format(0.35,0.05)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0  genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~modelNr = 56
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220207-1447genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMETrew; HT_x, HT_y replaced, Glorot Initializer".format(0.35,0.05)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0  genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~modelNr = 57
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220207-1633genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMETrew; HT_x, HT_y replaced, Glorot Initializer".format(0.35,0.05)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0  genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), 32 features
    #  ~modelNr = 632
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-0900genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMETrew; HT_x, HT_y replaced, Glorot Initializer".format(0.35,0.05)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0  genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), 22 features
    #  ~modelNr = 622
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-0918genMETweighted"
    #  ~modelPath1 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1011genMETweighted"
    #  ~modelPath2 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1024genMETweighted"
    #  ~modelPath3 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1050genMETweighted"
    #  ~modelPath4 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1108genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMETrew; HT_x, HT_y replaced, Glorot Initializer".format(0.35,0.05)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0  genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), 11/22 features
    #  ~modelNr = 611

    #  ~modelPath1 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1211genMETweighted"
    #  ~modelPath2 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1221genMETweighted"
    #  ~modelPath3 = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1236genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMETrew; HT_x, HT_y replaced, Glorot Initializer".format(0.35,0.05)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0  genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), 22 features, earlystopping on val_logcosh
    #  ~modelNr = 622
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1447genMETweighted"
    #  ~modelTitle = r"d_out={}; $\lambda$={}; genMETrew; HT_x, HT_y replaced, Glorot Initializer".format(0.35,0.05)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0  genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), early stopping on val_logcosh
    #  ~modelNr = 656
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1645genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.75, logcosh 16.65
    #  ~nInputs = 55
    #  ~modelNr = 755
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1708genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.63, 16.59
    #  ~#416/416 - 16s - loss: 22.8847 - mean_squared_error: 516.0791 - logcosh: 16.9480 - val_loss: 22.6927 - val_mean_squared_error: 496.7951 - val_logcosh: 16.6248
    #  ~nInputs = 32
    #  ~modelNr = 732
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-0944genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.55, 16.66
    #  ~nInputs = 22
    #  ~modelNr = 722
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1032genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.57, 16.86
    #  ~nInputs = 11
    #  ~modelNr = 711
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220208-1708genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.63, 16.59
    #  ~nInputs = 32
    #  ~modelNr = 732
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-0944genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.55, 16.66
    #  ~nInputs = 22
    #  ~modelNr = 722
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1032genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.57, 16.86
    #  ~nInputs = 11
    #  ~modelNr = 711
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1153genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.66, 16.61
    #  ~nInputs = 22
    #  ~modelNr = 7221
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1209genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.8, 16.8
    #  ~nInputs = 22
    #  ~modelNr = 7222
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1225genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin(), loss 22.63, 16.61
    #  ~nInputs = 22
    #  ~modelNr = 7223
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1557genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~nInputs = 32
    #  ~modelNr = 832
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1652genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~nInputs = 22
    #  ~modelNr = 822
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220209-1714genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~nInputs = 11
    #  ~modelNr = 811
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220301-1448genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~nInputs = 21
    #  ~modelNr = 60
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220301-2110genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~nInputs = 21
    #  ~modelNr = 60
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220314-1213genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, without CaloMET_X, CaloMET_Y, but with CaloMET and HT (since still 21 inputs)
    #  ~nInputs = 21
    #  ~modelNr = 62
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220317-1735genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, with new inputs (20)
    #  ~nInputs=int(len(inputVars))
    #  ~modelNr = 63
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220317-1934genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, with new inputs (<20)
    #  ~nInputs=int(len(inputVars))
    #  ~modelNr = 64
    
    modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220324-1725genMETweighted"
    lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, with xy corrected inputs (20)
    nInputs=int(len(inputVars))
    modelNr = 65
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220301-1732genMETweighted"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~nInputs = int(len(inputVars))
    #  ~modelNr = 61
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220302-1023"
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard, no genMETrew, added vecsum_pT_allJet*cos(HT_phi) and vecsum_pT_allJet*sin()
    #  ~nInputs = 21
    #  ~modelNr = 61
    
    
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    nodeFac2 = nodeFacs[int(np.round(nodeFac))]
    modelTitle = r"learnrate={:.2e};dropout={:.2};$\lambda$={:.2e};batchsize={:.2e};".format(np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))))+"\n"+r"n_layer={};nodefactor={};$\alpha$={:.2};n_inputs={};genMETreweighting".format(int(np.round(nLayer)),nodeFac2,alph,nInputs)
    


    print(modelTitle,"\n")
    
    ###############
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211118-1659genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.05)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2
    #  ~modelNr = 20
    ###############
    #  ~nInputs = 22
    #  ~print_targets(dataPath,inputVars[:nInputs],modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211125-1449genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.05)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2, using SeLU
    #  ~modelNr = 28

    plot_Output(dataPath,inputVars[:nInputs],modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    
    #  ~plot_Purity(dataPath,inputVars[:nInputs],modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    #  ~plot_Purity(dataPath,inputVars[:nInputs],modelPath,"TTbar_diLepton","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
    #  ~plot_Output(dataPath,inputVars,modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=False,genMETweighted=True)
    #  ~nInputs=30
    #  ~for (modelPath, modelNr) in [(modelPath1, 6221), (modelPath2, 6222), (modelPath3, 6223)]:
        #  ~print_targets(dataPath,inputVars[:nInputs],modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
        #  ~plot_Output(dataPath,inputVars[:22],modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)
        #  ~plot_Purity(dataPath,inputVars[:22],modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=True,overSample=False,underSample=False,doSmogn=False)

