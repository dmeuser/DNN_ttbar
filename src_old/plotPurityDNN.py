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

def getMETarrays(path,inputVars,modelPath,treeName,targetName,target):
    appendCount = 0
    for subList in [target, ["PtNuNu", "PtNuNu_phi", "Lep1_phi", "Lep2_phi", "PuppiMET", "PuppiMET_phi"]]:      # append target variables to be also read in
        for var in subList:
            inputVars.append(var)
            appendCount+=1
    
    root_file = uproot.open(path)
    events = root_file["ttbar_res100.0;1"][treeName+";1"]
    #  ~events = root_file["TTbar_diLepton;1"][treeName+";1"]
    cut = "(PuppiMET>0)&(PtNuNu>0)"
    inputFeatures = events.arrays(inputVars,cut,library="pd")
    
    # DNN prediction
    #  ~train_x, val_x, train_y, val_y, train_metVals, val_metVals = train_test_split(inputFeatures[inputVars[:-appendCount]],inputFeatures[target],inputFeatures[inputVars[2-appendCount:]], random_state=30, test_size=0.2, train_size=0.8)
    
    
    train_x, test_x, train_y, test_y, train_metVals, test_metVals = train_test_split(inputFeatures[inputVars[:-appendCount]],inputFeatures[target],inputFeatures[inputVars[2-appendCount:]], random_state=30, test_size=0.2, train_size=0.8)
    train_x, val_x, train_y, val_y, train_metVals, val_metVals = train_test_split(inputFeatures[inputVars[:-appendCount]],inputFeatures[target],inputFeatures[inputVars[2-appendCount:]], random_state=30, test_size=0.25, train_size=0.75)

    model = load_model(modelPath+".h5")
    
    for (data_x, data_y, data_metVals) in ([train_x, val_x, test_x], [train_y, val_y, test_y], [train_metVals, val_metVals, test_metVals]):
        
        y_hat_data = model.predict(data_x,use_multiprocessing=False)
        
        data_x["DNN_1"]=[row[0] for row in y_hat_data]
        data_x["DNN_2"]=[row[1] for row in y_hat_data]
        data_x[target[0]]=data_y[target[0]]
        data_x[target[1]]=data_y[target[1]]
        
        data_x["DNN_MET_X"]=data_x["PuppiMET*cos(PuppiMET_phi)"]-data_x["DNN_1"]
        data_x["DNN_MET_Y"]=data_x["PuppiMET*sin(PuppiMET_phi)"]-data_x["DNN_2"]
        data_metVals["DNN_MET"]=np.sqrt(data_x["DNN_MET_X"]**2+data_x["DNN_MET_Y"]**2)
        
        data_metVals["DNN_dPhiMetNearLep"] = np.array([np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep1_phi"])), np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep2_phi"]))]).min(axis=0)
        
        data_metVals["dPhi_PtNuNuNearLep"] = np.array([np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep1_phi"])),np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep2_phi"]),np.abs(2*np.pi-np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep2_phi"]))]).min(axis=0)
        
        data_metVals["dPhi_PuppiNearLep"] = np.array([np.abs(data_metVals["PuppiMET_phi"]-data_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(data_metVals["PuppiMET_phi"]-data_metVals["Lep1_phi"])),np.abs(data_metVals["PuppiMET_phi"]-data_metVals["Lep2_phi"]),np.abs(2*np.pi-np.abs(data_metVals["PuppiMET_phi"]-data_metVals["Lep2_phi"]))]).min(axis=0)

    #y_hat_train = model.predict(train_x,use_multiprocessing=True)
    #y_hat_val = model.predict(val_x,use_multiprocessing=True)
    
    #  ~train_x["DNN_1"]=[row[0] for row in y_hat_train]
    #  ~train_x["DNN_2"]=[row[1] for row in y_hat_train]
    #  ~train_x[target[0]]=train_y[target[0]]
    #  ~train_x[target[1]]=train_y[target[1]]
    
    #  ~val_x["DNN_1"]=[row[0] for row in y_hat_val]
    #  ~val_x["DNN_2"]=[row[1] for row in y_hat_val]
    #  ~val_x[target[0]]=val_y[target[0]]
    #  ~val_x[target[1]]=val_y[target[1]]
    
    #  ~train_x["DNN_MET_X"]=train_x["PuppiMET*cos(PuppiMET_phi)"]-train_x["DNN_1"]
    #  ~train_x["DNN_MET_Y"]=train_x["PuppiMET*sin(PuppiMET_phi)"]-train_x["DNN_2"]
    #  ~train_metVals["DNN_MET"]=np.sqrt(train_x["DNN_MET_X"]**2+train_x["DNN_MET_Y"]**2)
    #  ~val_x["DNN_MET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x["DNN_1"]
    #  ~val_x["DNN_MET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x["DNN_2"]
    #  ~val_metVals["DNN_MET"]=np.sqrt(val_x["DNN_MET_X"]**2+val_x["DNN_MET_Y"]**2)
    
    #  ~train_metVals["DNN_dPhiMetNearLep"] = np.array([np.abs(np.arctan2(train_x["DNN_MET_Y"],train_x["DNN_MET_X"])-train_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(train_x["DNN_MET_Y"],train_x["DNN_MET_X"])-train_metVals["Lep1_phi"])), np.abs(np.arctan2(train_x["DNN_MET_Y"],train_x["DNN_MET_X"])-train_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(train_x["DNN_MET_Y"],train_x["DNN_MET_X"])-train_metVals["Lep2_phi"]))]).min(axis=0)
    #  ~val_metVals["DNN_dPhiMetNearLep"] = np.array([np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep1_phi"])), np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep2_phi"]))]).min(axis=0)
    #  ~#val_metVals["DNN_dPhiMetNearLep"] = np.min(np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep1_phi"], np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep2_phi"]))
    
    #  ~train_metVals["dPhi_PtNuNuNearLep"] = np.array([np.abs(train_metVals["PtNuNu_phi"]-train_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(train_metVals["PtNuNu_phi"]-train_metVals["Lep1_phi"])),np.abs(train_metVals["PtNuNu_phi"]-train_metVals["Lep2_phi"]),np.abs(2*np.pi-np.abs(train_metVals["PtNuNu_phi"]-train_metVals["Lep2_phi"]))]).min(axis=0)
    #  ~val_metVals["dPhi_PtNuNuNearLep"] = np.array([np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep1_phi"])),np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep2_phi"]))]).min(axis=0)
    
    #  ~train_metVals["dPhi_PuppiNearLep"] = np.array([np.abs(train_metVals["PuppiMET_phi"]-train_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(train_metVals["PuppiMET_phi"]-train_metVals["Lep1_phi"])),np.abs(train_metVals["PuppiMET_phi"]-train_metVals["Lep2_phi"]),np.abs(2*np.pi-np.abs(train_metVals["PuppiMET_phi"]-train_metVals["Lep2_phi"]))]).min(axis=0)
    #  ~val_metVals["dPhi_PuppiNearLep"] = np.array([np.abs(val_metVals["PuppiMET_phi"]-val_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(val_metVals["PuppiMET_phi"]-val_metVals["Lep1_phi"])),np.abs(val_metVals["PuppiMET_phi"]-val_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(val_metVals["PuppiMET_phi"]-val_metVals["Lep2_phi"]))]).min(axis=0)
    
    return train_metVals, val_metVals, test_metVals
    #  ~return train_metVals, []
    


# function to derive different control performance plots based on trained model
def plot_Output(dataPath,inputVars,modelPath,treeName,targetName,target,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False):
    train_metVals, val_metVals, test_metVals = getMETarrays(dataPath,inputVars,modelPath,treeName,targetName,target)
    #  ~print(len(train_metVals["dPhi_PtNuNuNearLep"]), len(train_metVals["DNN_dPhiMetNearLep"]), len(train_metVals["PtNuNu"]))
    #  ~print(train_metVals, val_metVals)
    
    metBins = np.array([0,40,80,120,160,230,400])
    dphiBins = np.array([0,0.7,1.4,3.15])
    
    # Purity and Stability plots for training
    
    for (data_metVals, sampName) in ([train_metVals, val_metVals, test_metVals], ["train", "val", "test"]):
        
        met_gen = np.clip(data_metVals["PtNuNu"], metBins[0], metBins[-1])
        dphi_gen = data_metVals["dPhi_PtNuNuNearLep"]
        
        met_reco_DNN = np.clip(data_metVals["DNN_MET"], metBins[0], metBins[-1])
        dphi_reco_DNN = data_metVals["DNN_dPhiMetNearLep"]

        met_reco_Puppi = np.clip(data_metVals["PuppiMET"], metBins[0], metBins[-1])
        dphi_reco_Puppi = data_metVals["dPhi_PuppiNearLep"]
        
        for (met_reco, dphi_reco, dnnName) in ([met_reco_DNN, met_reco_Puppi], [dphi_reco_DNN, dphi_reco_Puppi], ["DNN", "Puppi"]):
            
            histo2D_Gen, xedges, yedges = np.histogram2d(met_gen, dphi_gen, bins=[metBins, dphiBins])
            histo2D_Reco, xedges, yedges = np.histogram2d(met_reco, dphi_reco, bins=[metBins, dphiBins])
            histo2D_Both = np.copy(histo2D_Gen)
            
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
            
            Xbins, Ybins = np.meshgrid(metBins, dphiBins)
            metBinsC = (metBins[1:]+metBins[:-1])*0.5
            dphiBinsC = (dphiBins[1:]+dphiBins[:-1])*0.5
            
            fig, ax = plt.subplots(1,1)
            fig.suptitle("{}, {} sample\n, {}".format(dnnName, sampName, title), fontsize=12, ha="left", x=0.1, y=0.99)
            mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Reco)
            for i,phiarr in enumerate(histo2D_Both/histo2D_Reco):
                for j,vali in enumerate(phiarr):
                    ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
                    tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Reco[i,j])
                    ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("purity")
            ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
            ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
            plt.savefig("outputComparison/2018/2D/{model}/{dnnName}_purity_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))
            
            fig, ax = plt.subplots(1,1)
            fig.suptitle("{}, {} sample\n, {}".format(dnnName, sampName, title), fontsize=12, ha="left", x=0.1, y=0.99)
            mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Gen)
            for i,phiarr in enumerate(histo2D_Both/histo2D_Gen):
                for j,vali in enumerate(phiarr):
                    ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
                    tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Gen[i,j])
                    ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
            cbar = fig.colorbar(mesh1, ax=ax)
            cbar.set_label("stability")
            ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
            ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
            plt.savefig("outputComparison/2018/2D/{model}/{dnnName}_stability_{sample}_{nr}.pdf".format(model=modelPath.split("/")[-1], dnnName=dnnName, sample=sampName, nr=str(modelNr)))
    
    metdphi = []
    metdphi.append([met_gen, met_reco, dphi_gen, dphi_reco])
    
    histo2D_Gen, xedges, yedges = np.histogram2d(met_gen, dphi_gen, bins=[metBins, dphiBins])
    histo2D_Reco, xedges, yedges = np.histogram2d(met_reco, dphi_reco, bins=[metBins, dphiBins])
    histo2D_Both = np.copy(histo2D_Gen)
    
    for i in range(len(metBins)-1):
        for j in range(len(dphiBins)-1):
            temp1 = np.where((met_gen>metBins[i]) & (met_gen<=metBins[i+1]) & (dphi_gen>dphiBins[j]) & (dphi_gen<=dphiBins[j+1]), True, False)
            temp2 = np.where((met_reco>metBins[i]) & (met_reco<=metBins[i+1]) & (dphi_reco>dphiBins[j]) & (dphi_reco<=dphiBins[j+1]), True, False)
            histo2D_Both[i,j] = sum(np.where(temp1 & temp2, 1, 0))
    
    histo2D_Gen=histo2D_Gen.T
    histo2D_Reco=histo2D_Reco.T
    histo2D_Both=histo2D_Both.T
    
    print("DNN Reco train Last Bin Count: {}".format(histo2D_Reco[-1,-1]))
    print("DNN Both train Last Bin Count: {}".format(histo2D_Both[-1,-1]))
    
    Xbins, Ybins = np.meshgrid(metBins, dphiBins)
    metBinsC = (metBins[1:]+metBins[:-1])*0.5
    dphiBinsC = (dphiBins[1:]+dphiBins[:-1])*0.5
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("DNN, training sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Reco)
    for i,phiarr in enumerate(histo2D_Both/histo2D_Reco):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Reco[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("purity")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/purity_train_"+str(modelNr)+".pdf")
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("DNN, training sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Gen)
    for i,phiarr in enumerate(histo2D_Both/histo2D_Gen):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Gen[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("stability")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/stability_train_"+str(modelNr)+".pdf")
    
    
    # Purity and Stability plots for validation
    
    met_gen_val = np.clip(val_metVals["PtNuNu"], metBins[0], metBins[-1])
    met_reco_val = np.clip(val_metVals["DNN_MET"], metBins[0], metBins[-1])
    dphi_gen_val = val_metVals["dPhi_PtNuNuNearLep"]
    dphi_reco_val = val_metVals["DNN_dPhiMetNearLep"]
    
    histo2D_Gen_val, xedges, yedges = np.histogram2d(met_gen_val, dphi_gen_val, bins=[metBins, dphiBins])
    histo2D_Reco_val, xedges, yedges = np.histogram2d(met_reco_val, dphi_reco_val, bins=[metBins, dphiBins])
    histo2D_Both_val = np.copy(histo2D_Gen_val)
    
    for i in range(len(metBins)-1):
        for j in range(len(dphiBins)-1):
            temp1 = np.where((met_gen_val>metBins[i]) & (met_gen_val<=metBins[i+1]) & (dphi_gen_val>dphiBins[j]) & (dphi_gen_val<=dphiBins[j+1]), True, False)
            temp2 = np.where((met_reco_val>metBins[i]) & (met_reco_val<=metBins[i+1]) & (dphi_reco_val>dphiBins[j]) & (dphi_reco_val<=dphiBins[j+1]), True, False)
            histo2D_Both_val[i,j] = sum(np.where(temp1 & temp2, 1, 0))
    
    histo2D_Gen_val=histo2D_Gen_val.T
    histo2D_Reco_val=histo2D_Reco_val.T
    histo2D_Both_val=histo2D_Both_val.T
    
    
    print("DNN Reco val Last Bin Count: {}".format(histo2D_Reco_val[-1,-1]))
    print("DNN Both val Last Bin Count: {}".format(histo2D_Both_val[-1,-1]))
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("DNN, validation sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both_val/histo2D_Reco_val)
    for i,phiarr in enumerate(histo2D_Both_val/histo2D_Reco_val):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both_val[i,j]+1/histo2D_Reco_val[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("purity")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/purity_val_"+str(modelNr)+".pdf")
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("DNN, validation sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both_val/histo2D_Gen_val)
    for i,phiarr in enumerate(histo2D_Both_val/histo2D_Gen_val):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both_val[i,j]+1/histo2D_Gen_val[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("stability")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/stability_val_"+str(modelNr)+".pdf")

    
    
    # same for puppi as comparison
    # Purity and Stability plots for training (Puppi)
    
    met_gen = np.clip(train_metVals["PtNuNu"], metBins[0], metBins[-1])
    met_reco = np.clip(train_metVals["PuppiMET"], metBins[0], metBins[-1])
    dphi_gen = train_metVals["dPhi_PtNuNuNearLep"]
    dphi_reco = train_metVals["dPhi_PuppiNearLep"]
    
    histo2D_Gen, xedges, yedges = np.histogram2d(met_gen, dphi_gen, bins=[metBins, dphiBins])
    histo2D_Reco, xedges, yedges = np.histogram2d(met_reco, dphi_reco, bins=[metBins, dphiBins])
    histo2D_Both = np.copy(histo2D_Gen)
    
    for i in range(len(metBins)-1):
        for j in range(len(dphiBins)-1):
            temp1 = np.where((met_gen>metBins[i]) & (met_gen<=metBins[i+1]) & (dphi_gen>dphiBins[j]) & (dphi_gen<=dphiBins[j+1]), True, False)
            temp2 = np.where((met_reco>metBins[i]) & (met_reco<=metBins[i+1]) & (dphi_reco>dphiBins[j]) & (dphi_reco<=dphiBins[j+1]), True, False)
            histo2D_Both[i,j] = sum(np.where(temp1 & temp2, 1, 0))
    
    histo2D_Gen=histo2D_Gen.T
    histo2D_Reco=histo2D_Reco.T
    histo2D_Both=histo2D_Both.T
    
    print("Puppi Reco train Last Bin Count: {}".format(histo2D_Reco[-1,-1]))
    print("Puppi Both train Last Bin Count: {}".format(histo2D_Both[-1,-1]))
    
    Xbins, Ybins = np.meshgrid(metBins, dphiBins)
    metBinsC = (metBins[1:]+metBins[:-1])*0.5
    dphiBinsC = (dphiBins[1:]+dphiBins[:-1])*0.5
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("Puppi, training sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Reco)
    for i,phiarr in enumerate(histo2D_Both/histo2D_Reco):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Reco[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("purity")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Puppi_purity_train_"+str(modelNr)+".pdf")
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("Puppi, training sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both/histo2D_Gen)
    for i,phiarr in enumerate(histo2D_Both/histo2D_Gen):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both[i,j]+1/histo2D_Gen[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("stability")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Puppi_stability_train_"+str(modelNr)+".pdf")
    
    
    # Purity and Stability plots for validation (Puppi)
    
    met_gen_val = np.clip(val_metVals["PtNuNu"], metBins[0], metBins[-1])
    met_reco_val = np.clip(val_metVals["PuppiMET"], metBins[0], metBins[-1])
    dphi_gen_val = val_metVals["dPhi_PtNuNuNearLep"]
    dphi_reco_val = val_metVals["dPhi_PuppiNearLep"]
    
    histo2D_Gen_val, xedges, yedges = np.histogram2d(met_gen_val, dphi_gen_val, bins=[metBins, dphiBins])
    histo2D_Reco_val, xedges, yedges = np.histogram2d(met_reco_val, dphi_reco_val, bins=[metBins, dphiBins])
    histo2D_Both_val = np.copy(histo2D_Gen_val)
    
    for i in range(len(metBins)-1):
        for j in range(len(dphiBins)-1):
            temp1 = np.where((met_gen_val>metBins[i]) & (met_gen_val<=metBins[i+1]) & (dphi_gen_val>dphiBins[j]) & (dphi_gen_val<=dphiBins[j+1]), True, False)
            temp2 = np.where((met_reco_val>metBins[i]) & (met_reco_val<=metBins[i+1]) & (dphi_reco_val>dphiBins[j]) & (dphi_reco_val<=dphiBins[j+1]), True, False)
            histo2D_Both_val[i,j] = sum(np.where(temp1 & temp2, 1, 0))
    
    histo2D_Gen_val=histo2D_Gen_val.T
    histo2D_Reco_val=histo2D_Reco_val.T
    histo2D_Both_val=histo2D_Both_val.T
    
    print("Puppi Reco val Last Bin Count: {}".format(histo2D_Reco_val[-1,-1]))
    print("Puppi Both val Last Bin Count: {}".format(histo2D_Both_val[-1,-1]))
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("Puppi, validation sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both_val/histo2D_Reco_val)
    for i,phiarr in enumerate(histo2D_Both_val/histo2D_Reco_val):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both_val[i,j]+1/histo2D_Reco_val[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("purity")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Puppi_purity_val_"+str(modelNr)+".pdf")
    
    fig, ax = plt.subplots(1,1)
    fig.suptitle("Puppi, validation sample\n"+title, fontsize=12, ha="left", x=0.1, y=0.99)
    mesh1 = ax.pcolormesh(Xbins, Ybins, histo2D_Both_val/histo2D_Gen_val)
    for i,phiarr in enumerate(histo2D_Both_val/histo2D_Gen_val):
        for j,vali in enumerate(phiarr):
            ax.text(metBinsC[j]-20, dphiBinsC[i], "{:.2f}".format(vali)[1:], fontsize=16, color="red", fontweight="bold")
            tempErr = vali*np.sqrt(1/histo2D_Both_val[i,j]+1/histo2D_Gen_val[i,j])
            ax.text(metBinsC[j]-20, dphiBinsC[i]-0.15, r"$\pm$"+"{:.2f}".format(tempErr)[1:], fontsize=11, color="red")
    cbar = fig.colorbar(mesh1, ax=ax)
    cbar.set_label("stability")
    ax.set_ylabel(r"$|\Delta\phi|(p_T^{\nu\nu}, $nearest $\ell)$")
    ax.set_xlabel(r"$p_T^{\nu\nu}$ (GeV)")
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Puppi_stability_val_"+str(modelNr)+".pdf")


    
#############################################################

if __name__ == "__main__":
    # Define input data path
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v04/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"
    #  ~dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v04/minTrees/100.0/Nominal/TTbar_diLepton_merged.root"

    # Define Input Variables
    inputVars = ["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211012-0945"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: none".format(0.1,0)
    #  ~modelNr = 2
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211118-1659genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.05)
    #  ~modelNr = 22
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1509genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,1)GeV".format(0.4,0.05)
    #  ~modelNr = 17
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211125-1228genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV, SeLU".format(0.1,0.00046)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0 using SeLU
    #  ~modelNr = 27
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211122-1454genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.5,0.00069)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.524903792932268, 0.5, -7.271889484384392, 7.1970043195604125, 8.0, 4.648503137689418, 0.0
    #  ~modelNr = 26
    
    modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1106genMETweighted"
    modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.00034)
    #lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    modelNr = 31
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1106genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; genMET weight binning: (0,500,5)GeV".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~modelNr = 32
    
    #  ~modelPath = "trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1202genMETweighted"
    #  ~modelTitle = r"dropout={}; $\lambda$={}; MSE loss".format(0.35,0.00034)
    #  ~#lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~modelNr = 33

    
    plot_Output(dataPath,inputVars,modelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,modelPath,"TTbar_diLepton","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],modelTitle,modelNr,updateInput=True,genMETweighted=True)
