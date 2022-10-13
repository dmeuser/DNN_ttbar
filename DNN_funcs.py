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

def getInputArray_allBins_nomDistr(year,path,inputVars,targetName,target,treeName,update=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False,cut="(PuppiMET_xy>0)"):
    binE = 500
    binW = 5
    #  ~if (year=="2016_preVFP" or year=="2016_preVFP"):
    if True:
        binE=400
        binW=8
    appendCount = 0
    for subList in [target, ["SF", "genMET", "PtNuNu", "PtNuNu_phi", "Lep1_phi", "Lep2_phi", "PuppiMET_xy", "PuppiMET_xy_phi", "genMET_phi"]]:      # append target variables to be also read in
        for var in subList:
            inputVars.append(var)
            appendCount+=1

    
    if not os.path.exists("input/"+year+"/2D/"):    # create output folder for plots if not available
        os.makedirs("input/"+year+"/2D/")
    
    outputPath="input/"+year+"/2D/"+treeName+"_"+targetName+"_nomDistr.pkl"     # output path of pkl
    
    # option to only use emu events for training (mainly for studies connected to 40 GeV cut)
    if treeName.split("_")[-1]=="emu":
        only_emu=True
        treeName=treeName.replace("_emu","")
    else:
        only_emu=False
    
    # rename pkl if input is normalized or standardized
    #  ~if normalize:
        #  ~outputPath=outputPath.replace("_nomDistr","normalized_nomDistr")
    if standardize:
        outputPath=outputPath.replace("_nomDistr","standardized_nomDistr")
    if overSample:
        outputPath=outputPath.replace("_nomDistr","overSample_nomDistr")
    if underSample:
        outputPath=outputPath.replace("_nomDistr","underSample_nomDistr")
    if doSmogn:
        outputPath=outputPath.replace("_nomDistr","doSmogn_nomDistr")
    
    
    # if update=true new pkl is created from root file (takes much longer than using existing pkl)
    if update:
        print("updating")
        root_file = uproot.open(path)
        events = root_file["ttbar_res100.0;1"][treeName+";1"]
        #  ~cut = "(PuppiMET>0)"    # use only events selected by reco selection
        if only_emu:
            cut+="&(emu==1)"    # use only emu events if selected
        inputFeatures = events.arrays(inputVars,cut,library="pd")   # load inputs from rootfile
        
   
        if genMETweighted:
            bins=list(range(0,binE,binW))
            bins.append(inputFeatures["genMET"].max())
            labels=list(range(1,len(bins)))
            inputFeatures["genMET_binNR"] = pd.cut(inputFeatures["genMET"], bins=bins, labels=labels)
            sampleWeights=compute_sample_weight("balanced",inputFeatures["genMET_binNR"])
            SF_weights = inputFeatures["SF"]
            sampleWeights=sampleWeights*SF_weights
        
        x,y,metVals = inputFeatures[inputVars[:-appendCount]],inputFeatures[target],inputFeatures[inputVars[-appendCount:]]
        
        #split MC data in training, validation and test samples, x=inputs, y=targets, metVals=other Variable, that are not inputs e.g. genMET
        if genMETweighted:
            train_x, test_x, train_y, test_y, train_weights, test_weights, train_metVals, test_metVals = train_test_split(x, y, sampleWeights, metVals, random_state=30, test_size=0.2, train_size=0.8)
            #  ~train_x, test_x, train_y, test_y, train_weights, test_weights, train_metVals, test_metVals = train_test_split(x, y, sampleWeights, metVals, random_state=30, test_size=0.01, train_size=0.99)
            #  ~train_x, test_x, train_y, test_y, train_weights, test_weights, train_metVals, test_metVals = train_test_split(x, y, sampleWeights, metVals, random_state=30, test_size=0.9, train_size=0.1)
            #  ~train_x, train_y, train_weights, train_metVals = x, y, sampleWeights, metVals
            train_x, val_x, train_y, val_y, train_weights, val_weights, train_metVals, val_metVals = train_test_split(train_x, train_y, train_weights, train_metVals, random_state=30, test_size=0.25, train_size=0.75)
            #  ~train_x, val_x, train_y, val_y, train_weights, val_weights, train_metVals, val_metVals = train_test_split(train_x, train_y, train_weights, train_metVals, random_state=30, test_size=0.01, train_size=0.99)
            #  ~train_x, val_x, train_y, val_y, train_weights, val_weights, train_metVals, val_metVals = train_test_split(train_x, train_y, train_weights, train_metVals, random_state=30, test_size=0.9, train_size=0.1)
            #  ~test_x, val_x, test_y, val_y, test_weights, val_weights, test_metVals, val_metVals = train_test_split(x, y, sampleWeights, metVals, random_state=30, test_size=0.01, train_size=0.01)
        else:
            train_x, test_x, train_y, test_y, train_metVals, test_metVals = train_test_split(x, y, metVals, random_state=30, test_size=0.2, train_size=0.8)
            train_x, val_x, train_y, val_y, train_metVals, val_metVals = train_test_split(train_x, train_y, train_metVals, random_state=30, test_size=0.25, train_size=0.75)
        
        if standardize:
            train_x_scaler = StandardScaler()
            train_x = pd.DataFrame(train_x_scaler.fit_transform(train_x.values), index=train_x.index, columns=train_x.columns)
            
            train_y_scaler = StandardScaler()
            train_y = pd.DataFrame(train_y_scaler.fit_transform(train_y.values), index=train_y.index, columns=train_y.columns)
            
            val_x_scaler = StandardScaler()
            val_x = pd.DataFrame(val_x_scaler.fit_transform(val_x.values), index=val_x.index, columns=val_x.columns)
            
            val_y_scaler = StandardScaler()
            val_y = pd.DataFrame(val_y_scaler.fit_transform(val_y.values), index=val_y.index, columns=val_y.columns)
            
            with open("trainedModel_Keras/"+year+"/2D/"+modelName+"_scalers.csv", "ab") as f:
                for scaler in [train_x_scaler, train_y_scaler, val_x_scaler, val_y_scaler]:
                    np.savetxt(f, scaler.mean_, delimiter=",")
                    np.savetxt(f, scaler.scale_, delimiter=",")
        
        binW = 25
        if overSample:
            bins=list(range(0,500,binW))
            bins.append(train_metVals["genMET"].max())
            labels=list(range(1,len(bins)))
            train_metVals["genMET_binNR2"] = pd.cut(train_metVals["genMET"], bins=bins, labels=labels)
            sample_weight=compute_sample_weight("balanced",train_metVals["genMET_binNR2"])
            train_metVals["weight"]=np.array(sample_weight*1/np.min(sample_weight), dtype=int)-1
            train_temp_x = pd.DataFrame()
            train_temp_y = pd.DataFrame()
            for k,i in enumerate(np.unique(train_metVals["weight"])):
                print(str(k)+"/~{}:  ".format(500/binW)+str(i))
                #  ~print(train_metVals["weight"].shape)
                #  ~print([train_x[train_metVals["weight"]==i]])
                if i>0: train_temp_x=train_temp_x.append([train_x[train_metVals["weight"]==i]]*i, ignore_index=True)
                if i>0: train_temp_y=train_temp_y.append([train_y[train_metVals["weight"]==i]]*i, ignore_index=True)
            train_x = train_x.append(train_temp_x, ignore_index=True)
            train_y = train_y.append(train_temp_y, ignore_index=True)
            
        if underSample:
            train_x[target[0]] = train_y[target[0]]
            train_x[target[1]] = train_y[target[1]] 
            metKeys = train_metVals.keys()
            for key in metKeys:
                train_x[key] = train_metVals[key]
            bins=list(range(0,500,binW))
            bins.append(train_metVals["genMET"].max())
            labels=list(range(1,len(bins)))
            train_metVals["genMET_binNR2"] = pd.cut(train_metVals["genMET"], bins=bins, labels=labels)
            sample_weight=compute_sample_weight("balanced",train_metVals["genMET_binNR2"])
            train_metVals["weight"]=np.array(sample_weight)/np.max(sample_weight)
            for k,weightval in enumerate(np.unique(train_metVals["weight"])):
                n_copy = int(weightval)
                print(str(k)+"/~{}:  {};{}".format(500/binW, n_copy,weightval))
                #  ~#print(weightval)
                #  ~#print(train_metVals["weight"].shape)
                #  ~#print([train_x[train_metVals["weight"]==i]])
                if weightval>1: 
                    print("error")
                else: 
                    train_x = train_x.drop(train_x[train_metVals["weight"]==weightval].sample(frac=1-weightval).index)
            train_metVals=pd.DataFrame()
            train_y=pd.DataFrame()
            for key in metKeys:
                train_metVals[key] = train_x[key]
            train_y[target[0]] = train_x[target[0]]
            train_y[target[1]] = train_x[target[1]]
            train_x = train_x.drop([target[0], target[1]]+list(metKeys), axis="columns")

        elif doSmogn:
            noiseFac=0.001
            #  ~print("shape before: ", train_x.shape)
            train_x[target[0]] = train_y[target[0]]
            train_x[target[1]] = train_y[target[1]] 
            #  ~train_x["genMET"] = train_metVals["genMET"]
            bins=list(range(0,500,binW))
            bins.append(train_metVals["genMET"].max())
            labels=list(range(1,len(bins)))
            train_x["genMET_binNR2"] = pd.cut(train_metVals["genMET"], bins=bins, labels=labels)
            sample_weight=compute_sample_weight("balanced",train_x["genMET_binNR2"])
            train_x["weight"]=np.array(sample_weight*1/np.min(sample_weight), dtype=int)-1
            train_temp_1 = train_x.drop(["weight", "genMET_binNR2"], axis="columns")
            for i,groupObj in enumerate(train_x.groupby(train_x["genMET_binNR2"])):
                print(i)
                df = groupObj[1].copy()
                if np.unique(df["weight"].to_numpy()).shape != np.array([1]).shape: print("Something is wrong in overSampling with noise")
                multiFac = df["weight"].to_numpy()[1]
                df=df.drop(["weight", "genMET_binNR2"], axis="columns")
                stdVec = np.std(np.array(df.to_numpy(), dtype=float), axis=0)*noiseFac
                nRows = df.shape[0]
                keyArr = df.keys()
                if multiFac > 0:
                    for i in range(multiFac):
                        train_temp = pd.DataFrame()
                        noiseArr = np.random.normal(loc=0, scale=stdVec, size=(nRows, len(stdVec)))
                        for j, noiseArr1 in enumerate(noiseArr.T):
                            #  ~if j==0: print(df[keyArr[j]], noiseArr, df[keyArr[j]]+noiseArr)
                            train_temp[keyArr[j]] = df[keyArr[j]]+noiseArr1
                            #  ~train_temp[keyArr[j]] = df[keyArr[j]]
                        train_temp_1 = train_temp_1.append(train_temp, ignore_index=True)
            #print(train_temp_1[["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)"]])
            #print(train_x[["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)"]])
            #train_x = train_x.append(train_temp_1, ignore_index=True)
            train_x = train_temp_1.copy()
            #print(train_x[["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)"]])
            train_y = pd.DataFrame()
            train_y[target[0]] = train_x[target[0]]
            train_y[target[1]] = train_x[target[1]]
            train_x = train_x.drop([target[0], target[1]], axis="columns")

        #  ~print("############## abort ##################")
        # write dataframes to pkl
        if genMETweighted:
            datafrNames = ["train_x", "val_x", "test_x", "train_y", "val_y", "test_y", "train_metVals", "val_metVals", "test_metVals", "train_weights", "val_weights", "test_weights"] 
            train_weights, val_weights, test_weights = pd.DataFrame({"train_weights": train_weights}), pd.DataFrame({"val_weights": val_weights}), pd.DataFrame({"test_weights": test_weights})
            for i, datafr in enumerate([train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights]):
                datafr.to_pickle(outputPath[:-4]+"_"+datafrNames[i]+".pkl")
        else:
            datafrNames = ["train_x", "val_x", "test_x", "train_y", "val_y", "test_y", "train_metVals", "val_metVals", "test_metVals"] 
            for i, datafr in enumerate([train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals]):
                datafr.to_pickle(outputPath[:-4]+"_"+datafrNames[i]+".pkl")
    else:
        # read dataframes from existing pkl
        train_x = pd.read_pickle(outputPath[:-4]+"_"+"train_x"+".pkl")
        val_x = pd.read_pickle(outputPath[:-4]+"_"+"val_x"+".pkl")
        test_x = pd.read_pickle(outputPath[:-4]+"_"+"test_x"+".pkl")  
        train_y = pd.read_pickle(outputPath[:-4]+"_"+"train_y"+".pkl")  
        val_y = pd.read_pickle(outputPath[:-4]+"_"+"val_y"+".pkl")  
        test_y = pd.read_pickle(outputPath[:-4]+"_"+"test_y"+".pkl")  
        train_metVals = pd.read_pickle(outputPath[:-4]+"_"+"train_metVals"+".pkl")  
        val_metVals = pd.read_pickle(outputPath[:-4]+"_"+"val_metVals"+".pkl")  
        test_metVals = pd.read_pickle(outputPath[:-4]+"_"+"test_metVals"+".pkl")  
        if genMETweighted:
            train_weights = pd.read_pickle(outputPath[:-4]+"_"+"train_weights"+".pkl")["train_weights"]
            val_weights = pd.read_pickle(outputPath[:-4]+"_"+"val_weights"+".pkl")["val_weights"]
            test_weights = pd.read_pickle(outputPath[:-4]+"_"+"test_weights"+".pkl")["test_weights"]
        #  ~print(train_metVals.keys())
    
    #  ~print(inputFeatures, inputFeatures_std, pd.DataFrame(scaler.inverse_transform(inputFeatures_std), index=inputFeatures.index, columns=inputFeatures.columns))
    
    print("Train with:",outputPath)
    print(train_x.keys())
    #  ~print(inputVars[:-appendCount])
    
    # returns inputs, targets, met variables (and weights if genMETweighting is done) split as 60%/20%/20% in training/validation/test data
    if genMETweighted: return train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights
    else: return train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals
    
    
def getMETarrays(year,path,inputVars,modelPath,treeName,targetName,target,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    
    if genMETweighted: train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, _, _, _ = getInputArray_allBins_nomDistr(year,path,inputVars,targetName,target,treeName,update=True,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn,cut="(PuppiMET_xy>0)&(PtNuNu>0)")
    else: getInputArray_allBins_nomDistr(path,inputVars,targetName,target,treeName,update=True,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn,cut="(PuppiMET_xy>0)&(PtNuNu>0)")

    model = load_model(modelPath+".h5")
    print(train_y.keys())
    
    for (data_x, data_y, data_metVals) in [(train_x, train_y, train_metVals), (val_x, val_y, val_metVals), (test_x, test_y, test_metVals)]:
        
        y_hat_data = model.predict(data_x,use_multiprocessing=False)
        
        print("here:::", data_y.keys())
        
        data_x["DNN_1"]=[row[0] for row in y_hat_data]
        data_x["DNN_2"]=[row[1] for row in y_hat_data]
        data_x[target[0]]=data_y[target[0]]
        data_x[target[1]]=data_y[target[1]]
        
        data_x["DNN_MET_X"]=data_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-data_x["DNN_1"]
        data_x["DNN_MET_Y"]=data_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-data_x["DNN_2"]
        data_metVals["DNN_MET"]=np.sqrt(data_x["DNN_MET_X"]**2+data_x["DNN_MET_Y"]**2)
        
        data_metVals["DNN_dPhiMetNearLep"] = np.array([np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep1_phi"])), np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(data_x["DNN_MET_Y"],data_x["DNN_MET_X"])-data_metVals["Lep2_phi"]))]).min(axis=0)
        
        data_metVals["dPhi_PtNuNuNearLep"] = np.array([np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep1_phi"])),np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep2_phi"]),np.abs(2*np.pi-np.abs(data_metVals["PtNuNu_phi"]-data_metVals["Lep2_phi"]))]).min(axis=0)
        
        data_metVals["dPhi_PuppiNearLep"] = np.array([np.abs(data_metVals["PuppiMET_xy_phi"]-data_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(data_metVals["PuppiMET_xy_phi"]-data_metVals["Lep1_phi"])),np.abs(data_metVals["PuppiMET_xy_phi"]-data_metVals["Lep2_phi"]),np.abs(2*np.pi-np.abs(data_metVals["PuppiMET_xy_phi"]-data_metVals["Lep2_phi"]))]).min(axis=0)
    
    return train_metVals, val_metVals, test_metVals
