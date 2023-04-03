from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
from subprocess import call
import argparse
from os.path import isfile
import sys
import struct
import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#  ~os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU, BatchNormalization
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras.constraints import NonNeg
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# from tensorflow.keras.models import load_model

import scipy.spatial.distance as ssd
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.colors as mcol
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler,StandardScaler
# from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
import shap
import seaborn as sns
# import tensorflow as tf
from DNN_funcs import getInputArray_allBins_nomDistr, getMETarrays



# ----------------------------------------------------------------------
# Following functions meanFunc... and rmsFunc... are for evaluation of rms and mean of difference between genMET and recoMET while taking SFweights into account; Input "x" is a 2 dimensional array, whith one column being the difference, one column being the SF weights
# def meanFunc2d(x):
    # y = np.array(list(map(np.array, np.array(x))))
    # res = np.average(y[:,0], weights=y[:,1])
    # return res

# def rmsFunc2d(x):
    # av = meanFunc2d(x)
    # y = np.array(list(map(np.array, np.array(x))))
    # res = np.sqrt(np.average((y[:,0]-av)**2, weights=y[:,1]))
    # return res

# def meanErr2d(x):
    # y = np.array(list(map(np.array, np.array(x))))
    # return 2*rmsFunc2d(x)/np.sqrt(y.shape[0])

# def rmsFUNC(x):
    # return np.sqrt(np.mean(np.square(x-np.mean(x))))

# def meanErr(x):
    # return 2*np.std(x)/np.sqrt(len(x))

# def dPhi_pd(phi_1,phi_2):
    # dPhi_arr = np.array([(phi_1-phi_2), (2*np.pi+(phi_1-phi_2)), (-2*np.pi+(phi_1-phi_2))])
    # return dPhi_arr.flatten()[np.arange(np.shape(dPhi_arr)[1]) + np.abs(dPhi_arr).argmin(axis=0)*dPhi_arr.shape[1]]

# def mean_std_SF(values,SFs):
    # mean = np.around(np.average(values, weights=SFs),decimals=2)
    # std = np.around(np.sqrt(np.average((values-mean)**2, weights=SFs)),decimals=2)
    # return mean,std
# ----------------------------------------------------------------------


# def gaus(x,a,x0,sigma):
    # return a*np.exp(-(x-x0)**2/(2*sigma**2))


def get_inputs(year,pathNameDict,inputVars,modelPath,targetName,target,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False,noTrainSplitting=False,testRun=False):
    
    if not os.path.exists("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]):    # create output folder for plots if not available
        os.makedirs("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1])
    if not os.path.exists("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/test"):
        os.makedirs("outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+"/test")
    

    if genMETweighted:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights = getInputArray_allBins_nomDistr(year,pathNameDict,inputVars,targetName,target,update=updateInput,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn,noTrainSplitting=noTrainSplitting,testRun=testRun)
    else:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(year,pathNameDict,inputVars,targetName,target,update=updateInput,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn,noTrainSplitting=noTrainSplitting,testRun=testRun)
    
    
    if standardize:
        train_x_scaler = StandardScaler()
        train_x_std = pd.DataFrame(train_x_scaler.fit_transform(train_x.values), index=train_x.index, columns=train_x.columns)
        
        train_y_scaler = StandardScaler()
        train_y_std = pd.DataFrame(train_y_scaler.fit_transform(train_y.values), index=train_y.index, columns=train_y.columns)
        
        val_x_scaler = StandardScaler()
        val_x_std = pd.DataFrame(val_x_scaler.fit_transform(val_x.values), index=val_x.index, columns=val_x.columns)
        
        val_y_scaler = StandardScaler()
        val_y_std = pd.DataFrame(val_y_scaler.fit_transform(val_y.values), index=val_y.index, columns=val_y.columns)
        

    
    # additional vars have to be also denfined in DNN_funcs.getInputArray_allBins_nomDistr
    additionalVars = ["PuppiMET_xy_phi","MET_xy_phi","genMET_phi","SF","PuppiMET_xy","MET_xy","DeepMET_reso","DeepMET_reso_phi","DeepMET_resp","DeepMET_resp_phi","n_Interactions"]
    
    # reads these variables into the data frames
    
    for var in additionalVars:
        train_x[var] = train_metVals[var]
      
    for var in [additionalVars]:
        val_x[var] = val_metVals[var]

    for var in additionalVars:
        test_x[var] = test_metVals[var]

    return train_x, val_x, test_x


def plot_Output(year,pathNameDict,inputVars,modelPath,targetName,target,correctedValues,title,modelNr,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False,sampleLabel="",noTrainSplitting=False,testRun=False):
    
    # Load Dataframes for plotting
    train_x, val_x, test_x = get_inputs(year,pathNameDict,inputVars,modelPath,targetName,target,updateInput=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn,noTrainSplitting=noTrainSplitting,testRun=testRun)
     
    dataPath = list(pathNameDict.values())[0]   # Use first path oft dict as standard (usually the same for all samples)
    treeName = "_".join(list(pathNameDict.keys()))  # Concat dataset strings, which are used for plotting
    
    #  ~if(dataPath.split("/")[-2] != "Nominal" and treeName == "TTbar_diLepton"):  #renaming required for plotting with systematic shifts
    if(dataPath.split("/")[-2] != "Nominal"):  #renaming required for plotting with systematic shifts
        treeName += dataPath.split("/")[-2]
    
    # Additional string to save path in case of test running or different sample
    testRunString = ""
    if testRun:
        testRunString ="/test"
    if treeName.find("TTbar_amcatnlo") == -1:
        testRunString += "/"+treeName
    
    # Check if path for storing is available and create if not
    storingPath = "outputComparison/"+year+"/2D/"+modelPath.split("/")[-1]+testRunString
    if not os.path.exists(storingPath):
        os.makedirs(storingPath)
    
    
    # define three different samples similar to splitting in DNN training, usually only done for TTBar_diLepton
        evalArr = [data_x]
        sampleNames=["tot"]
    else: 
        evalArr = [train_x, val_x, test_x]
        sampleNames=["train", "val", "test"]
    
    for i_sample,data_x in enumerate(evalArr):
        SF_weights = data_x["SF"]
        
        
        # Plot resolution in pT
        min_x=0
        max_x=400
        binsN=200
        
        plt.figure()
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.hist(data_x["DeepMET_reso"],alpha=0.7,bins=binsN,range=(min_x,max_x),density=False,weights=data_x["SF"], color="orange", histtype=u'step', linewidth=2.)
        plt.step([-100,-99], [0., 0.], alpha=0.7, color="orange",label="DeepMET resolution tune", linewidth=2.)

        plt.axvline(0, color="black", linewidth=1)
        plt.xlim(min_x,max_x) # x-axis limits
        ax = plt.gca()
        ax.text(0.05, 0.95, sampleLabel, transform=ax.transAxes)
        plt.xlabel(r"$p_{\rm T}^{\rm miss}$ (GeV)")
        plt.ylabel("Events")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)
        ax.get_yaxis().get_offset_text().set_x(-0.075)
        #  ~plt.ylabel("Normalized Counts")
        plt.legend()
        makeCMStitle(year)
        plt.tight_layout(pad=0.1)
        plt.savefig(storingPath+"/Spectrum_pTmiss_"+sampleNames[i_sample]+treeName+"_"+str(modelNr)+".pdf")
        plt.clf()


def makeCMStitle(year):
    if year=="2016_preVFP":
        lumi="19.5"
    elif year=="2016_postVFP":
        lumi="16.8"
    elif year=="2017":
        lumi="41.5"
    else:
        lumi="59.8"
    ax = plt.gca()
    ax.text(0.,1.,"CMS",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', weight="bold", fontsize=14)
    ax.text(0.,1.,r"           $\,$Simulation$\,\bullet\,$Private work",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    #  ~ax.text(0.,1.,r"Private Work (CMS Simulation)",transform=ax.transAxes,horizontalalignment='left',verticalalignment='bottom', style="italic", fontsize=10, color=(0.3,0.3,0.3))
    ax.text(1.,1.,lumi+r"$\,$fb${}^{-1}\,(13\,$TeV)",transform=ax.transAxes,horizontalalignment='right',verticalalignment='bottom',fontsize=12)


    
#############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018 (default)", default="2018")
    parser.add_argument('--version', type=str, help="treeVersion, such as v07 (default)", default="v08")
    parser.add_argument('--mode', type=int, help="Runninge mode, 1 (default) for plotting performance, 2 for plotting purity, 3 for printing target values", default=1)
    parser.add_argument('--test', default=False, help="Run with fraction of data to test", action='store_true')
    
    args = parser.parse_args()
    year, version, mode = args.year, args.version, args.mode
 

    #  ~dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/Nominal/".format(year=year, version=version)
    dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/removeMetCut/".format(year=year, version=version)
    
    ########################################
    # normal sample for DNN performance evaluation:
    sampleNameRoot = "TTbar_amcatnlo"
    dataPath += sampleNameRoot+"_merged.root"
    sampleLabel = "${t}\overline{{t}}$"
    noTrainSplitting = False
    
    # nominal dileptonic ttbar sample:
    #  ~sampleNameRoot = "TTbar_diLepton"
    #  ~dataPath += sampleNameRoot+"_merged.root"
    #  ~sampleLabel = "${t}\overline{{t}}$"
    #  ~noTrainSplitting = True
    
    # background and BSM processes: 
    #  ~sampleNameRoot = "DrellYan_NLO"
    #  ~dataPath += sampleNameRoot+"_merged.root"
    #  ~sampleLabel = "Drell-Yan"
    #  ~noTrainSplitting = True
    
    #  ~sampleNameRoot = "TTbar_diLepton_tau"
    #  ~dataPath += sampleNameRoot+"_merged.root"
    #  ~sampleLabel = "${t}\overline{{t}}$ tau"
    #  ~noTrainSplitting = True
    
    #  ~sampleNameRoot = "TTbar_singleLepton"
    #  ~dataPath += sampleNameRoot+"_merged.root"
    #  ~sampleLabel = r"${t}\overline{{t}}$ single $\ell$"
    #  ~noTrainSplitting = True
    
    #  ~sampleNameRoot = "SingleTop"
    #  ~dataPath += sampleNameRoot+"_merged.root"
    #  ~sampleLabel = "Single t"
    #  ~noTrainSplitting = True
    
    #  ~sampleNameRoot = "T2tt_525_350"
    #  ~dataPath += sampleNameRoot+"_1.root"
    #  ~sampleLabel = "T2tt_525_350"
    #  ~noTrainSplitting = True
    
    
    pathNameDict = { sampleNameRoot : dataPath}
    

    
    inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "mjj", "Jet1_E", "HT", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)"] # Finalized set of input variables
    
    
    # Define targets
    targets = ["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"]
    correctedValues = ["PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)"]

    # Defining a model ID (modelNR) to be able to differentiate the plot names from other iterations
    modelNr = 152
    
    # Defining the model pa
    #2018
    modelPath = "trainedModel_Keras/"+year+"/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20230113-1006genMETweighted"
    
    #2017
    #  ~modelPath = "trainedModel_Keras/"+year+"/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2017_20220621-1425genMETweighted"
    #  ~modelPath = "trainedModel_Keras/"+year+"/2D/Inlusive_noMetCut_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2017_20230206-0940genMETweighted"
    
    #2016_preVFP
    #  ~modelPath = "trainedModel_Keras/"+year+"/2D/Inlusive_noMetCut_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2016_preVFP_20230206-1026genMETweighted"
    
    #2016_preVFP
    #  ~modelPath = "trainedModel_Keras/"+year+"/2D/Inlusive_noMetCut_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2016_postVFP_20230206-1041genMETweighted"
    
    nInputs=int(len(inputVars))
    
    netStruct = {"alph": 0.11556803605322355, "batch": 6.501144102888323, "dout": 0.35075846582000303, "lamb": -5.941028038499022, "lr": -7.729770703881016, "nLayer": 2.2186773553565198, "nodeFac": 4.424425111826699} #genMETrew 0,400,8, final inputs
    alph, batch, dout, lamb, lr, nLayer, nodeFac = netStruct["alph"], netStruct["batch"], netStruct["dout"], netStruct["lamb"], netStruct["lr"], netStruct["nLayer"], netStruct["nodeFac"] 

    # Printing the DNN structure
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    modelTitle = r"learnrate={:.2e};dropout={:.2};$\lambda$={:.2e};batchsize={:.2e};".format(np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))))+"\n"+r"n_layer={};nodefactor={};$\alpha$={:.2};n_inputs={};genMETreweighting".format(int(np.round(nLayer)),nodeFacs[int(np.round(nodeFac))],alph,nInputs)
    # print(modelTitle,"\n")
    

        # Creating Plots of the Network performance, using the same training, validation and test sample as in the training process
    plot_Output(year,pathNameDict,inputVars[:nInputs],modelPath,"diff_xy",targets,correctedValues,modelTitle,modelNr,updateInput=True,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False,sampleLabel=sampleLabel,noTrainSplitting=noTrainSplitting,testRun=args.test)
    
