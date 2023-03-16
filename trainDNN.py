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

from tensorflow.keras import layers
from tensorflow.keras import Sequential, regularizers, activations
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
#  ~from imblearn.over_sampling import RandomOverSampler

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
import smogn

from DNN_funcs import getInputArray_allBins_nomDistr
from plotPerformanceDNN import plot_Output, plot_Purity
from CustomEarlyStopping import EarlyStoppingCombined

#  ~os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def rmsFUNC(x):
    return np.sqrt(np.mean(np.square(x)))

def meanErr(x):
    return 2*np.std(x)/np.sqrt(len(x))
    
# model roughly taken from https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=HIG-18-027
def bJetRegression_Model(lr=0.0001, dout=0.3, lamb=0.05, nLayer=6, nodeFac=1., alph=0.2, nInputs=54):
    #  ~dout = 0.3
    regLamb = lamb
    nodes = np.array(np.array([128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024])*nodeFac, dtype=int)
    initializer = tf.keras.initializers.GlorotNormal()
    
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(nInputs, input_dim=nInputs, kernel_regularizer=regularizers.l2(regLamb)))
    #  ~model.add(Dense(54, input_dim=54))
    model.add(BatchNormalization())

    for nodeNr in nodes[:nLayer][::-1]:
        model.add(Dense(nodeNr, kernel_regularizer=regularizers.l2(regLamb), kernel_initializer=initializer))
        #  ~model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Dropout(dout))
        model.add(LeakyReLU(alpha=alph))
    
    
    model.add(Dense(2, kernel_initializer='normal', activation='linear'))
    #  ~model.compile(loss="logcosh", optimizer=Adam(lr=lr),metrics=['mean_squared_error','mean_absolute_percentage_error',"logcosh"])
    model.compile(loss="logcosh", optimizer=Adam(lr=lr),metrics=['mean_squared_error',"logcosh"])
    return model

# function to train model
def trainKeras(year,pathNameDict,inputVars,name,targetName,target,correctedValues,lr,dout,lamb,batch,nLayer,nodeFac,alph,nInputs,updateInput=False,permuationImportance=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False,plotOutputs=False,modelNr=0,plotPurity=False):
    
    
    inputVarsV2 = inputVars.copy()
    print(inputVarsV2)
    
    # set number of epochs and batchsize
    #  ~epochs = 100
    epochs = 400
    #  ~batch_size = 5000
    batch_size = batch
    
    stringStart=datetime.datetime.now().strftime("%Y%m%d-%H%M")

    modelName=name+"_"+"_"+targetName+"_"+year+"_"+stringStart      # time added to name of trained model 
    
    if genMETweighted:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights = getInputArray_allBins_nomDistr(year,pathNameDict,inputVars,targetName,target,update=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
        modelName+="genMETweighted"
    else:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(year,pathNameDict,inputVars,targetName,target,update=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    print("..........................\n\n", train_x.shape)
    
    #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)   # could be used for early stopping of training
    logdir="./logs/"+year+"/2D/"+modelName
    tensorboard_callback = TensorBoard(log_dir=logdir)      # setup tensorboard to log training progress
    earlystopping = EarlyStoppingCombined(patience=10, percentage=2, percentagePatience=10, generalizationPatience=10) # Using own earlystopping function
    
    # setup keras and train model
    my_model = KerasRegressor(build_fn=bJetRegression_Model, lr=lr, dout=dout, lamb=lamb, nLayer=nLayer, nodeFac=nodeFac, alph=alph, nInputs=nInputs, epochs=epochs, batch_size=batch_size, verbose=2)
    #  ~print(my_model.summary)
    print(inputVarsV2)
    if genMETweighted:
        myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y,val_weights),sample_weight=train_weights,callbacks=[tensorboard_callback, earlystopping])
    else:
        myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[tensorboard_callback, earlystopping])
    
    # derive permutation importance, if selected
    if permuationImportance:
        perm = PermutationImportance(my_model, random_state=1).fit(train_x,train_y)
        output = eli5.format_as_text(eli5.explain_weights(perm, target_names = target,feature_names = inputVars[:-1]))
        print(name)
        print(output)
        output_file = open("PermutationImportance/2018/2D/"+modelName+".txt", "w")
        output_file.write(output)
        output_file.close()

    logcosh_val = myHistory.history["val_logcosh"][np.argmin(myHistory.history["val_loss"])]
    print("\nloss (logcosh w/o regularisation term): {0:.5g}\n".format(logcosh_val))
    
    #save Model
    my_model.model.save('trainedModel_Keras/'+year+'/2D/'+modelName)
    my_model.model.save('trainedModel_Keras/'+year+'/2D/'+modelName+".h5")
    print(inputVarsV2)
    
    if plotOutputs: 
        modelTitle = r"learnrate={:.2e};dropout={:.2};$\lambda$={:.2e};batchsize={:.2e};".format(lr,dout,lamb,batch)+"\n"+r"n_layer={};nodefactor={};$\alpha$={:.2};n_inputs={};".format(nLayer,nodeFac,alph,nInputs)
        if genMETweighted: modelTitle+=" genMETweighted"
        plot_Output(year,pathNameDict,inputVars,'trainedModel_Keras/'+year+'/2D/'+modelName,targetName,target,correctedValues,modelTitle,modelNr,updateInput=False,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn,noTrainSplitting=False)
        if plotPurity: 
            print(year,pathNameDict,inputVarsV2,'trainedModel_Keras/'+year+'/2D/'+modelName,targetName,target,modelTitle,modelNr)
            plot_Purity(year,pathNameDict,inputVarsV2,'trainedModel_Keras/'+year+'/2D/'+modelName,targetName,target,correctedValues,modelTitle,modelNr,updateInput=True,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)

# function to derive shapley values for trained model
def shapleyValues(year,pathNameDict,inputVars,modelPath,targetName,target,correctedValues,updateInput=False,underSample=False):
    
    # get inputs
    train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(year,pathNameDict,inputVars,targetName,target,update=updateInput,underSample=underSample)
    
    # use only n events for deriving shapley values (full statistics takes way too long!)
    x_test=train_x.sample(n=10000)
    #  ~x_test=train_x
    x_test=x_test.to_numpy()
    
    #  ~corr = x_test["PuppiMET*sin(PuppiMET_phi)"].corr(x_test["METsig"])
    #  ~corr2 = x_test["PuppiMET*sin(PuppiMET_phi)"].corr(x_test["MET*sin(PFMET_phi)"])
    #  ~print("Correlation between PuppiMET in y and METsig: {}\nCorrelation between Puppimet and PFMET in y: {}".format(corr, corr2))
    
    #  ~f = plt.figure(figsize=(12, 12))
    #  ~plt.matshow(x_test.corr(), fignum=f.number, cmap=plt.get_cmap("seismic"), vmin=-1., vmax=1.)
    #  ~plt.xticks(range(x_test.select_dtypes(['number']).shape[1]), x_test.select_dtypes(['number']).columns, fontsize=6, rotation=90)
    #  ~plt.yticks(range(x_test.select_dtypes(['number']).shape[1]), x_test.select_dtypes(['number']).columns, fontsize=6)
    #  ~cb = plt.colorbar()
    #  ~cb.ax.tick_params(labelsize=14)
    #  ~plt.show()
    
    #  ~f2 = plt.figure()
    #  ~x_test["METsig"].plot.hist(bins=100, range=(0,350))
    #  ~plt.show()
    
    # load trained model
    model = load_model(modelPath+".h5")
    
    # derive shapley values
    ex = shap.GradientExplainer(model, x_test)
    
    # plot shapley values
    shap_values = ex.shap_values(x_test)
    #  ~print(shap_values.abs.sum(0))
    max_display = x_test.shape[1]
    #  ~print(np.abs(shap_values).mean(0), "\n.............\n")
    for i in shap_values:
        #  ~print(np.mean(np.abs(i), axis=0))
        print(np.abs(i).mean(0))
        #  ~for j in np.mean(np.abs(i), axis=0):
            #  ~print(j)
    figure = plt.gcf()  # get current figure
    shap.summary_plot(shap_values, x_test, plot_type = "bar", feature_names = inputVars[:-1], max_display = max_display, show=False)
    #  ~shap.plots.bar(shap_values.abs.sum(0), x_test, plot_type = "bar", feature_names = inputVars[:-1], max_display = max_display, show=False)
    figure.set_size_inches(32, 18)
    #  ~plt.show()
    names = ([tex.get_text() for tex in plt.yticks()[1]])
    print(names[::-1])
    plt.savefig("ShapleyValues/shap_{0}_{1}.pdf".format(year,modelPath.split("/")[-1]))
    
    

    
#############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018 (default)",required=True)
    parser.add_argument('--version', type=str, help="treeVersion, such as v07 (default)",required=True)
    parser.add_argument('--mode', type=int, help="Runninge mode, 1 (default) for training network, 2 for deriving and plotting shapley values (needs already trained model as input)",required=True)
    args = parser.parse_args()
    
    if args.year: year = args.year
    if args.version: version = args.version
    if args.mode: mode = args.mode

    #  ~sampleNameRoot = "TTbar_amcatnlo"
    #  ~dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/Nominal/".format(year=year, version=version)
    #  ~sampleNameRoot = "TTbar_diLepton_MTOP175p5"
    #  ~dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/MTOP175p5/".format(year=year, version=version)
    #  ~sampleNameRoot = "TTbar_diLepton_MTOP169p5"
    #  ~dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/MTOP169p5/".format(year=year, version=version)
    #  ~sampleNameRoot = "TTbar_amcatnlo"
    #  ~dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/removeMetCut/".format(year=year, version=version)
    #  ~dataPath += sampleNameRoot+"_merged.root"
    
    # Dict which stores the path and the datasetname to be used for training (can have multiple entries)
    pathNameDict = {
        "TTbar_amcatnlo" : "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/removeMetCut/TTbar_amcatnlo_merged.root".format(year=year, version=version)
    }
        
    # Only run on mumu or emuevents
    #  ~sampleNameRoot += "_mumu"
    #  ~sampleNameRoot += "_emu"
        
    # Define Input Variables
    inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "mjj", "Jet1_E", "HT", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)"] # Finalized set of input variables
    
    #  ~inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "mjj", "Jet1_E", "HT", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)","DeepMET_reso*sin(DeepMET_reso_phi)","DeepMET_reso*cos(DeepMET_reso_phi)","DeepMET_resp*sin(DeepMET_resp_phi)","DeepMET_resp*cos(DeepMET_resp_phi)"] # Finalized set of input variables with DeepMET
    
    #  ~inputVars = ["METunc_Puppi", "PuppiMET*cos(PuppiMET_phi)", "PuppiMET*sin(PuppiMET_phi)", "MET*cos(PFMET_phi)", "MET*sin(PFMET_phi)", "CaloMET", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "Jet1_pt*sin(Jet1_phi)", "Jet1_pt*cos(Jet1_phi)", "MHT", "mass_l1l2_allJet", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "mjj", "n_Interactions", "MT2", "Lep2_pt*sin(Lep2_phi)", "dPhiMETleadJet_Puppi", "Lep2_pt*cos(Lep2_phi)", "HT", "dPhiMETleadJet", "dPhiLep1Jet1", "MT", "Lep1_pt*cos(Lep1_phi)", "vecsum_pT_allJet", "dPhiMETnearJet_Puppi", "vecsum_pT_l1l2_allJet", "nJets", "dPhiMETnearJet", "dPhiJet1Jet2", "Jet2_E", "Lep1_pt*sin(Lep1_phi)", "Jet1_E", "dPhiMETlead2Jet_Puppi", "dPhiLep1Lep2", "Lep1_E", "dPhiMETfarJet", "Jet2_eta", "dPhiMETbJet", "dPhiMETfarJet_Puppi", "mLL", "dPhiMETbJet_Puppi", "Lep2_flavor", "Lep2_E", "Jet1_eta", "Lep1_eta", "dPhiMETlead2Jet", "Lep1_flavor", "dPhiLep1bJet", "Lep2_eta", "METsig", "ratio_vecsumpTlep_vecsumpTjet", "looseLeptonVeto"]
    
    # Define targets
    targets = ["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"]
    correctedValues = ["PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)"]
    #  ~targets = ["DeepMET_reso*cos(DeepMET_reso_phi)-genMET*cos(genMET_phi)","DeepMET_reso*sin(DeepMET_reso_phi)-genMET*sin(genMET_phi)"]
    #  ~correctedValues = ["DeepMET_reso*cos(DeepMET_reso_phi)","DeepMET_reso*sin(DeepMET_reso_phi)"]
    
    # Standard network structure used before bayesian optimization
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard
    
    # Network structures from bayesian optimization
    netStruct = {"alph": 0.11556803605322355, "batch": 6.501144102888323, "dout": 0.35075846582000303, "lamb": -5.941028038499022, "lr": -7.729770703881016, "nLayer": 2.2186773553565198, "nodeFac": 4.424425111826699}
    
    alph, batch, dout, lamb, lr, nLayer, nodeFac = netStruct["alph"], netStruct["batch"], netStruct["dout"], netStruct["lamb"], netStruct["lr"], netStruct["nLayer"], netStruct["nodeFac"]
    

    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    nodeFac = nodeFacs[int(np.round(nodeFac))]
    print("lr: ", np.exp(lr),"dout: ", dout, "lamb: ", np.exp(lamb), "batch_size: ",  int(np.round(np.exp(batch))), "nlayer: ", int(np.round(nLayer)), "nodes: ", nodeFac, "alpha: ", alph)
    
    nInputs=int(len(inputVars))
    modelNr=152
    print("Number of inputs: {}".format(nInputs))


    if mode==2:
        # plotting shapley values for trained model saved in path "trainedModelPath"
        trainedModelPath = "trainedModel_Keras/2018/2D/Inlusive_noMetCut_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20230111-1004genMETweighted"
        shapleyValues(year,pathNameDict,inputVars[:nInputs],trainedModelPath,"diff_xy",targets,correctedValues,updateInput=True)
    else: 
        # training DNN with network parameters defined above
        #  ~trainKeras(year,pathNameDict,inputVars[:nInputs],"Inlusive_noMetCut_amcatnlo_xyComponent_JetLepXY_50EP","diff_xy",targets,correctedValues,np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))),int(np.round(nLayer)),nodeFac,alph,nInputs,updateInput=True,genMETweighted=True,plotOutputs=True,modelNr=modelNr,plotPurity=False)
        trainKeras(year,pathNameDict,inputVars[:nInputs],"Inlusive_noMetCut_genMETweight600enlarge_amcatnlo_xyComponent_JetLepXY_50EP","diff_xy",targets,correctedValues,np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))),int(np.round(nLayer)),nodeFac,alph,nInputs,updateInput=True,genMETweighted=True,plotOutputs=True,modelNr=modelNr,plotPurity=False)
        #  ~trainKeras(year,pathNameDict,inputVars[:nInputs],"Inlusive_noMetCut_genMETweight600_amcatnlo_xyComponent_JetLepXY_50EP","diff_xy",targets,correctedValues,np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))),int(np.round(nLayer)),nodeFac,alph,nInputs,updateInput=True,genMETweighted=True,plotOutputs=True,modelNr=modelNr,plotPurity=False)
