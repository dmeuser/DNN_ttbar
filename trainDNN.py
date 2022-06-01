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
from plotPerformanceDNN import plot_Output
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
def trainKeras(year,dataPath,inputVars,name,treeName,targetName,target,lr,dout,lamb,batch,nLayer,nodeFac,alph,nInputs,updateInput=False,permuationImportance=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False,plotOutputs=False,modelNr=0):
        
    # set number of epochs and batchsize
    #  ~epochs = 100
    epochs = 400
    #  ~batch_size = 5000
    batch_size = batch
    
    stringStart=datetime.datetime.now().strftime("%Y%m%d-%H%M")

    modelName=name+"_"+"_"+targetName+"_"+year+"_"+stringStart      # time added to name of trained model 
    
    if genMETweighted:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights = getInputArray_allBins_nomDistr(year,dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
        modelName+="genMETweighted"
    else:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(year,dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    print("..........................\n\n", train_x.shape)
    
    #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)   # could be used for early stopping of training
    logdir="./logs/"+year+"/2D/"+modelName
    tensorboard_callback = TensorBoard(log_dir=logdir)      # setup tensorboard to log training progress
    earlystopping = EarlyStoppingCombined(patience=10, percentage=2, percentagePatience=10, generalizationPatience=10) # Using own earlystopping function
    
    # setup keras and train model
    my_model = KerasRegressor(build_fn=bJetRegression_Model, lr=lr, dout=dout, lamb=lamb, nLayer=nLayer, nodeFac=nodeFac, alph=alph, nInputs=nInputs, epochs=epochs, batch_size=batch_size, verbose=2)
    #  ~print(my_model.summary)
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

    if plotOutputs: 
        modelTitle = r"learnrate={:.2e};dropout={:.2};$\lambda$={:.2e};batchsize={:.2e};".format(lr,dout,lamb,batch)+"\n"+r"n_layer={};nodefactor={};$\alpha$={:.2};n_inputs={};".format(nLayer,nodeFac,alph,nInputs)
        if genMETweighted: modelTitle+=" genMETweighted"
        plot_Output(year,dataPath,inputVars,'trainedModel_Keras/'+year+'/2D/'+modelName,treeName,targetName,target,modelTitle,modelNr,updateInput=False,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)

# function to derive shapley values for trained model
def shapleyValues(year,dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False,underSample=False):
    
    # get inputs
    #  ~x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput)
    train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(year,dataPath,inputVars,targetName,target,treeName,update=updateInput,underSample=underSample)
    
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
    max_display = x_test.shape[1]
    #  ~print(shap_values.shape)
    figure = plt.gcf()  # get current figure
    shap.summary_plot(shap_values, x_test, plot_type = "bar", feature_names = inputVars[:-1], max_display = max_display, show=False)
    figure.set_size_inches(32, 18)
    #  ~plt.show()
    names = ([tex.get_text() for tex in plt.yticks()[1]])
    print(names[::-1])
    plt.savefig("shap_{0}_{1}.pdf".format(year,modelPath.split("/")[-1]))

    
#############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018 (default)")
    parser.add_argument('--version', type=str, help="treeVersion, such as v07 (default)")
    parser.add_argument('--mode', type=int, help="Runninge mode, 1 (default) for training network, 2 for deriving and plotting shapley values (needs already trained model as input)")
    args = parser.parse_args()

    year = "2018"
    version = "v07"
    mode = 1
    
    if args.year: year = args.year
    if args.version: version = args.version
    if args.mode: mode = args.mode

    sampleNameRoot = "TTbar_amcatnlo"
    dataPath = "/net/data_cms1b/user/dmeuser/top_analysis/{year}/{version}/minTrees/100.0/Nominal/".format(year=year, version=version)
    dataPath += sampleNameRoot+"_merged.root"
        
    # Define Input Variables
    #  ~inputVars = ["PuppiMET_xy*cos(PuppiMET_xy_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]   # All input variables
    inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "mjj", "Jet1_E", "HT", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)"] # Finalized set of input variables
    
    # Standard network structure used before bayesian optimization
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard
    
    # Network structures from bayesian optimization
    #  ~netStruct = {"alph": 0.17599650023449842, "batch": 8.894636184503135, "dout": 0.1721645529848336, "lamb": -7.928067002424198, "lr": -8.61803474144627, "nLayer": 3.262036072600552, "nodeFac": 0.3549618413748268}
    #  ~netStruct = {"alph": 0.11561571686757542, "batch": 6.848504823076402, "dout": 0.1396968795456421, "lamb": -7.595457622529098, "lr": -8.975982634440493, "nLayer": 4.887771333464286, "nodeFac": 0.33526528228286}
    #  ~netStruct = {"alph": 0.2349345205718364, "batch": 8.02317400604099, "dout": 0.13318509762384187, "lamb": -2.1081084090465985, "lr": -7.824162070713905, "nLayer": 2.0486796312986497, "nodeFac": 0.9343665382925298}
    #  ~netStruct = {"alph": 0.14543941501295446, "batch": 6.902627778737529, "dout": 0.31655922316415264, "lamb": -8.87955161852822, "lr": -9.568007271513965, "nLayer": 1.0, "nodeFac": 2.772054780435249}
    #  ~netStruct = {'alph': 0.19910693195333412, 'batch': 6.5256786079929405, 'dout': 0.2851066788030514, 'lamb': -8.981251866419107, 'lr': -11.499238007076016, 'nLayer': 1.078089616804189, 'nodeFac': 2.1780041830595853}
    #  ~netStruct = {'alph': 0.3188498809145474, 'batch': 8.140987890590951, 'dout': 0.48683691567284126, 'lamb': -9.153676773898741, 'lr': -10.287249537440374, 'nLayer': 7.997912709072231, 'nodeFac': 4.988908590168827}
    #  ~netStruct = {"alph": 0.2089130076875134, "batch": 5.815243202632221, "dout": 0.12694298707859258, "lr": -10.690751787974715, "nLayer": 2.486790327777428, "nodeFac": 1.1879832778054233}
    netStruct = {"alph": 0.18306907123612454, "batch": 6.722536144841143, "dout": 0.3424114940723806, "lr": -9.232751103823428, "nLayer": 1.9863309928370811, "nodeFac": 1.4378897355741938}
    
    #  ~alph, batch, dout, lamb, lr, nLayer, nodeFac = netStruct["alph"], netStruct["batch"], netStruct["dout"], netStruct["lamb"], netStruct["lr"], netStruct["nLayer"], netStruct["nodeFac"]
    alph, batch, dout, lamb, lr, nLayer, nodeFac = netStruct["alph"], netStruct["batch"], netStruct["dout"], np.log(0.05), netStruct["lr"], netStruct["nLayer"], netStruct["nodeFac"]
    
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    nodeFac = nodeFacs[int(np.round(nodeFac))]
    print("lr: ", np.exp(lr),"dout: ", dout, "lamb: ", np.exp(lamb), "batch_size: ",  int(np.round(np.exp(batch))), "nlayer: ", int(np.round(nLayer)), "nodes: ", nodeFac, "alpha: ", alph)
    
    nInputs=int(len(inputVars))
    print("Number of inputs: {}".format(nInputs))
    
    
    
    if mode==2:
        # plotting shapley values for trained model saved in path "trainedModelPath"
        trainedModelPath = "trainedModel_Keras/2016_preVFP/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2016_preVFP_20220504-1236genMETweighted"
        shapleyValues(year,dataPath,inputVars[:nInputs],trainedModelPath,"TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True)
    else: 
        # training DNN with network parameters defined above
        trainKeras(year,dataPath,inputVars[:nInputs],"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))),int(np.round(nLayer)),nodeFac,alph,nInputs,updateInput=True,genMETweighted=True,plotOutputs=True,modelNr=105)
