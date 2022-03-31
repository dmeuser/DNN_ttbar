from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
from subprocess import call
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
        model.add(Dense(nodeNr, kernel_regularizer=regularizers.l2(regLamb)))
        #  ~model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Dropout(dout))
        model.add(LeakyReLU(alpha=alph))
    
    
    model.add(Dense(2, kernel_initializer='normal', activation='linear'))
    #  ~model.compile(loss="logcosh", optimizer=Adam(lr=lr),metrics=['mean_squared_error','mean_absolute_percentage_error',"logcosh"])
    model.compile(loss="logcosh", optimizer=Adam(lr=lr),metrics=['mean_squared_error',"logcosh"])
    return model

# function to train model
def trainKeras(dataPath,inputVars,name,treeName,targetName,target,lr,dout,lamb,batch,nLayer,nodeFac,alph,nInputs,updateInput=False,permuationImportance=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
        
    # set number of epochs and batchsize
    #  ~epochs = 100
    epochs = 200
    #  ~batch_size = 5000
    batch_size = batch
    
    stringStart=datetime.datetime.now().strftime("%Y%m%d-%H%M")

    modelName=name+"_"+"_"+targetName+"_2018_"+stringStart      # time added to name of trained model 
    
    if genMETweighted:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights = getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
        modelName+="genMETweighted"
    else:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,underSample=underSample,doSmogn=doSmogn)
    
    print("..........................\n\n", train_x.shape)
    
    #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)   # could be used for early stopping of training
    logdir="./logs/2018/2D/"+modelName
    tensorboard_callback = TensorBoard(log_dir=logdir)      # setup tensorboard to log training progress
    earlystopping = EarlyStoppingCombined(patience=10, percentage=2, percentagePatience=10, generalizationPatience=10)
    
    # setup keras and train model
    my_model = KerasRegressor(build_fn=bJetRegression_Model, lr=lr, dout=dout, lamb=lamb, nLayer=nLayer, nodeFac=nodeFac, alph=alph, nInputs=nInputs, epochs=epochs, batch_size=batch_size, verbose=2)
    #  ~print(my_model.summary)
    if genMETweighted:
        myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y,val_weights),sample_weight=train_weights,callbacks=[tensorboard_callback, earlystopping])
        #  ~myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y,val_weights),sample_weight=train_weights,callbacks=[tensorboard_callback])
    else:
        #  ~myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[tensorboard_callback])
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
    my_model.model.save('trainedModel_Keras/2018/2D/'+modelName)
    my_model.model.save('trainedModel_Keras/2018/2D/'+modelName+".h5")

    #  ~plot_Output(dataPath,inputVars,'trainedModel_Keras/2018/2D/'+modelName,treeName,targetName,target,updateInput=False,normalize=normalize,standardize=standardize,genMETweighted=genMETweighted,overSample=overSample,doSmogn=doSmogn)

# function to derive shapley values for trained model
def shapleyValues(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False,underSample=False):
    
    # get inputs
    #  ~x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput)
    train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput,underSample=underSample)
    
    # use only n events for deriving shapley values (full statistics takes way too long!)
    x_test=train_x.sample(n=100000)
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
    figure = plt.gcf()  # get current figure
    shap.summary_plot(shap_values, x_test, plot_type = "bar", feature_names = inputVars[:-1], max_display = max_display, show=False)
    figure.set_size_inches(32, 18)
    names = ([tex.get_text() for tex in plt.yticks()[1]])
    print(names[::-1])
    plt.savefig("shap_{0}.pdf".format(modelPath.split("/")[-1]))

# function to derive different control performance plots based on trained model
def plot_Output(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
    
    if not os.path.exists("outputComparison/2018/2D/"+modelPath.split("/")[-1]):    # create output folder for plots if not available
        os.makedirs("outputComparison/2018/2D/"+modelPath.split("/")[-1])
    
    # get inputs
    if genMETweighted:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals, train_weights, val_weights, test_weights = getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInput,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,doSmogn=doSmogn)
        
    else:
        train_x, val_x, test_x, train_y, val_y, test_y, train_metVals, val_metVals, test_metVals = getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,treeName,update=updateInputs,normalize=normalize,standardize=False,genMETweighted=genMETweighted,overSample=overSample,doSmogn=doSmogn)
    
    model = load_model(modelPath+".h5")
    
    
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
    
    train_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/trainResults.pkl")
    
    val_x["DNN_1"]=[row[0] for row in y_hat_val]
    val_x["DNN_2"]=[row[1] for row in y_hat_val]
    val_x[target[0]]=val_y[target[0]]
    val_x[target[1]]=val_y[target[1]]
    
    val_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/valResults.pkl")
    
    test_x["DNN_1"]=[row[0] for row in y_hat_test]
    test_x["DNN_2"]=[row[1] for row in y_hat_test]
    test_x[target[0]]=test_y[target[0]]
    test_x[target[1]]=test_y[target[1]]
    
    test_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/testResults.pkl")
    
    
    
#############################################################
if __name__ == "__main__":
    # Define input data path
    #  ~dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v06/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v07/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"

    # Define Input Variables
    #  ~inputVars1 = ["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj",]

    #  ~inputVars = ["METunc_Puppi","PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","CaloMET*sin(CaloMET_phi)","CaloMET*cos(CaloMET_phi)","vecsum_pT_allJet*sin(HT_phi)","vecsum_pT_allJet*cos(HT_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_pt*cos(Jet1_phi)","MT2","Lep2_pt*sin(Lep2_phi)","Lep2_pt*cos(Lep2_phi)","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_pt*cos(Lep1_phi)","n_Interactions","dPhiJet1Jet2","dPhiMETnearJet_Puppi","dPhiLep1Jet1","vecsum_pT_allJet","dPhiMETleadJet","nJets","MT","dPhiMETnearJet","Jet2_E","Jet1_eta","dPhiLep1Lep2","mLL","dPhiMETbJet_Puppi","dPhiMETfarJet","Lep2_flavor","vecsum_pT_l1l2_allJet","looseLeptonVeto","Lep2_eta","Lep1_eta","dPhiMETlead2Jet","dPhiMETlead2Jet_Puppi","dPhiMETbJet","dPhiLep1bJet","Lep1_E","Lep1_flavor","ratio_vecsumpTlep_vecsumpTjet","mjj","Lep2_E","dPhiMETfarJet_Puppi","mass_l1l2_allJet","METsig","dPhiMETleadJet_Puppi","Jet2_eta","MHT","Jet1_E","HT"]
    #  ~inputVars = ["METunc_Puppi","PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","CaloMET*sin(CaloMET_phi)","CaloMET*cos(CaloMET_phi)","vecsum_pT_allJet*sin(HT_phi)","vecsum_pT_allJet*cos(HT_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_pt*cos(Jet1_phi)","MT2","Lep2_pt*sin(Lep2_phi)","Lep2_pt*cos(Lep2_phi)","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_pt*cos(Lep1_phi)","n_Interactions","dPhiJet1Jet2","dPhiMETnearJet_Puppi","dPhiLep1Jet1","vecsum_pT_allJet","dPhiMETleadJet","nJets","MT","dPhiMETnearJet","Jet2_E","Jet1_eta","dPhiLep1Lep2","mLL","dPhiMETbJet_Puppi","dPhiMETfarJet","Lep2_flavor","vecsum_pT_l1l2_allJet","looseLeptonVeto","Lep2_eta","Lep1_eta","dPhiMETlead2Jet","dPhiMETlead2Jet_Puppi","dPhiMETbJet","dPhiLep1bJet","Lep1_E","Lep1_flavor","ratio_vecsumpTlep_vecsumpTjet","mjj","Lep2_E","dPhiMETfarJet_Puppi","mass_l1l2_allJet","METsig","dPhiMETleadJet_Puppi","Jet2_eta","MHT","Jet1_E","HT"]
    #  ~inputVars = ["METunc_Puppi", "PuppiMET*cos(PuppiMET_phi)", "PuppiMET*sin(PuppiMET_phi)", "MET*cos(PFMET_phi)", "MET*sin(PFMET_phi)", "CaloMET*sin(CaloMET_phi)", "CaloMET*cos(CaloMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "Jet1_pt*sin(Jet1_phi)", "Jet1_pt*cos(Jet1_phi)", "MHT", "mass_l1l2_allJet", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "mjj", "n_Interactions", "MT2", "Lep2_pt*sin(Lep2_phi)", "dPhiMETleadJet_Puppi", "Lep2_pt*cos(Lep2_phi)", "HT", "dPhiMETleadJet", "dPhiLep1Jet1", "MT", "Lep1_pt*cos(Lep1_phi)", "vecsum_pT_allJet", "dPhiMETnearJet_Puppi", "vecsum_pT_l1l2_allJet", "nJets", "dPhiMETnearJet", "dPhiJet1Jet2", "Jet2_E", "Lep1_pt*sin(Lep1_phi)", "Jet1_E", "dPhiMETlead2Jet_Puppi", "dPhiLep1Lep2", "Lep1_E", "dPhiMETfarJet", "Jet2_eta", "dPhiMETbJet", "dPhiMETfarJet_Puppi", "mLL", "dPhiMETbJet_Puppi", "Lep2_flavor", "Lep2_E", "Jet1_eta", "Lep1_eta", "dPhiMETlead2Jet", "Lep1_flavor", "dPhiLep1bJet", "Lep2_eta", "METsig", "ratio_vecsumpTlep_vecsumpTjet", "looseLeptonVeto"]
    #  ~inputVars = ["METunc_Puppi", "PuppiMET*cos(PuppiMET_phi)", "PuppiMET*sin(PuppiMET_phi)", "MET*cos(PFMET_phi)", "MET*sin(PFMET_phi)", "CaloMET", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "Jet1_pt*sin(Jet1_phi)", "Jet1_pt*cos(Jet1_phi)", "MHT", "mass_l1l2_allJet", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "mjj", "n_Interactions", "MT2", "Lep2_pt*sin(Lep2_phi)", "dPhiMETleadJet_Puppi", "Lep2_pt*cos(Lep2_phi)", "HT", "dPhiMETleadJet", "dPhiLep1Jet1", "MT", "Lep1_pt*cos(Lep1_phi)", "vecsum_pT_allJet", "dPhiMETnearJet_Puppi", "vecsum_pT_l1l2_allJet", "nJets", "dPhiMETnearJet", "dPhiJet1Jet2", "Jet2_E", "Lep1_pt*sin(Lep1_phi)", "Jet1_E", "dPhiMETlead2Jet_Puppi", "dPhiLep1Lep2", "Lep1_E", "dPhiMETfarJet", "Jet2_eta", "dPhiMETbJet", "dPhiMETfarJet_Puppi", "mLL", "dPhiMETbJet_Puppi", "Lep2_flavor", "Lep2_E", "Jet1_eta", "Lep1_eta", "dPhiMETlead2Jet", "Lep1_flavor", "dPhiLep1bJet", "Lep2_eta", "METsig", "ratio_vecsumpTlep_vecsumpTjet", "looseLeptonVeto"]
    #  ~inputVars = ["PuppiMET*sin(PuppiMET_phi)", "PuppiMET*cos(PuppiMET_phi)", "MET*sin(PFMET_phi)", "MET*cos(PFMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "MT2", "mjj", "nJets", "Jet1_E", "HT", "METunc_Puppi", "n_Interactions", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "Lep2_pt*cos(Lep2_phi)", "dPhiMETnearJet_Puppi", "mLL", "dPhiMETleadJet_Puppi", "vecsum_pT_l1l2_allJet", "Jet2_E", "dPhiMETbJet_Puppi", "dPhiMETleadJet", "dPhiMETbJet", "Lep2_pt*sin(Lep2_phi)", "MT", "Lep1_E", "dPhiLep1Jet1", "Lep2_E", "dPhiMETlead2Jet", "dPhiLep1Lep2", "Jet1_eta", "dPhiMETfarJet", "Lep2_eta", "dPhiLep1bJet", "dPhiMETfarJet_Puppi", "dPhiMETnearJet", "dPhiJet1Jet2", "Lep2_flavor", "dPhiMETlead2Jet_Puppi", "Lep1_flavor", "METsig", "Lep1_eta", "Jet2_eta", "looseLeptonVeto", "ratio_vecsumpTlep_vecsumpTjet"]
    
    #  ~inputVars = ["PuppiMET*sin(PuppiMET_phi)", "PuppiMET*cos(PuppiMET_phi)", "MET*sin(PFMET_phi)", "MET*cos(PFMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "MT2", "mjj", "nJets", "Jet1_E", "HT", "METunc_Puppi"]
    #  ~inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "MT2", "mjj", "nJets", "Jet1_E", "HT", "METunc_Puppi"]
    #  ~inputVars = ["PuppiMET*sin(PuppiMET_phi)", "PuppiMET*cos(PuppiMET_phi)", "MET*sin(PFMET_phi)", "MET*cos(PFMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "mjj", "Jet1_E"]
    inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "CaloMET", "vecsum_pT_allJet", "mjj", "Jet1_E"]
	#  ~out: MT2, HT(?), nJets, METunc_Puppi
    
    #  ~inputVars = ["PuppiMET*cos(PuppiMET_phi)", "PuppiMET*sin(PuppiMET_phi)", "MET*cos(PFMET_phi)", "MET*sin(PFMET_phi)", "CaloMET*sin(CaloMET_phi)", "CaloMET*cos(CaloMET_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "Jet1_pt*sin(Jet1_phi)", "Jet1_pt*cos(Jet1_phi)", "MHT", "mass_l1l2_allJet", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)", "mjj", "n_Interactions", "METunc_Puppi", "MT2", "Lep2_pt*sin(Lep2_phi)", "dPhiMETleadJet_Puppi", "Lep2_pt*cos(Lep2_phi)"]
    #  ~print(len(inputVars1), len(inputVars))
    #  ~print(np.where(np.unique(np.array(inputVars))==np.unique(np.array(inputVars1))))
    #  ~trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True,genMETweighted=True)
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.097075752974773, 0.2899393936036301, -8.01970544548232, 8.154158507965985, 2.402508434459325, 3.5646202688746493, 0.11396439396039146
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.524903792932268, 0.5, -7.271889484384392, 7.1970043195604125, 8.0, 4.648503137689418, 0.0
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, np.log(50000), 6.0, 3.0, 0.2 
    lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.40951173838937, 0.4307905292558073, -6.478574338240861, 5.525847797310747, 7.743792064550437, 1.6496830299486087, 0.21534075102851583 
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.724648816372166, 0.42019722418191996, -1.9871840218385974, 7.288464822183116, 3.7077713293386814, 3.098255409797496, 0.10318236454640405
    
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    nodeFac2 = nodeFacs[int(np.round(nodeFac))]
    print("lr: ", np.exp(lr),"dout: ", dout, "lamb: ", np.exp(lamb), "batch_size: ",  int(np.round(np.exp(batch))), "nlayer: ", int(np.round(nLayer)), "nodes: ", nodeFac2, "alpha: ", alph)
    
    #  ~nInputs=11
    nInputs=int(len(inputVars))
    #  ~nInputs=21
    print("Number of inputs: {}".format(nInputs))
    
    trainKeras(dataPath,inputVars[:nInputs],"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))),int(np.round(nLayer)),nodeFac2,alph,nInputs,updateInput=True,genMETweighted=True,standardize=False,overSample=False,underSample=False,doSmogn=False)
    #  ~trainKeras(dataPath,inputVars[:nInputs],"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))),int(np.round(nLayer)),nodeFac2,alph,nInputs,updateInput=True,genMETweighted=False,standardize=False,overSample=False,underSample=False,doSmogn=False)
    
    #  ~shapleyValues(dataPath,inputVars[:nInputs],"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20220317-1151genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True)
    
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_/home/home4/institut_1b/nattland/DNN_ttbar/outputComparison/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211214-1022","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=False,standardize=False, overSample=False)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211122-1150genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True,genMETweighted=True,standardize=False,overSample=False,doSmogn=False)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211122-1454genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"], np.exp(lr), dout, np.exp(lamb), int(np.round(np.exp(batch))), int(np.round(nLayer)), nodeFac2, alph,updateInput=True,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211125-1228genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)

