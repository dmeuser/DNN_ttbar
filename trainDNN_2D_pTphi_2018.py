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
    
def bJetRegression_Model():
    model = Sequential()
    #  ~model.add(Dense(24, input_dim=24))
    #  ~model.add(Dense(18, input_dim=18))
    #  ~model.add(Dense(47, input_dim=47))
    model.add(BatchNormalization())
    model.add(Dense(54, input_dim=54))
    model.add(BatchNormalization())
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LeakyReLU(alpha=0.2))
    
    #  ~model.add(Dense(1, kernel_initializer='normal', activation='linear',kernel_constraint=NonNeg()))
    #  ~model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.add(Dense(2, kernel_initializer='normal', activation='linear'))
    #  ~model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    #  ~model.compile(loss="logcosh", optimizer=Adam(lr=0.001),metrics=['mean_squared_error','mean_absolute_percentage_error'])
    model.compile(loss="logcosh", optimizer=Adam(lr=0.0001),metrics=['mean_squared_error','mean_absolute_percentage_error'])
    #  ~model.compile(loss="mean_absolute_percentage_error", optimizer=Adam(lr=0.0001),metrics=['mean_squared_error','logcosh'])
    #  ~model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
    #  ~model.compile(loss="huber_loss", optimizer=Adam(lr=0.001))
    return model

def dPhi(x,y):
    dPhi=x-y
    if (dPhi)>np.pi:
        return dPhi-2*np.pi
    elif (dPhi)>(-1*np.pi):
        return dPhi+2*np.pi
    else:
        return dPhi
  
def getInputArray_allBins_nomDistr(path,inputVars,targetName,target,update,treeName,normalize=False,standardize=False):
    for var in target:
        if var=="dPhi(PuppiMET_phi,genMET_phi)":
            inputVars.append("genMET_phi")
        else:
            inputVars.append(var)
    
    outputPath="input/2D/"+treeName+"_"+targetName+"_nomDistr.pkl"
        
    if treeName.split("_")[-1]=="emu":
        only_emu=True
        treeName=treeName.replace("_emu","")
    else:
        only_emu=False
        
    if normalize:
        outputPath=outputPath.replace("_nomDistr","normalized_nomDistr")
    if standardize:
        outputPath=outputPath.replace("_nomDistr","standardized_nomDistr")
        
    if update:
        root_file = uproot.open(path)
        events = root_file["ttbar_res100.0;1"][treeName+";1"]
        cut = "(PuppiMET>0)"
        if only_emu:
            cut+="&(emu==1)"
        inputFeatures = events.arrays(inputVars,cut,library="pd")
        
        if target[1]=="dPhi(PuppiMET_phi,genMET_phi)":
            inputFeatures[target[1]]=inputFeatures["PuppiMET_phi"]-inputFeatures["genMET_phi"]
            conditions=[(inputFeatures[target[1]]>np.pi),(inputFeatures[target[1]]<(-1.*np.pi)),True]
            values=[inputFeatures[target[1]]-2*np.pi,inputFeatures[target[1]]+2*np.pi,inputFeatures[target[1]]]
            inputFeatures[target[1]]=np.select(conditions, values)
            inputFeatures=inputFeatures.drop(["genMET_phi"],axis=1)
        
        if normalize:
            scaler = MinMaxScaler()
            scaledFeatures = scaler.fit_transform(inputFeatures.values)
            inputFeatures = pd.DataFrame(scaledFeatures, index=inputFeatures.index, columns=inputFeatures.columns)
        elif standardize:
            scaler = StandardScaler()
            scaledFeatures = scaler.fit_transform(inputFeatures.values)
            inputFeatures = pd.DataFrame(scaledFeatures, index=inputFeatures.index, columns=inputFeatures.columns)
        
        inputFeatures.to_pickle(outputPath)
    else:
        inputFeatures=pd.read_pickle(outputPath)
    
    if target[1]=="dPhi(PuppiMET_phi,genMET_phi)":
        inputVars.remove("genMET_phi")
    
    print("Train with:",outputPath)
    
    if normalize or standardize:
        return inputFeatures[inputVars[:-1]],inputFeatures[target],inputVars,scaler
    else:
        return inputFeatures[inputVars[:-1]],inputFeatures[target],inputVars

def trainKeras(dataPath,inputVars,name,treeName,targetName,target,updateInput=False,permuationImportance=False,normalize=False,standardize=False):
        stringStart=datetime.datetime.now().strftime("%Y%m%d-%H%M")

        modelName=name+"_"+"_"+targetName+"_2018_"+stringStart
        
        x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize)
        modelName+="normDistr"
        
        train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
        
        #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        logdir="./logs/2D/"+modelName
        tensorboard_callback = TensorBoard(log_dir=logdir)
        
        #  ~my_model = KerasRegressor(build_fn=bJetRegression_Model, epochs=200, batch_size=5000, verbose=2)
        my_model = KerasRegressor(build_fn=bJetRegression_Model, epochs=30, batch_size=5000, verbose=2)
        my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[tensorboard_callback])
        #  ~my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[es])
        
        if permuationImportance:
            perm = PermutationImportance(my_model, random_state=1).fit(train_x,train_y)
            output = eli5.format_as_text(eli5.explain_weights(perm, target_names = target,feature_names = inputVars[:-1]))
            print(name)
            print(output)
            output_file = open("PermutationImportance/2D/"+modelName+".txt", "w")
            output_file.write(output)
            output_file.close()
        
        y_hat_train = my_model.predict(train_x)
        y_hat_test = my_model.predict(val_x)

        # display error values
        print ('Train RMSE: ', round(np.sqrt(((train_y - y_hat_train)**2).mean()), 4))    
        print ('Train MEAN: ', round(((train_y - y_hat_train).mean()), 4))    
        print ('Test RMSE: ', round(np.sqrt(((val_y - y_hat_test)**2).mean()), 4))
        print ('Test MEAN: ', round(((val_y - y_hat_test).mean()), 4))
        
        #save Model
        my_model.model.save('trainedModel_Keras/2D/'+modelName)
        my_model.model.save('trainedModel_Keras/2D/'+modelName+".h5")

def shapleyValues(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False):     #Shapley Values for Trained DNN
        
    x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName)
    
    x_test=x.sample(n=10000)
    x_test=x_test.to_numpy()
    
    model = load_model(modelPath+".h5")
    
    ex = shap.GradientExplainer(model, x_test)
    shap_values = ex.shap_values(x_test)
    max_display = x_test.shape[1]
    figure = plt.gcf()  # get current figure
    shap.summary_plot(shap_values, x_test, plot_type = "bar", feature_names = inputVars[:-1], max_display = max_display, show=False)
    figure.set_size_inches(32, 18)
    plt.savefig("ShapleyValues/2D/shap_{0}.pdf".format(modelPath.split("/")[-1]))
    
def plot_Output(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False,normalize=False,standardize=False):
    if not os.path.exists("outputComparison/2D/"+modelPath.split("/")[-1]):
        os.makedirs("outputComparison/2D/"+modelPath.split("/")[-1])
    
    if normalize or standardize:
        x,y,inputVars,scaler=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize)
    else:
        x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize)
    
    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
    
    model = load_model(modelPath+".h5")
    
    #  ~train_x=train_x.head(100000)
    #  ~val_x=val_x.head(100000)
    #  ~train_y=train_y.head(100000)
    #  ~val_y=val_y.head(100000)
    y_hat_train = model.predict(train_x,use_multiprocessing=True)
    y_hat_val = model.predict(val_x,use_multiprocessing=True)
    
    # Inverse transformation
    if normalize or standardize:
        train=pd.concat([train_x,train_y], axis=1)
        scaled=scaler.inverse_transform(train.values)
        train=pd.DataFrame(scaled, index=train.index, columns=train.columns)
        
    train_x["DNN_1"]=[row[0] for row in y_hat_train]
    train_x["DNN_2"]=[row[1] for row in y_hat_train]
    
    if normalize or standardize:
        scaled=scaler.inverse_transform(train_x.values)
        train_x=pd.DataFrame(scaled, index=train_x.index, columns=train_x.columns)
    
    train_x=pd.concat([train_x,train[target]], axis=1)
    print(train_x)
    
    if normalize or standardize:
        val=pd.concat([val_x,val_y], axis=1)
        scaled=scaler.inverse_transform(val.values)
        val=pd.DataFrame(scaled, index=val.index, columns=val.columns)
        
    val_x["DNN_1"]=[row[0] for row in y_hat_val]
    val_x["DNN_2"]=[row[1] for row in y_hat_val]
    
    if normalize or standardize:
        scaled=scaler.inverse_transform(val_x.values)
        val_x=pd.DataFrame(scaled, index=val_x.index, columns=val_x.columns)
    
    val_x=pd.concat([val_x,val[target]], axis=1)
    print(val_x)
    # End inverse transformation
    
    # Compare DNN and target - training sample
    plt.figure()
    train_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/train_x.pdf")
    plt.figure()
    train_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/train_y.pdf")
    
    # Compare DNN and target - validation sample
    plt.figure()
    val_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/val_x.pdf")
    plt.figure()
    val_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/val_y.pdf")
    
    # Compare DNN output between both samples
    min=train_x["DNN_1"].min()
    max=train_x["DNN_1"].max()
    plt.figure()
    train_x["DNN_1"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x["DNN_1"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/DNNcomparison_x.pdf")
    min=train_x["DNN_2"].min()
    max=train_x["DNN_2"].max()
    plt.figure()
    train_x["DNN_2"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x["DNN_2"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/DNNcomparison_y.pdf")
    
    # Compare target between both samples
    plt.figure()
    train_x[target[0]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x[target[0]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/TARGETcomparison_x.pdf")
    plt.figure()
    train_x[target[1]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x[target[1]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/TARGETcomparison_y.pdf")
    
    # Compare corrected MET to genMET pT - validation sample
    val_x["genMET_pT"]=val_x["PuppiMET"]-val_x[target[0]]
    val_x["DNN_MET_pT"]=val_x["PuppiMET"]-val_x["DNN_1"]
    
    min=val_x["genMET_pT"].min()
    max=val_x["genMET_pT"].max()
    plt.figure()
    val_x["genMET_pT"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_pT")
    val_x["DNN_MET_pT"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_pT")
    plt.legend()
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_val.pdf")
    
    # Compare corrected MET to genMET pT - training sample
    train_x["genMET_pT"]=train_x["PuppiMET"]-train_x[target[0]]
    train_x["DNN_MET_pT"]=train_x["PuppiMET"]-train_x["DNN_1"]
    
    min=train_x["genMET_pT"].min()
    max=train_x["genMET_pT"].max()
    plt.figure()
    train_x["genMET_pT"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_pT")
    train_x["DNN_MET_pT"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_pT")
    plt.legend()
    plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_train.pdf")
    
    #  ~# Compare corrected MET to genMET Y - validation sample
    #  ~val_x["genMET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x[target[1]]
    #  ~val_x["DNN_MET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x["DNN_2"]
    
    #  ~min=val_x["genMET_Y"].min()
    #  ~max=val_x["genMET_Y"].max()
    #  ~plt.figure()
    #  ~val_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y")
    #  ~val_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/METdistr_y_val.pdf")
    
    #  ~# Compare corrected MET to genMET Y - training sample
    #  ~train_x["genMET_Y"]=train_x["PuppiMET*sin(PuppiMET_phi)"]-train_x[target[1]]
    #  ~train_x["DNN_MET_Y"]=train_x["PuppiMET*sin(PuppiMET_phi)"]-train_x["DNN_2"]
    
    #  ~min=train_x["genMET_Y"].min()
    #  ~max=train_x["genMET_Y"].max()
    #  ~plt.figure()
    #  ~train_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y")
    #  ~train_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/METdistr_y_train.pdf")
    
    #  ~# Compare corrected MET to genMET pT - validation sample
    #  ~val_x["genMET"]=np.sqrt(val_x["genMET_X"]**2+val_x["genMET_Y"]**2)
    #  ~val_x["DNN_MET"]=np.sqrt(val_x["DNN_MET_X"]**2+val_x["DNN_MET_Y"]**2)
    
    #  ~min=val_x["genMET"].min()
    #  ~max=val_x["genMET"].max()
    #  ~plt.figure()
    #  ~val_x["genMET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET")
    #  ~val_x["DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_val.pdf")
    
    #  ~# Compare corrected MET to genMET pT - training sample
    #  ~train_x["genMET"]=np.sqrt(train_x["genMET_X"]**2+train_x["genMET_Y"]**2)
    #  ~train_x["DNN_MET"]=np.sqrt(train_x["DNN_MET_X"]**2+train_x["DNN_MET_Y"]**2)
    
    #  ~min=train_x["genMET"].min()
    #  ~max=train_x["genMET"].max()
    #  ~plt.figure()
    #  ~train_x["genMET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET")
    #  ~train_x["DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_train.pdf")
    
    #  ~# Compare resolution for corrected MET and Puppi - validation
    #  ~val_x["PuppiMET"]=np.sqrt(val_x["PuppiMET*cos(PuppiMET_phi)"]**2+val_x["PuppiMET*sin(PuppiMET_phi)"]**2)
    #  ~val_x["genMET-PuppiMET"]=val_x["genMET"]-val_x["PuppiMET"]
    #  ~val_x["genMET-DNN_MET"]=val_x["genMET"]-val_x["DNN_MET"]
    
    #  ~min=val_x["genMET-PuppiMET"].min()
    #  ~max=val_x["genMET-PuppiMET"].max()
    #  ~plt.figure()
    #  ~val_x["genMET-DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET-DNN_MET")
    #  ~val_x["genMET-PuppiMET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET-PuppiMET")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_val.pdf")
    
    #  ~print("Mean Puppi val:", val_x["genMET-PuppiMET"].mean())
    #  ~print("RMS Puppi val:", val_x["genMET-PuppiMET"].std())
    #  ~print("Mean DNN val:", val_x["genMET-DNN_MET"].mean())
    #  ~print("RMS DNN val:", val_x["genMET-DNN_MET"].std())
    
    #  ~# Compare resolution for corrected MET and Puppi - trainidation
    #  ~train_x["PuppiMET"]=np.sqrt(train_x["PuppiMET*cos(PuppiMET_phi)"]**2+train_x["PuppiMET*sin(PuppiMET_phi)"]**2)
    #  ~train_x["genMET-PuppiMET"]=train_x["genMET"]-train_x["PuppiMET"]
    #  ~train_x["genMET-DNN_MET"]=train_x["genMET"]-train_x["DNN_MET"]
    
    #  ~min=train_x["genMET-PuppiMET"].min()
    #  ~max=train_x["genMET-PuppiMET"].max()
    #  ~plt.figure()
    #  ~train_x["genMET-DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET-DNN_MET")
    #  ~train_x["genMET-PuppiMET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET-PuppiMET")
    #  ~plt.legend()
    #  ~plt.savefig("outputComparison/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_train.pdf")
    
    #  ~print("Mean Puppi train:", train_x["genMET-PuppiMET"].mean())
    #  ~print("RMS Puppi train:", train_x["genMET-PuppiMET"].std())
    #  ~print("Mean DNN train:", train_x["genMET-DNN_MET"].mean())
    #  ~print("RMS DNN train:", train_x["genMET-DNN_MET"].std())
    
#############################################################

if __name__ == "__main__":
    #  ~print("---------------------------No negative contraint on output----------------------------")
    # Define input data path
    #  ~dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v01/minTrees/100.0/TTbar_diLepton.root"
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v01/minTrees/100.0/TTbar_amcatnlo.root"

    # Define Input Variables
    inputVars = ["PuppiMET","METunc_Puppi","MET","HT","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","PFMET_phi","PuppiMET_phi","CaloMET","CaloMET_phi","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj","HT_phi"]

    #  ~trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_pTphi_30EP","TTbar_amcatnlo","diff_pTphi",["PuppiMET-genMET","dPhi(PuppiMET_phi,genMET_phi)"],updateInput=True)
    #  ~trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_pTphi","TTbar_amcatnlo","diff_pTphi",["PuppiMET-genMET","dPhi(PuppiMET_phi,genMET_phi)"],updateInput=True)
    #  ~trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_pTphi_30EP_MinMaxScaled","TTbar_amcatnlo","diff_pTphi",["PuppiMET-genMET","dPhi(PuppiMET_phi,genMET_phi)"],updateInput=True,normalize=True)
    
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2D/Inlusive_amcatnlo_pTphi_30EP__diff_pTphi_2018_20210517-1443normDistr","TTbar_amcatnlo","diff_pTphi",["PuppiMET-genMET","dPhi(PuppiMET_phi,genMET_phi)"])
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2D/Inlusive_amcatnlo_pTphi__diff_pTphi_2018_20210518-0918normDistr","TTbar_amcatnlo","diff_pTphi",["PuppiMET-genMET","dPhi(PuppiMET_phi,genMET_phi)"])
    plot_Output(dataPath,inputVars,"trainedModel_Keras/2D/Inlusive_amcatnlo_pTphi_30EP_MinMaxScaled__diff_pTphi_2018_20210518-1541normDistr","TTbar_amcatnlo","diff_pTphi",["PuppiMET-genMET","dPhi(PuppiMET_phi,genMET_phi)"],updateInput=True,normalize=True)

