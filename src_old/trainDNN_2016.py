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
    model.add(Dense(53, input_dim=53))
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
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    #  ~model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    #  ~model.compile(loss="logcosh", optimizer=Adam(lr=0.001),metrics=['mean_squared_error','mean_absolute_percentage_error'])
    model.compile(loss="logcosh", optimizer=Adam(lr=0.0001),metrics=['mean_squared_error','mean_absolute_percentage_error'])
    #  ~model.compile(loss="mean_absolute_percentage_error", optimizer=Adam(lr=0.0001),metrics=['mean_squared_error','logcosh'])
    #  ~model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
    #  ~model.compile(loss="huber_loss", optimizer=Adam(lr=0.001))
    return model

def getInputArray_allBins_patched(path,inputVars,target,update,treeName):
    inputVars.append(target)
    if target!="genMET/PuppiMET":
        outputPath="input/"+treeName+"_"+target+".pkl"
    else:
        outputPath="input/"+treeName+".pkl"
    if update:
        root_file = uproot.open(path)
        events = root_file["ttbar_res100.0;1"][treeName+";1"]
        cut = "(PuppiMET>0)"
        inputFeatures = events.arrays(inputVars,cut,library="pd")
        Bin1=inputFeatures.loc[inputFeatures["PuppiMET"]<40]
        Bin2=inputFeatures.loc[(inputFeatures["PuppiMET"]>40) & (inputFeatures["PuppiMET"]<80)]
        Bin3=inputFeatures.loc[(inputFeatures["PuppiMET"]>80) & (inputFeatures["PuppiMET"]<120)]
        Bin4=inputFeatures.loc[(inputFeatures["PuppiMET"]>120) & (inputFeatures["PuppiMET"]<160)]
        Bin5=inputFeatures.loc[(inputFeatures["PuppiMET"]>160) & (inputFeatures["PuppiMET"]<230)]
        Bin6=inputFeatures.loc[inputFeatures["PuppiMET"]>230]
        
        numEvt=Bin6.shape[0]
        
        combined=Bin1.head(int(numEvt))
        combined=combined.append(Bin2.head(int(numEvt)))
        combined=combined.append(Bin3.head(int(numEvt)))
        combined=combined.append(Bin4.head(int(numEvt)))
        combined=combined.append(Bin5.head(int(numEvt)))
        combined=combined.append(Bin6.head(int(numEvt)))
        
        combined.to_pickle(outputPath)
    else:
        combined=pd.read_pickle(outputPath)
        
    return combined[inputVars[:-1]],combined[inputVars[-1]],inputVars
    
def getInputArray_allBins_nomDistr(path,inputVars,target,update,treeName,normalize=False,standardize=False):
    inputVars.append(target)
    
    if target!="genMET/PuppiMET":   #check target
        outputPath="input/"+treeName+"_"+target+"_nomDistr.pkl"
    else:
        outputPath="input/"+treeName+"_nomDistr.pkl"
        
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
        if target=="PtNuNu":
            cut+="&(PtNuNu>0)"
        if only_emu:
            cut+="&(emu==1)"
        inputFeatures = events.arrays(inputVars,cut,library="pd")
        
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
    
    print("Train with:",outputPath)
        
    return inputFeatures[inputVars[:-1]],inputFeatures[inputVars[-1]],inputVars

def getInputArray_allBins_genMETweighted(path,inputVars,target,update,treeName,normalize=False,standardize=False):
    inputVars.append(target)
    inputVars.append("genMET")
    
    if target!="genMET/PuppiMET":   #check target
        outputPath="input/"+treeName+"_"+target+"_genMETweighted.pkl"
    else:
        outputPath="input/"+treeName+"_genMETweighted.pkl"
        
    if treeName.split("_")[-1]=="emu":
        only_emu=True
        treeName=treeName.replace("_emu","")
    else:
        only_emu=False
        
    if normalize:
        outputPath=outputPath.replace("_genMETweighted","normalized_genMETweighted")
    if standardize:
        outputPath=outputPath.replace("_genMETweighted","standardized_genMETweighted")
        
    if update:
        root_file = uproot.open(path)
        events = root_file["ttbar_res100.0;1"][treeName+";1"]
        cut = "(PuppiMET>0)"
        if target=="PtNuNu":
            cut+="&(PtNuNu>0)"
        if only_emu:
            cut+="&(emu==1)"
        inputFeatures = events.arrays(inputVars,cut,library="pd")
        
        if normalize:
            scaler = MinMaxScaler()
            scaledFeatures = scaler.fit_transform(inputFeatures.values)
            inputFeatures = pd.DataFrame(scaledFeatures, index=inputFeatures.index, columns=inputFeatures.columns)
        elif standardize:
            scaler = StandardScaler()
            scaledFeatures = scaler.fit_transform(inputFeatures.values)
            inputFeatures = pd.DataFrame(scaledFeatures, index=inputFeatures.index, columns=inputFeatures.columns)
        
        #add bin number to dataframe
        #  ~bins=list(range(0,305,5))
        bins=list(range(0,300,20))
        bins.append(inputFeatures["genMET"].max())
        labels=list(range(1,len(bins)))
        inputFeatures["genMET_binNR"] = pd.cut(inputFeatures["genMET"], bins=bins, labels=labels)
        
        sample_weight=compute_sample_weight("balanced",inputFeatures["genMET_binNR"])
        
        inputFeatures["weight"]=sample_weight
        
        inputFeatures.to_pickle(outputPath)
    else:
        inputFeatures=pd.read_pickle(outputPath)
    
    print("Train with:",outputPath)
        
    return inputFeatures[inputVars[:-2]],inputFeatures[target],inputVars[:-1],inputFeatures["weight"]

def trainKeras(dataPath,inputVars,name,treeName,target="genMET/PuppiMET",patchedTrain=True,genMETweighted=False,updateInput=False,permuationImportance=False,normalize=False,standardize=False):
        stringStart=datetime.datetime.now().strftime("%Y%m%d-%H%M")
        
        if target!="genMET/PuppiMET":
            modelName=name+"_"+"_"+target+"_2016_"+stringStart
        else:
            modelName=name+"_2016_"+stringStart
        
        if patchedTrain:
            x,y,inputVars=getInputArray_allBins_patched(dataPath,inputVars,target,updateInput,treeName,normalize,standardize)
        elif genMETweighted:
            x,y,inputVars,sampleWeights=getInputArray_allBins_genMETweighted(dataPath,inputVars,target,updateInput,treeName,normalize,standardize)
            modelName+="genMETweighted"
        else:
            x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,target,updateInput,treeName,normalize,standardize)
            modelName+="normDistr"
        
        if genMETweighted:
            train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(x, y, sampleWeights, random_state=30, test_size=0.2, train_size=0.8)
        else:
            train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
        
        #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        logdir="./logs/"+modelName
        tensorboard_callback = TensorBoard(log_dir=logdir)
        
        #  ~my_model = KerasRegressor(build_fn=baseline_Model, epochs=40, batch_size=64, verbose=1)
        #  ~my_model = KerasRegressor(build_fn=bJetRegression_Model, epochs=40, batch_size=512, verbose=1)
        #  ~my_model = KerasRegressor(build_fn=bJetRegression_Model, epochs=200, batch_size=5000, verbose=2)
        my_model = KerasRegressor(build_fn=bJetRegression_Model, epochs=50, batch_size=5000, verbose=2)
        if genMETweighted:
            my_model.fit(train_x,train_y,validation_data=(val_x,val_y,np.sqrt(val_weights)),sample_weight=np.sqrt(train_weights),callbacks=[tensorboard_callback])
        else:
            my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[tensorboard_callback])
        #  ~my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[es])
        
        if permuationImportance:
            perm = PermutationImportance(my_model, random_state=1).fit(train_x,train_y)
            output = eli5.format_as_text(eli5.explain_weights(perm, target_names = target,feature_names = inputVars[:-1]))
            print(name)
            print(output)
            output_file = open("PermutationImportance/"+modelName+".txt", "w")
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
        my_model.model.save('trainedModel_Keras/2016/'+modelName)
        my_model.model.save('trainedModel_Keras/2016/'+modelName+".h5")

def shapleyValues(dataPath,inputVars,modelPath,treeName,target,patchedTrain=True,updateInput=False):     #Shapley Values for Trained DNN
        
    if patchedTrain:
        x,y,inputVars=getInputArray_allBins_patched(dataPath,inputVars,target,updateInput,treeName)
    else:
        x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,target,updateInput,treeName)
    
    x_test=x.sample(n=10000)
    x_test=x_test.to_numpy()
    
    model = load_model(modelPath+".h5")
    
    ex = shap.GradientExplainer(model, x_test)
    shap_values = ex.shap_values(x_test)
    max_display = x_test.shape[1]
    figure = plt.gcf()  # get current figure
    shap.summary_plot(shap_values, x_test, plot_type = "bar", feature_names = inputVars[:-1], max_display = max_display, show=False)
    figure.set_size_inches(32, 18)
    plt.savefig("ShapleyValues/shap_{0}.pdf".format(modelPath.split("/")[-1]))
    
def plot_Output(dataPath,inputVars,modelPath,treeName,target,patchedTrain=True,updateInput=False,normalize=False,standardize=False):
    if not os.path.exists("outputComparison/"+modelPath.split("/")[1]):
        os.makedirs("outputComparison/"+modelPath.split("/")[1])
    
    if patchedTrain:
        x,y,inputVars=getInputArray_allBins_patched(dataPath,inputVars,target,updateInput,treeName,normalize,standardize)
    else:
        x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,target,updateInput,treeName,normalize,standardize)
    
    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
    
    model = load_model(modelPath+".h5")
    
    #  ~train_x=train_x.head(100000)
    #  ~val_x=val_x.head(100000)
    y_hat_train = model.predict(train_x,use_multiprocessing=True)
    y_hat_val = model.predict(val_x,use_multiprocessing=True)
    
    train_x["DNN"]=y_hat_train
    train_x[target]=train_y
    
    print(train_x)
    
    # Compare DNN and target - training sample
    plt.figure()
    train_x[["DNN",target]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/train.pdf")
    
    val_x["DNN"]=y_hat_val
    val_x[target]=val_y
    
    print(val_x)
    
    # Compare DNN and target - validation sample
    plt.clf()
    val_x[["DNN",target]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/val.pdf")
    
    # Compare DNN output between both samples
    min=train_x["DNN"].min()
    max=train_x["DNN"].max()
    plt.clf()
    train_x["DNN"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x["DNN"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/DNNcomparison.pdf")
    
    # Compare target between both samples
    plt.clf()
    train_x[target].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x[target].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/TARGETcomparison.pdf")
    
    # Compare corrected MET to genMET - validation sample
    val_x["genMET"]=val_x["PuppiMET"]-val_x[target]
    val_x["DNN_MET"]=val_x["PuppiMET"]-val_x["DNN"]
    
    min=val_x["genMET"].min()
    max=val_x["genMET"].max()
    plt.clf()
    val_x["genMET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET")
    val_x["DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET")
    plt.legend()
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/METdistr_val.pdf")
    
    # Compare corrected MET to genMET - training sample
    if target=="PuppiMET-genMET":
        train_x["genMET"]=train_x["PuppiMET"]-train_x[target]
        train_x["DNN_MET"]=train_x["PuppiMET"]-train_x["DNN"]
    else:
        train_x["DNN_MET"]=train_x["DNN"]
    
    min=train_x["genMET"].min()
    max=train_x["genMET"].max()
    plt.clf()
    train_x["genMET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET")
    train_x["DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET")
    plt.legend()
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/METdistr_train.pdf")
    
    # 2D plot Puppi vs. DNN output - training sample
    plt.clf()
    train_x.plot.hexbin(x="PuppiMET", y="DNN", gridsize=100, bins="log")
    ax = plt.gca()
    ax.axline((1, 1), slope=1)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/2D_DNN_Puppi.pdf")
    
    # 2D plot GenMET vs. DNN output - training sample
    plt.clf()
    train_x.plot.hexbin(x="genMET", y="DNN", gridsize=100, bins="log")
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/2D_DNN_GenMET.pdf")
    
    # 2D plot Puppi vs. DNN_MET - training sample
    plt.clf()
    train_x.plot.hexbin(x="PuppiMET", y="DNN_MET", gridsize=100, bins="log")
    ax = plt.gca()
    ax.axline((1, 1), slope=1)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/2D_DNNmet_Puppi.pdf")
    
    # 2D plot GenMET vs. DNN_MET - training sample
    plt.clf()
    train_x.plot.hexbin(x="genMET", y="DNN_MET", gridsize=100, bins="log")
    ax = plt.gca()
    ax.axline((1, 1), slope=1)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/2D_DNNmet_GenMET.pdf")
    
    # 2D plot Target vs. DNN output - training sample
    plt.clf()
    train_x.plot.hexbin(x=target, y="DNN", gridsize=100, bins="log")
    ax = plt.gca()
    ax.axline((1, 1), slope=1)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/2D_DNN_Target.pdf")
    
    # 2D plot Target vs. Puppi - training sample
    plt.clf()
    train_x.plot.hexbin(x="PuppiMET", y=target, gridsize=100, bins="log")
    ax = plt.gca()
    ax.axline((1, 1), slope=1)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/2D_Target_Puppi.pdf")
    
    # 2D plot Target vs. Puppi - training sample
    plt.clf()
    train_x.plot.hexbin(x="genMET", y=target, gridsize=100, bins="log")
    ax = plt.gca()
    ax.axline((1, 1), slope=1)
    plt.savefig("outputComparison/"+modelPath.split("/")[1]+"/2D_Target_GenMET.pdf")
    
    
#############################################################

if __name__ == "__main__":
    #  ~print("---------------------------No negative contraint on output----------------------------")
    # Define input data path
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2016/v24/minTrees/100.0/TTbar_amcatnlo.root"

    # Define Input Variables
    inputVars = ["PuppiMET","METunc_Puppi","MET","HT","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt","Lep1_phi","Lep1_eta","Lep1_E","Lep2_pt","Lep2_phi","Lep2_eta","Lep2_E","Jet1_pt","Jet1_phi","Jet1_eta","Jet1_E","Jet2_pt","Jet2_phi","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","PFMET_phi","PuppiMET_phi","CaloMET","CaloMET_phi","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    #  ~trainKeras(dataPath,inputVars,"Inlusive_amcatnlo","TTbar_amcatnlo","PuppiMET-genMET",patchedTrain=False,updateInput=True)
    
    shapleyValues(dataPath,inputVars,"trainedModel_Keras/Inlusive_amcatnlo__PuppiMET-genMET_2018_20210427-1222normDistr","TTbar_amcatnlo","PuppiMET-genMET",patchedTrain=False)
    
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/Inlusive_amcatnlo__PuppiMET-genMET_2018_20210427-1222normDistr","TTbar_amcatnlo","PuppiMET-genMET",patchedTrain=False)
