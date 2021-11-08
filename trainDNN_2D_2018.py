from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
from subprocess import call
from os.path import isfile
import sys
import struct
import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#  ~os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from tensorflow.keras import Sequential, regularizers
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
    
# model roughly taken from https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=HIG-18-027
def bJetRegression_Model():
    dout = 0.3
    #  ~regLamb = 0.05
    
    model = Sequential()
    model.add(BatchNormalization())
    #  ~model.add(Dense(53, input_dim=53))
    #  ~model.add(Dense(54, input_dim=54, kernel_regularizer=regularizers.l2(regLamb)))
    model.add(Dense(54, input_dim=54))
    model.add(BatchNormalization())
    
    #  ~model.add(Dense(1024, kernel_regularizer=regularizers.l2(regLamb)))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(dout))
    model.add(LeakyReLU(alpha=0.2))
    
    #  ~model.add(Dense(1024, kernel_regularizer=regularizers.l2(regLamb)))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(dout))
    model.add(LeakyReLU(alpha=0.2))
    
    #  ~model.add(Dense(1024, kernel_regularizer=regularizers.l2(regLamb)))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(dout))
    model.add(LeakyReLU(alpha=0.2))
    
    #  ~model.add(Dense(512, kernel_regularizer=regularizers.l2(regLamb)))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dropout(dout))
    model.add(LeakyReLU(alpha=0.2))
    
    #  ~model.add(Dense(256, kernel_regularizer=regularizers.l2(regLamb)))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Dropout(dout))
    model.add(LeakyReLU(alpha=0.2))
    
    #  ~model.add(Dense(128, kernel_regularizer=regularizers.l2(regLamb)))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Dropout(dout))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(2, kernel_initializer='normal', activation='linear'))
    model.compile(loss="logcosh", optimizer=Adam(lr=0.0001),metrics=['mean_squared_error','mean_absolute_percentage_error',"logcosh"])
    return model

# function to get input from pkl if available, if update=true the pkl is constructed from the root file  
def getInputArray_allBins_nomDistr(path,inputVars,targetName,target,update,treeName,normalize=False,standardize=False,genMETweighted=False):
	
	
    for var in target:      # append target variables to be also read in
        inputVars.append(var)
    inputVars.append("genMET")
    
    if not os.path.exists("input/2018/2D/"):    # create output folder for plots if not available
        os.makedirs("input/2018/2D/")
    
    outputPath="input/2018/2D/"+treeName+"_"+targetName+"_nomDistr.pkl"     # output path of pkl
    
    
    
    # option to only use emu events for training (mainly for studies connected to 40 GeV cut)
    if treeName.split("_")[-1]=="emu":
        only_emu=True
        treeName=treeName.replace("_emu","")
    else:
        only_emu=False
    
    # rename pkl if input is normalized or standardized
    if normalize:
        outputPath=outputPath.replace("_nomDistr","normalized_nomDistr")
    if standardize:
        outputPath=outputPath.replace("_nomDistr","standardized_nomDistr")
    
    # if update=true new pkl is created from root file (takes much longer than using existing pkl)
    if update:
        root_file = uproot.open(path)
        events = root_file["ttbar_res100.0;1"][treeName+";1"]
        cut = "(PuppiMET>0)"    # use only events selected by reco selection
        if only_emu:
            cut+="&(emu==1)"    # use only emu events if selected
        inputFeatures = events.arrays(inputVars,cut,library="pd")   # load inputs from rootfile
        
        if normalize:   # normalize inputs
            scaler = MinMaxScaler()
            scaledFeatures = scaler.fit_transform(inputFeatures.values)
            inputFeatures = pd.DataFrame(scaledFeatures, index=inputFeatures.index, columns=inputFeatures.columns)
        elif standardize:   # standardize inputs
            scaler = StandardScaler()
            scaledFeatures = scaler.fit_transform(inputFeatures.values)
            inputFeatures = pd.DataFrame(scaledFeatures, index=inputFeatures.index, columns=inputFeatures.columns)
        
        if genMETweighted:
            bins=list(range(0,500,5))
            bins.append(inputFeatures["genMET"].max())
            labels=list(range(1,len(bins)))
            inputFeatures["genMET_binNR"] = pd.cut(inputFeatures["genMET"], bins=bins, labels=labels)
            sample_weight=compute_sample_weight("balanced",inputFeatures["genMET_binNR"])
            inputFeatures["weight"]=sample_weight
        
        inputFeatures.to_pickle(outputPath)     # write inputs to pkl
    else:
        inputFeatures=pd.read_pickle(outputPath)    # read inputs from existing pkl
    
    print("Train with:",outputPath)
    print(inputVars)
    # returns nD array of inputs, 2D array of targets and vector of inputNames
    if genMETweighted: return inputFeatures[inputVars[:-3]],inputFeatures[target],inputVars[:-1],inputFeatures["weight"],inputFeatures["genMET"]
    else: return inputFeatures[inputVars[:-3]],inputFeatures[target],inputVars[:-1],

# function to train model
def trainKeras(dataPath,inputVars,name,treeName,targetName,target,updateInput=False,permuationImportance=False,normalize=False,standardize=False,genMETweighted=False):
            
        # set number of epochs and batchsize
        epochs = 50
        batch_size = 5000
        
        stringStart=datetime.datetime.now().strftime("%Y%m%d-%H%M")

        modelName=name+"_"+"_"+targetName+"_2018_"+stringStart      # time added to name of trained model 
        
        if genMETweighted:
            x,y,inputVars,sampleWeights,_=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
            modelName+="genMETweighted"
        else:
            x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
		    
        # split sample to into training and test sample (random_state produces reproducible splitting)
        #  ~if genMETweighted:
            #  ~train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(x, y, sampleWeights, random_state=30, test_size=0.2, train_size=0.8)
        #  ~else:
            #  ~train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
        if genMETweighted:
            train_x, test_x, train_y, test_y, train_weights, test_weights = train_test_split(x, y, sampleWeights, random_state=30, test_size=0.2, train_size=0.8)
            train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(train_x, train_y, train_weights, random_state=30, test_size=0.25, train_size=0.75)
        else:
            train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, random_state=30, test_size=0.25, train_size=0.75)
        #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)   # could be used for early stopping of training
        logdir="./logs/2018/2D/"+modelName
        tensorboard_callback = TensorBoard(log_dir=logdir)      # setup tensorboard to log training progress
        
        #  ~print("###############################\n min and max of train weights:")
        #  ~print(min(train_weights), max(train_weights))
        
        # setup keras and train model
        my_model = KerasRegressor(build_fn=bJetRegression_Model, epochs=epochs, batch_size=batch_size, verbose=2)
        if genMETweighted:
            my_model.fit(train_x,train_y,validation_data=(val_x,val_y,val_weights),sample_weight=train_weights,callbacks=[tensorboard_callback])
            #  ~my_model.fit(train_x,train_y,validation_data=(val_x,val_y,np.sqrt(val_weights)),sample_weight=np.sqrt(train_weights),callbacks=[tensorboard_callback])
        else:
            my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[tensorboard_callback])
        
        # derive permutation importance, if selected
        if permuationImportance:
            perm = PermutationImportance(my_model, random_state=1).fit(train_x,train_y)
            output = eli5.format_as_text(eli5.explain_weights(perm, target_names = target,feature_names = inputVars[:-1]))
            print(name)
            print(output)
            output_file = open("PermutationImportance/2018/2D/"+modelName+".txt", "w")
            output_file.write(output)
            output_file.close()
        
        # evaluate model with training and validation sample
        y_hat_train = my_model.predict(train_x)
        y_hat_test = my_model.predict(val_x)

        # display error values
        print ('Train RMSE: ', round(np.sqrt(((train_y - y_hat_train)**2).mean()), 4))    
        print ('Train MEAN: ', round(((train_y - y_hat_train).mean()), 4))    
        print ('Test RMSE: ', round(np.sqrt(((val_y - y_hat_test)**2).mean()), 4))
        print ('Test MEAN: ', round(((val_y - y_hat_test).mean()), 4))
        
        #save Model
        my_model.model.save('trainedModel_Keras/2018/2D/'+modelName)
        my_model.model.save('trainedModel_Keras/2018/2D/'+modelName+".h5")

# function to derive shapley values for trained model
def shapleyValues(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False):
    
    # get inputs
    x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName)
    
    # use only n events for deriving shapley values (full statistics takes way too long!)
    x_test=x.sample(n=10000)
    x_test=x_test.to_numpy()
    
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
    plt.savefig("ShapleyValues/2018/2D/shap_{0}.pdf".format(modelPath.split("/")[-1]))

# function to derive different control performance plots based on trained model
def plot_Output(dataPath,inputVars,modelPath,treeName,targetName,target,updateInput=False,normalize=False,standardize=False,genMETweighted=False):
    
    if not os.path.exists("outputComparison/2018/2D/"+modelPath.split("/")[-1]):    # create output folder for plots if not available
        os.makedirs("outputComparison/2018/2D/"+modelPath.split("/")[-1])
    
    # get inputs
    #  ~if genMETweighted:
        #  ~x,y,inputVars,sample_weights,genMET=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
        #  ~train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(x, y, sample_weights, random_state=30, test_size=0.2, train_size=0.8)
    #  ~else:
        #  ~x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
        #  ~train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
    if genMETweighted:
        x,y,inputVars,sample_weights,genMET=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
        train_x, test_x, train_y, test_y, train_weights, test_weights = train_test_split(x, y, sampleWeights, random_state=30, test_size=0.2, train_size=0.8)
        train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(train_x, train_y, train_weights, random_state=30, test_size=0.25, train_size=0.75)
    else:
        x,y,inputVars=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=30, test_size=0.2, train_size=0.8)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, random_state=30, test_size=0.25, train_size=0.75)
    
    # perform same splitting as in training (random_state!)
    
    model = load_model(modelPath+".h5")
    
    #  ~train_x=train_x.head(100000)    # could be used to only create plots for limited statistics (mainly for debugging)
    #  ~val_x=val_x.head(100000)
    
    # evaluate trained model
    y_hat_train = model.predict(train_x,use_multiprocessing=True)
    y_hat_val = model.predict(val_x,use_multiprocessing=True)
    
    train_x["DNN_1"]=[row[0] for row in y_hat_train]
    train_x["DNN_2"]=[row[1] for row in y_hat_train]
    train_x[target[0]]=train_y[target[0]]
    train_x[target[1]]=train_y[target[1]]
    
    train_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/trainResults.pkl")
    # print(train_x)
    
    val_x["DNN_1"]=[row[0] for row in y_hat_val]
    val_x["DNN_2"]=[row[1] for row in y_hat_val]
    val_x[target[0]]=val_y[target[0]]
    val_x[target[1]]=val_y[target[1]]
    
    val_x.to_pickle("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/valResults.pkl")
    """
    plt.figure()
    train_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/train_x.pdf")
    plt.figure()
    train_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/train_y.pdf")
    
    
    # Compare DNN and target - validation sample
    plt.figure()
    val_x[["DNN_1",target[0]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/val_x.pdf")
    plt.figure()
    val_x[["DNN_2",target[1]]].plot.hist(alpha=0.5,bins=500,density=True)
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/val_y.pdf")
    
    # Compare DNN output between both samples
    min=train_x["DNN_1"].min()
    max=train_x["DNN_1"].max()
    plt.figure()
    train_x["DNN_1"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x["DNN_1"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/DNNcomparison_x.pdf")
    min=train_x["DNN_2"].min()
    max=train_x["DNN_2"].max()
    plt.figure()
    train_x["DNN_2"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x["DNN_2"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/DNNcomparison_y.pdf")
    
    # Compare target between both samples
    plt.figure()
    train_x[target[0]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x[target[0]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/TARGETcomparison_x.pdf")
    plt.figure()
    train_x[target[1]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Train")
    val_x[target[1]].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="Valid")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/TARGETcomparison_y.pdf")
    
    # Compare corrected MET to genMET X - validation sample
    val_x["genMET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x[target[0]]
    val_x["DNN_MET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x["DNN_1"]
    
    min=val_x["genMET_X"].min()
    max=val_x["genMET_X"].max()
    plt.figure()
    val_x["genMET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_X")
    val_x["DNN_MET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_X")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_x_val.pdf")
    
    # Compare corrected MET to genMET X - training sample
    train_x["genMET_X"]=train_x["PuppiMET*cos(PuppiMET_phi)"]-train_x[target[0]]
    train_x["DNN_MET_X"]=train_x["PuppiMET*cos(PuppiMET_phi)"]-train_x["DNN_1"]
    
    min=train_x["genMET_X"].min()
    max=train_x["genMET_X"].max()
    plt.figure()
    train_x["genMET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_X")
    train_x["DNN_MET_X"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_X")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_x_train.pdf")
    
    # Compare corrected MET to genMET Y - validation sample
    val_x["genMET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x[target[1]]
    val_x["DNN_MET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x["DNN_2"]
    
    min=val_x["genMET_Y"].min()
    max=val_x["genMET_Y"].max()
    plt.figure()
    val_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y")
    val_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_y_val.pdf")
    
    # Compare corrected MET to genMET Y - training sample
    train_x["genMET_Y"]=train_x["PuppiMET*sin(PuppiMET_phi)"]-train_x[target[1]]
    train_x["DNN_MET_Y"]=train_x["PuppiMET*sin(PuppiMET_phi)"]-train_x["DNN_2"]
    
    min=train_x["genMET_Y"].min()
    max=train_x["genMET_Y"].max()
    plt.figure()
    train_x["genMET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="genMET_Y")
    train_x["DNN_MET_Y"].plot.hist(alpha=0.5,bins=500,range=(min,max),density=True,label="DNN_MET_Y")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_y_train.pdf")
    
    # Compare corrected MET to genMET pT - validation sample
    val_x["genMET"]=np.sqrt(val_x["genMET_X"]**2+val_x["genMET_Y"]**2)
    val_x["DNN_MET"]=np.sqrt(val_x["DNN_MET_X"]**2+val_x["DNN_MET_Y"]**2)
    
    min_x=0
    max_x=400

    subRatio = dict(height_ratios=[5,3])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=subRatio)
    fig.suptitle("validation sample")
    countsDNN,bins = np.histogram(val_x["DNN_MET"],bins=500,range=(min_x,max_x))
    countsGen,bins = np.histogram(val_x["genMET"],bins=500,range=(min_x,max_x))
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
    plt.tight_layout()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_val.pdf")
    
    
    # Compare corrected MET to genMET pT - training sample
    train_x["genMET"]=np.sqrt(train_x["genMET_X"]**2+train_x["genMET_Y"]**2)
    train_x["DNN_MET"]=np.sqrt(train_x["DNN_MET_X"]**2+train_x["DNN_MET_Y"]**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=subRatio)
    fig.suptitle("training sample")
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
    plt.tight_layout()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/METdistr_pT_train.pdf")
    
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
        plt.tight_layout()
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_val"+split_str[i]+".pdf")
        
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
        plt.suptitle("training sample; genMET range: "+split_str[i])
        train_temp["genMET-DNN_MET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="DNN", color="b")
        train_temp["genMET-PuppiMET"].plot.hist(alpha=0.5,bins=500,range=(min_x,max_x),density=True,label="Puppi", color="r")
        plt.text(-148, 0.022, puppiMeanTrain, bbox={"facecolor":"none", "pad":5})
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("genMET-recoMET [GeV]")
        plt.ylabel("Normalized Counts")
        plt.ylim(0,0.028)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/Resolution_pT_train"+split_str[i]+".pdf")
        
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
    plt.tight_layout()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/MeanDiff_vs_genMET_train.pdf")
    
    binsMET = np.linspace(0,600,30)
    val_x["bin"] = np.digitize(val_x["genMET"],bins=binsMET)
    res_DNN_MET = val_x.groupby("bin")["genMET-DNN_MET"].agg(["mean",rmsFUNC,meanErr])
    res_PuppiMET = val_x.groupby("bin")["genMET-PuppiMET"].agg(["mean",rmsFUNC,meanErr])
    res_DNN_MET["metBins"] = binsMET
    plt.figure()
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
    plt.tight_layout()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/MeanDiff_vs_genMET_val.pdf")
    
    bins=list(range(0,500,1))
    bins.append(genMET.max())
    
    
    plt.figure()
    train_x["genMET"].plot.hist(alpha=0.3, color="blue", bins=bins, density=True, weights=train_weights, label="genMET with weights")
    train_x["genMET"].plot.hist(alpha=0.3, color="red", bins=10000, density=True, weights=train_weights, label="genMET with weights other binning")
    train_x["genMET"].plot.hist(alpha=0.3, color="green", bins=300, density=True, label="genMET without weights")
    plt.legend()
    plt.savefig("outputComparison/2018/2D/"+modelPath.split("/")[-1]+"/genMET_test_train.pdf")
    """
#############################################################

if __name__ == "__main__":
    # Define input data path
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v04/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"

    # Define Input Variables
    inputVars = ["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True,genMETweighted=True)

    #  ~shapleyValues(dataPath,inputVars,"trainedModel_Keras/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20210519-1014normDistr","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"])
    
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20210519-1014normDistr","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1017genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1141genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1509genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)

