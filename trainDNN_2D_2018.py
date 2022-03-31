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

from CustomEarlyStopping import EarlyStoppingCombined

def rmsFUNC(x):
    return np.sqrt(np.mean(np.square(x)))

def meanErr(x):
    return 2*np.std(x)/np.sqrt(len(x))
    
# model roughly taken from https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=HIG-18-027
def bJetRegression_Model(lr=0.0001, dout=0.3, lamb=0.05, nLayer=6, nodeFac=1., alph=0.2):
    #  ~dout = 0.3
    regLamb = lamb
    nodes = np.array(np.array([128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024])*nodeFac, dtype=int)
    
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(54, input_dim=54, kernel_regularizer=regularizers.l2(regLamb)))
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

# function to get input from pkl if available, if update=true the pkl is constructed from the root file  
def getInputArray_allBins_nomDistr(path,inputVars,targetName,target,update,treeName,normalize=False,standardize=False,genMETweighted=False):
	
    appendCount = 0
    for subList in [target, ["genMET", "PtNuNu", "PtNuNu_phi", "Lep1_phi", "Lep2_phi", "PuppiMET", "PuppiMET_phi"]]:      # append target variables to be also read in
        for var in subList:
            inputVars.append(var)
            appendCount+=1
    #  ~for var in target:      # append target variables to be also read in
        #  ~inputVars.append(var)
    #  ~inputVars.append("genMET")
    
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
    if genMETweighted: return inputFeatures[inputVars[:-appendCount]],inputFeatures[target],inputVars[:-appendCount],inputFeatures["weight"],inputFeatures[inputVars[2-appendCount:]]
    else: return inputFeatures[inputVars[:-appendCount]],inputFeatures[target],inputVars[:-appendCount],inputFeatures[inputVars[2-appendCount:]]

# function to train model
def trainKeras(dataPath,inputVars,name,treeName,targetName,target, lr, dout, lamb, batch, nLayer, nodeFac, alph,updateInput=False,permuationImportance=False,normalize=False,standardize=False,genMETweighted=False):
            
        # set number of epochs and batchsize
        epochs = 100
        #  ~batch_size = 5000
        batch_size = batch
        
        stringStart=datetime.datetime.now().strftime("%Y%m%d-%H%M")

        modelName=name+"_"+"_"+targetName+"_2018_"+stringStart      # time added to name of trained model 
        
        if genMETweighted:
            x,y,inputVars,sampleWeights,metVals=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
            modelName+="genMETweighted"
        else:
            x,y,inputVars,metVals=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
		    
        # split sample to into training and test sample (random_state produces reproducible splitting)
        if genMETweighted:
            train_x, test_x, train_y, test_y, train_weights, test_weights, train_metVals, test_metVals = train_test_split(x, y, sampleWeights, metVals, random_state=30, test_size=0.2, train_size=0.8)
            train_x, val_x, train_y, val_y, train_weights, val_weights, train_metVals, val_metVals = train_test_split(train_x, train_y, train_weights, train_metVals, random_state=30, test_size=0.25, train_size=0.75)
        else:
            train_x, test_x, train_y, test_y, train_metVals, test_metVals = train_test_split(x, y, metVals, random_state=30, test_size=0.2, train_size=0.8)
            train_x, val_x, train_y, val_y, train_metVals, val_metVals = train_test_split(train_x, train_y, train_metVals, random_state=30, test_size=0.25, train_size=0.75)
        
        #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)   # could be used for early stopping of training
        logdir="./logs/2018/2D/"+modelName
        tensorboard_callback = TensorBoard(log_dir=logdir)      # setup tensorboard to log training progress
        earlystopping = EarlyStoppingCombined(patience=10, percentage=2, percentagePatience=10, generalizationPatience=10)

        
        # setup keras and train model
        my_model = KerasRegressor(build_fn=bJetRegression_Model, lr=lr, dout=dout, lamb=lamb, nLayer=nLayer, nodeFac=nodeFac, alph=alph, epochs=epochs, batch_size=batch_size, verbose=2)
        if genMETweighted:
            myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y,val_weights),sample_weight=train_weights,callbacks=[tensorboard_callback, earlystopping])
        else:
            myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y),callbacks=[tensorboard_callback])
        
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
        train_x, test_x, train_y, test_y, train_weights, test_weights = train_test_split(x, y, sample_weights, random_state=30, test_size=0.2, train_size=0.8)
        train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(train_x, train_y, train_weights, random_state=30, test_size=0.25, train_size=0.75)
    else:
        x,y,inputVars,genMET=getInputArray_allBins_nomDistr(dataPath,inputVars,targetName,target,updateInput,treeName,normalize,standardize,genMETweighted)
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
    
    
    
#############################################################
if __name__ == "__main__":
    # Define input data path
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v06/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"

    # Define Input Variables
    inputVars = ["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    #  ~trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True,genMETweighted=True)
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.097075752974773, 0.2899393936036301, -8.01970544548232, 8.154158507965985, 2.402508434459325, 3.5646202688746493, 0.11396439396039146
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.213461710731084, 0.1, -7.675330884712512, 6.6861398777274, 1.0, 5.0, 0.0
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.524903792932268, 0.5, -7.271889484384392, 7.1970043195604125, 8.0, 4.648503137689418, 0.0
    lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.21034037198, 0.35, -2.99573227355, 8.51719319142, 6.0, 3.0, 0.2 #standard
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.40951173838937, 0.4307905292558073, -6.478574338240861, 5.525847797310747, 7.743792064550437, 1.6496830299486087, 0.21534075102851583 
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -9.206217443761252, 0.3487143909602609, -7.971921315506171, 7.198234091870927, 3.197878169454659, 3.8910339597526784, 0.29973628543770203
    #  ~lr, dout, lamb, batch, nLayer, nodeFac, alph = -6.724648816372166, 0.42019722418191996, -1.9871840218385974, 7.288464822183116, 3.7077713293386814, 3.098255409797496, 0.10318236454640405
    
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    nodeFac2 = nodeFacs[int(np.round(nodeFac))]
    print(np.exp(lr), dout, np.exp(lamb), int(np.round(np.exp(batch))), int(np.round(nLayer)), nodeFac2, alph)
    
    trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"], np.exp(lr), dout, np.exp(lamb), int(np.round(np.exp(batch))), int(np.round(nLayer)), nodeFac2, alph,updateInput=True,genMETweighted=True)
    
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211130-1032genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True,genMETweighted=True,standardize=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211129-1202genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211122-1454genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"], np.exp(lr), dout, np.exp(lamb), int(np.round(np.exp(batch))), int(np.round(nLayer)), nodeFac2, alph,updateInput=True,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211125-1228genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)

