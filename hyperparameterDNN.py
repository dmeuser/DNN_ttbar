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
from bayes_opt import BayesianOptimization
import scipy.spatial.distance as ssd

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
    model.compile(loss="logcosh", optimizer=Adam(lr=lr),metrics=['mean_squared_error','mean_absolute_percentage_error',"logcosh"])
    return model

# function to get input from pkl if available, if update=true the pkl is constructed from the root file  
def getInputArray_allBins_nomDistr(path,inputVars,targetName,target,update,treeName,normalize=False,standardize=False,genMETweighted=False):
	
    appendCount = 0
    for subList in [target, ["PtNuNu", "PtNuNu_phi", "Lep1_phi", "Lep2_phi", "PuppiMET", "PuppiMET_phi"]]:      # append target variables to be also read in
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
        epochs = 50
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
            train_x, test_x, train_y, test_y, train_weights, test_weights, train_metVals, test_metVals = train_test_split(x, y, sampleWeights, metVals, random_state=28, test_size=0.2, train_size=0.8)
            train_x, val_x, train_y, val_y, train_weights, val_weights, train_metVals, val_metVals = train_test_split(train_x, train_y, train_weights, train_metVals, random_state=28, test_size=0.25, train_size=0.75)
        else:
            train_x, test_x, train_y, test_y, train_metVals, test_metVals = train_test_split(x, y, metVals, random_state=28, test_size=0.2, train_size=0.8)
            train_x, val_x, train_y, val_y, train_metVals, val_metVals = train_test_split(train_x, train_y, train_metVals, random_state=28, test_size=0.25, train_size=0.75)
        
        #  ~es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)   # could be used for early stopping of training
        logdir="./logs/2018/2D/"+modelName
        tensorboard_callback = TensorBoard(log_dir=logdir)      # setup tensorboard to log training progress
        
        #  ~print("###############################\n min and max of train weights:")
        #  ~print(min(train_weights), max(train_weights))
        
        # setup keras and train model
        my_model = KerasRegressor(build_fn=bJetRegression_Model, lr=lr, dout=dout, lamb=lamb, nLayer=nLayer, nodeFac=nodeFac, alph=alph, epochs=epochs, batch_size=batch_size, verbose=2)
        if genMETweighted:
            myHistory = my_model.fit(train_x,train_y,validation_data=(val_x,val_y,val_weights),sample_weight=train_weights,callbacks=[tensorboard_callback])
            #  ~my_model.fit(train_x,train_y,validation_data=(val_x,val_y,np.sqrt(val_weights)),sample_weight=np.sqrt(train_weights),callbacks=[tensorboard_callback])
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

        logcosh_val = myHistory.history["val_logcosh"][-1]

        
        #save Model
        #  ~my_model.model.save('trainedModel_Keras/2018/2D/'+modelName)
        #  ~my_model.model.save('trainedModel_Keras/2018/2D/'+modelName+".h5")
        
        #  ~y_hat_train = my_model.predict(train_x,use_multiprocessing=True)
        y_hat_val = my_model.predict(val_x,use_multiprocessing=True)
        

        
        val_x["DNN_1"]=[row[0] for row in y_hat_val]
        val_x["DNN_2"]=[row[1] for row in y_hat_val]
        val_x[target[0]]=val_y[target[0]]
        val_x[target[1]]=val_y[target[1]]
        val_x["DNN_MET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x["DNN_1"]
        val_x["DNN_MET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x["DNN_2"]
        val_x["DNN_MET"]=np.sqrt(val_x["DNN_MET_X"]**2+val_x["DNN_MET_Y"]**2)
        val_x["genMET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x[target[0]]
        val_x["genMET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x[target[1]]
        val_x["genMET"]=np.sqrt(val_x["genMET_X"]**2+val_x["genMET_Y"]**2)
        
        #  ~val_x["DNN_1"]=[row[0] for row in y_hat_val]
        #  ~val_x["DNN_2"]=[row[1] for row in y_hat_val]
        #  ~val_x[target[0]]=val_y[target[0]]
        #  ~val_x[target[1]]=val_y[target[1]]
        #  ~val_x["DNN_MET_X"]=val_x["PuppiMET*cos(PuppiMET_phi)"]-val_x["DNN_1"]
        #  ~val_x["DNN_MET_Y"]=val_x["PuppiMET*sin(PuppiMET_phi)"]-val_x["DNN_2"]
        #  ~val_metVals["DNN_MET"]=np.sqrt(val_x["DNN_MET_X"]**2+val_x["DNN_MET_Y"]**2)
        
        met_MSE_val=np.mean((val_x["DNN_MET"]-val_x["genMET"])**2)
        metVec_mean_val=np.mean((val_x["DNN_MET_X"]-val_x["genMET_X"])**2+(val_x["DNN_MET_Y"]-val_x["genMET_Y"])**2)
        
        min_x, max_x = 0., 400.
        countsDNN,bins = np.histogram(np.clip(val_x["DNN_MET"],min_x+0.01,max_x-0.01),bins=500,range=(min_x,max_x))
        countsGen,bins = np.histogram(np.clip(val_x["genMET"],min_x+0.01,max_x-0.01),bins=500,range=(min_x,max_x))
        countsDNN, countsGen = np.array(countsDNN), np.array(countsGen)
        print("\n\nshould be 0",sum(countsDNN)-sum(countsGen)) 
        jensenShannon_met_val = ssd.jensenshannon(countsDNN, countsGen)
        chisquare_met_val = sum((countsDNN-countsGen)**2/countsGen)/500
        
        #Purity and Stability
        val_metVals["DNN_MET"] = val_x["DNN_MET"]
        val_metVals["DNN_MET_X"] = val_x["DNN_MET_X"]
        val_metVals["DNN_MET_Y"] = val_x["DNN_MET_Y"]
        val_metVals["genMET"] = val_x["genMET"]
        temp_metVals,_ = val_metVals.groupby(np.where(val_metVals["PuppiMET"]>=0, 0, 1))
        val_metVals = temp_metVals[1]
        
        val_metVals["DNN_dPhiMetNearLep"] = np.array([np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep1_phi"])), np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(np.arctan2(val_x["DNN_MET_Y"],val_x["DNN_MET_X"])-val_metVals["Lep2_phi"]))]).min(axis=0)
        
        val_metVals["dPhi_PtNuNuNearLep"] = np.array([np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep1_phi"]), np.abs(2*np.pi-np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep1_phi"])),np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep2_phi"]), np.abs(2*np.pi-np.abs(val_metVals["PtNuNu_phi"]-val_metVals["Lep2_phi"]))]).min(axis=0)
        
        metBins = np.array([0,40,80,120,160,230,400])
        dphiBins = np.array([0,0.7,1.4,3.15])
        
        
        met_gen = np.clip(train_metVals["PtNuNu"], metBins[0], metBins[-1])
        met_reco = np.clip(train_metVals["DNN_MET"], metBins[0], metBins[-1])
        dphi_gen = train_metVals["dPhi_PtNuNuNearLep"]
        dphi_reco = train_metVals["DNN_dPhiMetNearLep"]
        
        print("loss (logcosh w/o regularisation term): {0:.5g}".format(logcosh_val))
        print("MSE met difference absolute: {0:.5g}".format(met_MSE_val))
        print("mean met difference vectorial: {0:.5g}".format(metVec_mean_val))
        print("Jensen-Shannon distance met: {0:.5g}".format(jensenShannon_met_val))
        print("chi square met distribution: {0:.5g}".format(chisquare_met_val))
        
        return [logcosh_val,met_MSE_val,metVec_mean_val,jensenShannon_met_val,chisquare_met_val]
        
    
def fit_test(x,y):
    #  ~x=y
    return -x ** 2 - (y - 1) ** 2 + 1, -x ** 2 - (y + 1) ** 2 + 1

def fit_opt(lr, dout, lamb, batch, nLayer, nodeFac, alph):
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v04/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"
    inputVars = ["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    val_MET_mean = trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"], lr, dout, lamb, batch, nLayer, nodeFac, alph,updateInput=True,genMETweighted=True)
    
    return val_MET_mean

def hyperOpt():
    #  ~pbounds = {'lr': (1e-5, 1e-3), 'dout':(0.1,0.5), 'lamb':(0.001,0.1)}
    pbounds = {'x': (2, 4), 'y': (-3, 3)}

    optimizer = BayesianOptimization(
        f=fit_test,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=30,
    )

    optimizer.maximize(init_points=3, n_iter=12,)



    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)

def GridSearch():
    #  ~lrVals = [0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05]    #0.0001 before
    #  ~dout = [0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]     #0.35 for grid, 0.3,0.4 before
    #  ~lambs = [0.,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]     #0.05 for grid, 0.05, ... before
    #  ~batches = [250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000]      #5000 before
    #  ~nLayers = [1, 2, 3, 4, 5, 6, 7, 8]      #6 before
    #  ~nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]       #1 before
    alph = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]       #0.2 before
    #  ~bestvals = [0.0001, 0.25, 0.01, 2000, 5, 2, 0.2]
    gridName="alphVals"
    results=[]
    #  ~results.append(["ValLoss", "MSE"])
    for a in alph:
        results.append(fit_opt(0.0001, 0.35, 0.05, 5000, 6, 1., a))
        print(a)
    results=np.array(results)
    print(results)
    np.savetxt("HyperParameterResuls/"+gridName+".csv", results, delimiter=",")

if __name__ == "__main__":
    # Define input data path
    dataPath="/net/data_cms1b/user/dmeuser/top_analysis/2018/v04/minTrees/100.0/Nominal/TTbar_amcatnlo_merged.root"

    # Define Input Variables
    inputVars = ["PuppiMET*cos(PuppiMET_phi)","PuppiMET*sin(PuppiMET_phi)","METunc_Puppi","MET*cos(PFMET_phi)","MET*sin(PFMET_phi)","HT*cos(HT_phi)","HT*sin(HT_phi)","nJets","n_Interactions","Lep1_flavor","Lep2_flavor","Lep1_pt*cos(Lep1_phi)","Lep1_pt*sin(Lep1_phi)","Lep1_eta","Lep1_E","Lep2_pt*cos(Lep2_phi)","Lep2_pt*sin(Lep2_phi)","Lep2_eta","Lep2_E","Jet1_pt*cos(Jet1_phi)","Jet1_pt*sin(Jet1_phi)","Jet1_eta","Jet1_E","Jet2_pt*cos(Jet2_phi)","Jet2_pt*sin(Jet2_phi)","Jet2_eta","Jet2_E","dPhiMETnearJet","dPhiMETfarJet","dPhiMETleadJet","dPhiMETlead2Jet","dPhiMETbJet","dPhiLep1Lep2","dPhiJet1Jet2","METsig","MHT","MT","looseLeptonVeto","dPhiMETnearJet_Puppi","dPhiMETfarJet_Puppi","dPhiMETleadJet_Puppi","dPhiMETlead2Jet_Puppi","dPhiMETbJet_Puppi","dPhiLep1bJet","dPhiLep1Jet1","mLL","CaloMET*cos(CaloMET_phi)","CaloMET*sin(CaloMET_phi)","MT2","vecsum_pT_allJet","vecsum_pT_l1l2_allJet","mass_l1l2_allJet","ratio_vecsumpTlep_vecsumpTjet","mjj"]
    
    #  ~trainKeras(dataPath,inputVars,"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=True,genMETweighted=False)
    
    #  ~hyperOpt()
    GridSearch()
    #  ~print(fit_opt(0.0001, 0.4, 0.05))
    #  ~fit_opt(0.0001,0.4,0.001)
    
    #  ~shapleyValues(dataPath,inputVars,"trainedModel_Keras/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20210519-1014normDistr","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"])
    
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20210519-1014normDistr","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1017genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1141genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)
    #  ~plot_Output(dataPath,inputVars,"trainedModel_Keras/2018/2D/Inlusive_amcatnlo_xyComponent_JetLepXY_50EP__diff_xy_2018_20211021-1509genMETweighted","TTbar_amcatnlo","diff_xy",["PuppiMET*cos(PuppiMET_phi)-genMET*cos(genMET_phi)","PuppiMET*sin(PuppiMET_phi)-genMET*sin(genMET_phi)"],updateInput=False,genMETweighted=True)

