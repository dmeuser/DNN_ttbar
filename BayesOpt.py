from ROOT import TMVA, TFile, TTree, TCut, TMath, TH1F
from subprocess import call
from os.path import isfile
import argparse
import sys
import struct
import datetime

import os
#  ~os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

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
#  ~gpus = tf.config.experimental.list_physical_devices('GPU')
#  ~tf.config.experimental.set_memory_growth(gpus[0], True)

from CustomEarlyStopping import EarlyStoppingCombined
from DNN_funcs import getInputArray_allBins_nomDistr

from BayesianOptimization.bayes_opt import BayesianOptimization
from BayesianOptimization.bayes_opt.logger import JSONLogger
from BayesianOptimization.bayes_opt.event import Events
from BayesianOptimization.bayes_opt.util import load_logs
import scipy.spatial.distance as ssd

def limitgpu(maxmem):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate a fraction of GPU memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

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
def trainKeras(year,dataPath,inputVars,name,treeName,targetName,target,lr,dout,lamb,batch,nLayer,nodeFac,alph,nInputs,updateInput=False,permuationImportance=False,normalize=False,standardize=False,genMETweighted=False,overSample=False,underSample=False,doSmogn=False):
        
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
    loss_val = myHistory.history["val_loss"][np.argmin(myHistory.history["val_loss"])]
    saveResolutionMean = True
    if saveResolutionMean:
        y_hat_test = my_model.predict(test_x,use_multiprocessing=True)
        
        test_x["SF"] = test_metVals["SF"]
        test_x["DNN_1"]=[row[0] for row in y_hat_test]
        test_x["DNN_2"]=[row[1] for row in y_hat_test]
        test_x[target[0]]=test_y[target[0]]
        test_x[target[1]]=test_y[target[1]]
        test_x["DNN_MET_X"]=test_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-test_x["DNN_1"]
        test_x["DNN_MET_Y"]=test_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-test_x["DNN_2"]
        test_x["DNN_MET"]=np.sqrt(test_x["DNN_MET_X"]**2+test_x["DNN_MET_Y"]**2)
        test_x["genMET_X"]=test_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]-test_x[target[0]]
        test_x["genMET_Y"]=test_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]-test_x[target[1]]
        test_x["genMET"]=np.sqrt(test_x["genMET_X"]**2+test_x["genMET_Y"]**2)
        
        test_x["PuppiMET_xy"]=np.sqrt(test_x["PuppiMET_xy*cos(PuppiMET_xy_phi)"]**2+test_x["PuppiMET_xy*sin(PuppiMET_xy_phi)"]**2)
        test_x["genMET-PuppiMET_xy"]=test_x["genMET"]-test_x["PuppiMET_xy"]
        test_x["genMET-DNN_MET"]=test_x["genMET"]-test_x["DNN_MET"]
        # Calculate mean and std while taking SF_weights into account
        tempMeanDNN = np.average(test_x["genMET-DNN_MET"], weights=test_x["SF"])
        tempStdDNN = np.sqrt(np.average((test_x["genMET-DNN_MET"]-tempMeanDNN)**2, weights=test_x["SF"]))
        with open("BayesOptLogs/2018/Run4/Fluct_test_std.txt", "a") as f:
            f.write("{}, {}, {}, {},        ,{},{},{},{},{},{},{}\n".format(loss_val, logcosh_val, tempMeanDNN, tempStdDNN, lr,dout,lamb,batch,nLayer,nodeFac,alph))
        return [loss_val, logcosh_val]
        
    with open("BayesOptLogs/2018/Run7/fluctComp_v2.txt", "a") as f:
        f.write("{}, {}, ,{},{},{},{},{},{},{}\n".format(loss_val, logcosh_val, lr,dout,lamb,batch,nLayer,nodeFac,alph))
    # return loss and logcosh to choose from as target
    return [loss_val, logcosh_val]
    

#  ~def fit_opt(lr, dout, batch, nLayer, nodeFac, alph):
def fit_opt(lr, dout, batch, lamb, nLayer, nodeFac, alph):
    # get year, config
    global optimizer_config
    year = optimizer_config["year"]
    version = optimizer_config["version"]
    print(year, version)
    
    # define data path, bring network structure in right shape
    dataPath="/eos/user/p/phnattla/ttbar_inputs/"+year+"/"+version+"/TTbar_amcatnlo_merged.root"
    inputVars = ["PuppiMET_xy*sin(PuppiMET_xy_phi)", "PuppiMET_xy*cos(PuppiMET_xy_phi)", "MET_xy*sin(MET_xy_phi)", "MET_xy*cos(MET_xy_phi)", "vecsum_pT_allJet*sin(HT_phi)", "vecsum_pT_allJet*cos(HT_phi)", "mass_l1l2_allJet", "Jet1_pt*sin(Jet1_phi)", "MHT", "Lep1_pt*cos(Lep1_phi)", "Lep1_pt*sin(Lep1_phi)", "Jet1_pt*cos(Jet1_phi)", "mjj", "Jet1_E", "HT", "Jet2_pt*sin(Jet2_phi)", "Jet2_pt*cos(Jet2_phi)"]
    nInputs = int(len(inputVars))
    nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]
    nodeFac2 = nodeFacs[int(np.round(nodeFac))]
    
    print("lr: ", np.exp(lr),"dout: ", dout, "lamb: ", np.exp(lamb), "nbatch: ",  int(np.round(np.exp(batch))), "nlayer: ", int(np.round(nLayer)), "nodes: ", nodeFac2, "alpha: ", alph)
    
    # train network with corresponding settings, returning validatino loss and other quantities
    val_MET_mean = trainKeras(year,dataPath,inputVars,"Inlusive_amcatnlo_xyComponent_JetLepXY_50EP","TTbar_amcatnlo","diff_xy",["PuppiMET_xy*cos(PuppiMET_xy_phi)-genMET*cos(genMET_phi)","PuppiMET_xy*sin(PuppiMET_xy_phi)-genMET*sin(genMET_phi)"],np.exp(lr),dout,np.exp(lamb),int(np.round(np.exp(batch))),int(np.round(nLayer)),nodeFac2,alph,nInputs,updateInput=True,genMETweighted=True,standardize=False,overSample=False,underSample=False,doSmogn=False)
    
    # return target to maximize
    return -val_MET_mean[1]

def hyperOpt(netStruct, n_init=0, n_iter=1, kappa=2.576):
    pbounds = {'lr': (np.log(0.00001), np.log(0.05)), 'dout': (0.1, 0.5), 'lamb': (np.log(0.0001), np.log(0.2)), 'batch': (np.log(250), np.log(10000)), 'nLayer': (1, 8), 'nodeFac': (0,5), 'alph': (0, 0.4), }
    print(netStruct)
    optimizer = BayesianOptimization(
        f=fit_opt,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        #  ~random_state=3,
    )
    

    logfile = "BayesOptLogs/"+year+"/Run7/"
    filename = "logBayes_fluctComp_v2.json"
    if not os.path.exists(logfile):
        os.mkdir(logfile)
    with open(logfile+filename, "a") as f:
        print("new logfile created")
    #  ~load_logs(optimizer, logs=[logfile+filename]);
    logger = JSONLogger(path=logfile+filename, reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    if netStruct: 
        print("probing")
        optimizer.probe(
            params=netStruct,
            lazy=False,
        )
        optimizer.maximize(init_points=0, n_iter=0)
    else:
        optimizer.maximize(init_points=n_init, n_iter=n_iter, kappa=kappa)
    
    
    for i, res in enumerate(optimizer.res[-1:]):
        print("Iteration {}: {}".format(len(optimizer.res)-1+i, res))


def GridSearch():
    #  ~lrVals = [0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05]    #0.0001 before
    #  ~dout = [0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]     #0.35 for grid, 0.3,0.4 before
    #  ~lambs = [0.,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]     #0.05 for grid, 0.05, ... before
    #  ~batches = [250, 500, 750, 1000, 2000, 5000, 7500, 10000, 25000, 50000]      #5000 before
    #  ~nLayers = [1, 2, 3, 4, 5, 6, 7, 8]      #6 before
    #  ~nodeFacs = [1./8, 1./4, 1./2, 1., 2., 4.]       #1 before
    #  ~alph = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]       #0.2 before
    #  ~standardvals = [0.0001, 0.35, 0.05, 5000, 6, 1, 0.2]
    #  ~bestvals = [0.0001, 0.25, 0.01, 2000, 5, 2, 0.2]
    gridName="batchVals"
    results=[]
    #  ~results.append(["ValLoss", "MSE"])
    for val in batches:
        res = fit_opt(0.0001, 0.25, 0.01, val, 5, 4, 0.2)
        results.append(res)
        print("\n\n################################################\n"+gridName + ": " + str(val)+"\n################################################\n")
        with open("HyperParameterResuls/"+gridName+"_opt1_backup.csv", "ab") as f:
            np.savetxt(f, np.array([np.zeros(len(res)), res]), delimiter=",")
    results=np.array(results)
    print(results)
    np.savetxt("HyperParameterResuls/"+gridName+"_opt1.csv", results, delimiter=",")
    
    

########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, help="Year, choose between 2016_preVFP, 2016_postVFP, 2017, 2018 (default)")
    parser.add_argument('--version', type=str, help="treeVersion, such as v07 (default)")
    parser.add_argument('--probe', type=str, help="line string of dictionary of network structure to be probed, needing alph, batch, dout, lamb, lr, nLayer and nodeFac as inputs")
    # For example '--probe """{"alph": 0.11556803605322355, "batch": 6.501144102888323, "dout": 0.35075846582000303, "lamb": -5.941028038499022, "lr": -7.729770703881016, "nLayer": 2.2186773553565198, "nodeFac": 4.424425111826699}"""'
    parser.add_argument('--n_init', type=int, help="number of initial guesses in the bayesian optimization")
    parser.add_argument('--n_iter', type=int, help="number of optimization steps in the bayesian optimization")
    parser.add_argument('--kappa', type=float, help="dictionary of network structure to be probed, needing alph, batch, dout, lamb, lr, nLayer and nodeFac as inputs")
    
    args = parser.parse_args()
    
    
    year = "2018"
    version = "v07"
    netStruct = {}
    n_init = 0
    n_iter = 1
    kappa = 2.576
    
    if args.year: year = args.year
    if args.version: version = args.version
    if args.n_init: n_init = args.n_init
    if args.n_iter: n_iter = args.n_iter
    if args.kappa: kappa = args.kappa
    if args.probe: 
        print(args.probe)
        for pair in args.probe[1:-1].split(","):
            pair = pair.split(":")
            netStruct[pair[0]] = float(pair[1])
    
    optimizer_config = {"year": year, "version": version}

    hyperOpt(netStruct, n_init=n_init, n_iter=n_iter, kappa=kappa)
