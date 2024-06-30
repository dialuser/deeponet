#author: Alex Sun
#date: 2/1/2024
#adapted from https://github.com/cmoyacal/DeepONet-Grid-UQ/blob/master/src/training/supervisor.py
#this is seq2seq model used in the manuscript
#rev: 06262024, revised to add ensemble crps score for WRR revision
#===============================================================================================
import os,sys
import torch
import numpy as np
import modulus
from modulus.hydra import to_absolute_path, ModulusConfig
import pickle as pkl
import pandas as pd

from torch import distributions
from tqdm.auto import trange
from typing import Any
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset, DataLoader
import matplotlib.pyplot as plt
import hydrostats as HydroStats

from myutil import getEnsembleUSGSData, transformQ
from uq_deeponet import DPODataset, DeepONet, prob_DeepONet, probabilistic_train,test

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#@todo: move this to yaml file
SEQF  = 3
SEQB  = 80

def normalizeFeature(forcingDict, selectedStations, time_ind):
    """ Normalize forcing using nldas data
    Params
    ------
    forcingDict: dictionary of forcing features
    selectedStations: usgs station
    scalers should not be none for testing dataset

    Note: nldas data was generated using readcamels.py
    """
    #['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    #normalize forcing array
    #choose which variables to use
    vars = ['prcp', 'srad', 'tmax', 'tmin', 'vp']
    #load nldas scalers
    nldas_scaler = pkl.load(open(to_absolute_path(os.path.join('data', 'camels_forcing_scaler_nldas.pkl')), 'rb'))
    allStations = []
    for key in forcingDict.keys():
        if key in selectedStations:
            arr = []
            for avar in vars:
                arr.append(forcingDict[key][avar])
            allStations.append(np.stack(arr)[:,time_ind])
    allStations = np.stack(allStations, axis=0)
    input_mean = nldas_scaler['input_means']
    input_std = nldas_scaler['input_stds']
    #make the same size as data
    input_mean = np.repeat(input_mean.reshape(-1,1), repeats=allStations.shape[2], axis=1)
    input_std = np.repeat(input_std.reshape(-1,1), repeats=allStations.shape[2], axis=1)

    for i in range(allStations.shape[0]):
        allStations[i] = (allStations[i] - input_mean)/input_std

    print ('forcing max', np.max(allStations), 'forcing min', np.min(allStations))

    return allStations

def loadData(cfg, mode='train', basescaler=None, dqscaler=None):
    gageids = [
        '01435000', '01434498',   '01434176',   '01434105', '01434025', 
        '0143402265', '01434021',  '01434017', '01434013',  '0143400680'
    ]    
    gage_lon = [-74.5898056, -74.57438889, -74.5497222, -74.5295998,-74.5002222,
                -74.4808333, -74.4144444, -74.54027778, -74.50527778, -74.4480556
    ]
    gage_lat = [41.88994444, 41.92041667, 41.95555556, 41.9350922, 41.9960833,
                41.99027778, 42.0111111, 41.92527778, 41.9383333, 41.96694444
    ]
    gage_xy = np.c_[np.array(gage_lat), np.array(gage_lon)]
    
    #Load CAMELS static scaler
    static_scaler = pkl.load(open(to_absolute_path('data/static_scaler.pkl'), 'rb'))
    gage_xy = static_scaler.transform(gage_xy)
    
    #id_train for DeepONet training [not needed?]
    id_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #id_ga_train for GA training
    id_ga_train = cfg.custom.uqdpo.id_train # [4] # 6, 7, 9]
    id_ga_test =  cfg.custom.uqdpo.id_test # [4] # 1, 2, 3, 4, 5, 6, 7, 8, 9]

    startdate     = cfg.custom.start_date 
    enddate       = cfg.custom.end_date  
    cal_startDate = cfg.custom.cal_start_date  #'1993/10/01'
    cal_endDate   = cfg.custom.cal_end_date    #'2000/09/30'
    test_startDate= cfg.custom.test_start_date #'1991/10/01'
    test_endDate  = cfg.custom.test_end_date   #'1993/09/30'

    #use calibration/testing period
    #find index of calibration start/end
    tstart       = None
    tend         = None        
    t_test_start = None
    t_test_end   = None

    full_daterng = pd.date_range(startdate, enddate, freq='1D')

    for ix,item in enumerate(full_daterng):
        if item == pd.to_datetime(cal_startDate):
            tstart=ix
        elif item == pd.to_datetime(cal_endDate):
            tend = ix
        elif item == pd.to_datetime(test_startDate):
            t_test_start = ix
        elif item == pd.to_datetime(test_endDate):
            t_test_end = ix

    if (tstart is None) or (tend is None):
        raise Exception('calibration period not set right')

    if (t_test_start is None) or (t_test_end is None):
        raise Exception('test period not set right')

    #======assuming train dates are contiguous
    t_train_ind = list(range(tstart, tend+1))        
    t_test_ind  = list(range(t_test_start, t_test_end+1))
    t_train_ind = np.array(t_train_ind, dtype=int)
    t_test_ind  = np.array(t_test_ind, dtype=int)
    nTrain = len(t_train_ind)
    nTest = len(t_test_ind)

    #Load ensemble data [these are generated using readensemble.py in atsauto]
    _, _, allForcingDFDict, allUSGSData = pkl.load(open(to_absolute_path(cfg.custom.data_file), 'rb'))
    
    #Load best case result from GA (generated by running upperneversink_uq_multi.py)
    exp_no  = 3
    best_u_dict_train, best_u_dict_test = pkl.load(open(to_absolute_path(f"data/best_u_spacetime_{exp_no}.pkl"), 'rb'))

    #now separate train and testing data
    if mode == 'train':
        udict = best_u_dict_train
        t_ind = t_train_ind
        #make the data lengths match
        for akey in best_u_dict_train.keys():
            assert(len(best_u_dict_train[akey]) == nTrain)
    else:
        udict = best_u_dict_test
        t_ind = t_test_ind
        print ('len t_ind', len(t_ind))
        for akey in best_u_dict_test.keys():
            assert(len(best_u_dict_test[akey]) == nTest)
        
    #lookback and prediction steps
    seqB = cfg.custom.uqdpo.backward    
    seqF = cfg.custom.uqdpo.forward

    if mode == 'train':
        stationList =[gageids[id] for id in id_ga_train]
    else:
        stationList =[gageids[id] for id in id_ga_test] 

    #Load actual gage data
    daterng = pd.to_datetime(pd.date_range(start=startdate, end=enddate, freq='1D', tz='UTC'))

    allQ = []
    obsDict = {}
    maskDict = {}
    base_mu = None
    climatologyDict = {}
    for gage_id in stationList:
        obsdf   = getEnsembleUSGSData(allUSGSData, gageid=gage_id, startDate=startdate, endDate=enddate, returnDF=True)                
        #asun: 06262024, calculate daily climatology,
        climatologyDict[gage_id]=obsdf.groupby([obsdf.index.day,obsdf.index.month]).mean().values

        #this makes all DF the same length
        obsdf   = obsdf.reindex(daterng)
        
        #generate binary mask
        mask = np.ones(obsdf.shape[0])
        mask[obsdf['Q'].isna()] = 0

        if gage_id == '01435000':
            #data normalization
            base_mu  = np.nanmean(transformQ(obsdf.iloc[t_train_ind,:]))
            base_std = np.nanstd(transformQ(obsdf.iloc[t_train_ind,:]))
            basescaler = [base_mu, base_std]
            pkl.dump(basescaler, open(to_absolute_path('basescaler.pkl'), 'wb'))
        elif base_mu is None and not basescaler is None:
            base_mu, base_std = basescaler 
        else:
            basescaler = pkl.load(open(to_absolute_path('basescaler.pkl'), 'rb'))
            print ('basemu', base_mu)
            base_mu, base_std = basescaler
        
        if mode == 'test':
            obsDict[gage_id] = obsdf.iloc[t_ind, :]
            maskDict[gage_id] = mask[t_ind]

        allQ.append((transformQ(obsdf.values) - base_mu)/base_std)
    allQ = np.stack(allQ).squeeze(-1)[:, t_ind]

    #Assemble training/testing data
    #data structure [ngage, nBack]
    allSimuQ = []
    for akey in stationList:
        Q = (transformQ(udict[akey]) - base_mu)/base_std
        allSimuQ.append(Q)
    allSimuQ = np.stack(allSimuQ)

    if mode == 'train':
        allForcing = normalizeFeature(allForcingDFDict, stationList, t_ind)
    else:
        allForcing = normalizeFeature(allForcingDFDict, stationList, t_ind)

    print ('allQ', allQ.shape, 'allSimuQ', allSimuQ.shape, 'allForcing', allForcing.shape)
    print ('max values', np.max(allQ), np.max(allSimuQ), np.max(allForcing))
    print ('min values', np.min(allQ), np.min(allSimuQ), np.min(allForcing))

    #Calculate log-Q difference anyway to make my life easier
    DQ = []
    if mode=='train':        
        dqscaler = {}
        for ix, akey in enumerate(id_ga_train):
            ar = allQ[ix] - allSimuQ[ix]            
            minDQ,maxDQ = np.nanmin(ar), np.nanmax(ar)
            DQ.append(2.0*(ar-minDQ)/(maxDQ-minDQ) - 1.0)
            dqscaler[akey] = [minDQ, maxDQ]
    else:
        for ix,akey in enumerate(id_ga_test):
            minDQ,maxDQ = dqscaler[akey]            
            ar = allQ[ix] - allSimuQ[ix]                        
            DQ.append( 2.0*(ar-minDQ)/(maxDQ-minDQ) - 1.0)
        #need to pass id_ga_test for inverse transform
        dqscaler['id_test'] = id_ga_test
    DQ = np.stack(DQ, axis=0)

    #form input/output pairs
    learnDelta = cfg.custom.learn_delta
    if learnDelta:
        print ('***learning delta mode', np.nanmin(DQ),np.nanmax(DQ))

    X = []  #input daymet forcing + static_feature
    Y = []  #target
    T = []  #target time coord
    Z = []  #model simulation for model diff calculation 
    test_horizon = 0
    for id in range(len(stationList)):
        static_feature = np.repeat(gage_xy[id:id+1,:].transpose(), repeats=seqB, axis=1)
        for it in range(seqB, len(t_ind)-seqF):
            if not np.isnan(allQ[id, it:it+seqF]).any():
                #
                #here I assume allForcing and allSimuQ data are never nan
                #concatenate forcing data and simulated Q   
                #             
                datablock = np.r_[allForcing[id, :, it-seqB:it], 
                                  static_feature,
                                  allSimuQ[id:id+1,it-seqB:it],
                            ]
                #datablock = allSimuQ[id:id+1,it-seqB:it]
                for it_f in range(it,it+seqF):
                    if mode == 'test' and (it_f - it) == test_horizon: 
                        #[seqB, input_feature]
                        X.append(datablock.transpose())
                        #[seqF, output_feature]           
                        if learnDelta:
                            Y.append(DQ[id:id+1, it_f:it_f+1].transpose())
                        else:
                            Y.append(allQ[id:id+1, it_f:it_f+1])
                        T.append([(it_f-it)/seqF])
                        Z.append(allSimuQ[id:id+1,it_f:it_f+1])
                    elif mode == 'train':
                        X.append(datablock.transpose())
                        #
                        if learnDelta:
                            Y.append(DQ[id:id+1, it_f:it_f+1].transpose())
                        else:
                            Y.append(allQ[id:id+1, it_f:it_f+1].transpose())
                        T.append([(it_f-it)/seqF])
                        Z.append(allSimuQ[id:id+1,it_f:it_f+1])
    if mode == 'test':
        dataDict = {
            'obs' : obsDict,
            'simu': best_u_dict_test,
            'basescaler': basescaler,
            'dqscaler' : dqscaler,
            'time_axis': full_daterng,
            'test_ind': t_test_ind,
            'mask': maskDict,
            'climatology': climatologyDict
        }
        return X, Y, T, Z, dataDict 
    else:
        return X, Y, T, Z, basescaler, dqscaler

def genDataSets(cfg):
    X, Y, T, Z, basescaler, dqscaler = loadData(cfg, mode='train')
    dataset = DPODataset(X, Y, Tout=T, Z=Z, learnDelta=cfg.custom.learn_delta, mode='train')
    val_split=0.2
    train_idx, val_idx = train_test_split(list(range(len(X))), test_size=val_split)
    
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)

    print ('train data len', len(datasets['train']))
    in_dim = X[0].shape[-1]

    Xtest, Ytest, Ttest, Ztest, obsDict = loadData(cfg, mode='test',basescaler=basescaler, dqscaler=dqscaler)
    datasets['test'] = DPODataset(Xtest, Ytest, Tout = Ttest, Z=Ztest, learnDelta=cfg.custom.learn_delta, mode='test')
    print ('test data len', len(datasets['test']))

    return datasets, in_dim, obsDict

# l2 relative error
def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true) 

# l1 relative error
def l1_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_true - y_pred, ord=1) / np.linalg.norm(y_true, ord=1)

@modulus.main(config_path="conf", config_name="config_uns_uq_multi")
def uq_deeponet(cfg: ModulusConfig) -> None:
    import hydrostats.ens_metrics as em
    cfg.custom.uqdpo.id_train = [0]
    cfg.custom.uqdpo.id_test = [0]

    datasets, in_dim, dataDict = genDataSets(cfg)
    basescaler = dataDict['basescaler']
    dqscaler = dataDict['dqscaler']

    allSeeds = [20230101, 202301311, 2022023, 20211015, 202303311, 20240209, 20240211, 20240213,20240215, 20240217]
    #don't forget to change the retrain in ymal
    reTrain = cfg.custom.uqdpo.retrain
    seqB = cfg.custom.uqdpo.backward
    seqF = cfg.custom.uqdpo.forward

    if reTrain:
        allPred = []
        for seed in allSeeds:
            print ('*'*30, 'Training model using seed ', seed)
            #seed = int(cfg.custom.uqdpo.seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            #@todo: moving this to configuration file

            n_sensors = seqB         # cfg.custom.lstm.backward                  #@param {type: "integer"} # of sensors  
            n_basis =  100           #@param {type: "integer"} # of basis functions  
            branch_type = "modified" #@param ["modified", "MLP"]
            trunk_type = "modified"  #@param ["modified", "MLP"]
            width = 256              #@param {type: "integer"}
            depth = 3                #@param {type: "integer"}
            activation = "sin"       #@param ["leaky", "silu", "Rrelu", "Mish", "sin", "relu", "tanh", "selu", "gelu"]
            n_feature = 8

            #@markdown training parameters
            learning_rate = cfg.custom.uqdpo.learning_rate     #@param {type: "raw"}
            batch_size    = cfg.custom.uqdpo.batch_size         #@param {type: "integer"}
            n_epochs      = cfg.custom.uqdpo.n_epochs           #@param {type: "integer"}
            dim  = 1

            branch = {}
            branch["type"] = branch_type
            branch["layer_size"] = [n_feature*n_sensors] + [width]*depth + [n_basis]
            branch["activation"] = activation

            trunk = {}
            trunk["type"] = trunk_type
            trunk["layer_size"] = [dim] + [width] * depth + [n_basis]
            trunk["activation"] = activation

            model = prob_DeepONet(branch, trunk).to(device)

            ###################################
            # Step 7: define training parameters
            ###################################
            train_params = {}
            train_params["learning rate"] = learning_rate
            train_params["batch size"] = batch_size
            train_params["epochs"] = n_epochs
            train_params["print every"] = 10
            train_params["eval every"] = 1

            ###################################
            # Step 8: define scheduler parameters
            ###################################
            scheduler_params = {}
            scheduler_params["patience"] = 1000
            scheduler_params["factor"] = 0.8


            #need to save train and test id
            trainid_str = ''.join([str(i) for i in cfg.custom.uqdpo.id_train])
            testid_str = ''.join([str(i) for i in cfg.custom.uqdpo.id_test ])

            #save the model
            model_path = to_absolute_path(f'uq_models/best_uqdeeponet_seed{seed}_{trainid_str}_{testid_str}.pth')

            trained_model = probabilistic_train(
                model,
                datasets,
                train_params,
                scheduler_params,
                verbose = False,
                loss_history = [],
                model_path = model_path,
                metrics = [l1_relative_error,l2_relative_error],
                reTrain = True,
                scaler = basescaler,
                dqscaler=dqscaler,
                learn_delta=cfg.custom.learn_delta
            )
            
            testLoader   = DataLoader(datasets['test'], 
                                    batch_size=train_params["batch size"], 
                                    shuffle=False,
                                    drop_last =  False,
                                    pin_memory = True)

            if cfg.custom.learn_delta:
                pred, std, targets = test(trained_model, testLoader, basescaler, cfg.custom.learn_delta, dqscaler)
            else:
                pred, std, targets = test(trained_model, testLoader, basescaler, cfg.custom.learn_delta)

            print (np.max(pred), np.max(targets))
            allPred.append(pred)
        #
        allPred = np.stack(allPred,axis=0)
        if cfg.custom.learn_delta:
            pkl.dump(allPred, open(to_absolute_path('outputs/uq_main_res_delta.pkl'), 'wb'))
        else:
            pkl.dump(allPred, open(to_absolute_path('outputs/uq_main_res.pkl'), 'wb'))

    if cfg.custom.learn_delta:
        allPred = pkl.load(open(to_absolute_path('outputs/uq_main_res_delta.pkl'), 'rb'))
    else:
        allPred = pkl.load(open(to_absolute_path('outputs/uq_main_res.pkl'), 'rb'))

    predMin  = np.nanmin(allPred, axis=0).squeeze()
    predMax  = np.nanmax(allPred, axis=0).squeeze()
    predMean = np.nanmean(allPred, axis=0).squeeze()    
    predMedian = np.nanmedian(allPred, axis=0).squeeze()    
    simuDict = dataDict['simu']
    obsDict  = dataDict['obs']    

    t_axis = dataDict['time_axis'][dataDict['test_ind']][seqB:-seqF]
    
    #convert t_axis to day of the year
    print (t_axis)
    dayofyear_arr = pd.to_datetime(t_axis)
    dayofyear_arr = dayofyear_arr.dayofyear.values

    for key in obsDict.keys():
        fig, axes = plt.subplots(1,1, figsize=(8,6))
        mask = dataDict['mask'][key]
        #seqB:-seqF, remove the initial offset and the last offset
        obsval = obsDict[key].values[seqB:-seqF,0]
        axes.plot(t_axis, obsval, linestyle=':', color='tab:green', label='Obs', linewidth=1.5)
        simuval = simuDict[key][seqB:-seqF]
        axes.plot(t_axis, simuval, color='tab:blue', label='DeepONet-P', linewidth=1.0)

        axes.fill_between(t_axis, predMin, predMax, color='darkgray', label='DPO bound')
        axes.plot(t_axis, predMean, color='black', linewidth=1.7, label='DPO mean')

        axes.set_xlabel('Time')
        axes.set_ylabel('Q (m$^3$/s)')
        axes.set_ylim([0, 100])
        #we only label subplot (a)
        if not cfg.custom.learn_delta:
            plt.legend()

        if cfg.custom.learn_delta:
            plt.savefig(to_absolute_path(f'outputs/uqdpo{key}_delta.eps'))
        else:
            plt.savefig(to_absolute_path(f'outputs/uqdpo{key}.eps'))
        plt.close()
        
        print ('kge simu vs obs', HydroStats.kge_2012(simuval, obsval))                
        print ('kge encoder vs obs', HydroStats.kge_2012(predMean, obsval.squeeze()))
        allPred = np.transpose(allPred.squeeze(), axes=[1,0])
        crps_dictionary_dpo = em.ens_crps(obsval, allPred)
        #form climatology array
        Q_clim = np.zeros((len(dayofyear_arr), 1))
        for i,item in enumerate(dayofyear_arr):
            #item is 1 based, need to minus 1
            Q_clim[i] = dataDict['climatology'][key][item-1]
        crps_dictionary_clim = em.ens_crps(obsval, Q_clim)
        #ref: https://hess.copernicus.org/articles/21/4841/2017/hess-21-4841-2017.pdf
        print ('mean CRPS', crps_dictionary_dpo['crpsMean'], 'clim CRPS', crps_dictionary_clim['crpsMean'])
        #calculate relative CRPS (eqn 1 in the above ref) in percentage
        crpss = (crps_dictionary_clim['crpsMean']-crps_dictionary_dpo['crpsMean'])/crps_dictionary_clim['crpsMean']
        print ('CRPSS ', crpss*100)
if __name__ == "__main__":
    uq_deeponet()