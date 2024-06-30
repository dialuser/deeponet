#author: alex sun
#==========This is for multi-site illustration=====================================
#rev date: 09282023, use this for exasheds final allhands meeting
#Note: need to install hydrostats and deap every time
#rev date: 11072023, update the code for new ensemble
#rev date: 11092023, fixed a bug in subsetting the ensemble output data (see getQarray())
#rev date: 11102023, split time calibration/testing
#Note: to detach container w/o stopping it, ctrl+P, ctrl+Q
#      to retach 
#rev date: 01212024, add UQ
#rev date: removed gage 01434092 permanently
#rev date: 2/7/2024, add barplot
#rev date: this is the final version used for generating deeponet paper results
#rev date: 06252024 modified for WRR revision
#
#=========================================================================================
import torch
import numpy as np

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig, to_yaml
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.fourier_net import FourierNetArch
from modulus.models.deeponet import DeepONetArch
from modulus.domain.constraint.continuous import DeepONetConstraint
from modulus.domain.validator.discrete import GridValidator
from modulus.dataset.discrete import DictGridDataset
from modulus.domain.inferencer import PointwiseInferencer
from modulus.graph import Graph
from modulus.domain.constraint import Constraint
from modulus.key import Key

import pickle as pkl
import random
import pandas as pd
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydrostats as HydroStats
import time
import sys
from myutil import LpLoss, MyGridValidator, getEnsembleUSGSData,transformQ, inverseTransformQ,printParams, setLossFun

def getData(cfg, testid=None, mode='train'):
    """ Data preparation for DeepONet
    """
    from sklearn.model_selection import KFold

    #this random seed determines the ats ensemble train/test split 
    random.seed(cfg.custom.ats_seed)

    def getQarray(gageidset):
        #get flow rates into Qarr, dimensions N_gage x N_ens x N_times
        Qarr = None
        for ix, gage_id in enumerate(gageidset):
            print (gage_id)
            Qsingle = []
            for rz in rzList:
                #get Q
                stationDF = allDFDict[gage_id]
                #11/09/2023 bug fix, added ensemble output subsetting!
                stationDF = stationDF[(stationDF.index >= pd.to_datetime(startdate)) & (stationDF.index <= pd.to_datetime(enddate)) ]
                #01/23/2024 drop bad values
                stationDF = stationDF.dropna()
                Q = stationDF[f'ens.{rz}'].to_numpy()                                            
                Qsingle.append(transformQ(Q/86400, imethod=1))     #convert to m3/s
            
            Qsingle= np.stack(Qsingle)
            if Qarr is None:
                #dimensions: nGages, nEns, nTimes
                Qarr = np.zeros((len(gageidset), Qsingle.shape[0], Qsingle.shape[1]))
            Qarr[ix, :, :] = Qsingle

        return Qarr

    gageids = [
        '01435000', '01434498',   '01434176',   '01434105', '01434025', 
        '0143402265', '01434021',  '01434017', '01434013',  '0143400680'
    ]
    """
    #number of records at each site
    #gageid, number of records
    0 01435000 10258
    1 01434498 9955
    2 01434176 1115
    3 01434105 1825
    4 01434025 10258
    5 0143402265 1176
    6 01434021 8126
    7 01434017 9955
    8 01434013 1175
    9 0143400680 8218
    """    
    #split into train/test
    #select gages for training
    #01232024, don't use 01434092, the ensemble runs are problematic!!!!
    id_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    expno= 1

    if expno==1:
        id_ga_train = [0, 1, 4, 6, 7, 9]
        id_ga_test =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        cal_startDate = '1993/10/01'
        cal_endDate   = '2000/09/30'
        test_startDate= '1991/10/01'
        test_endDate  = '1993/09/30'
    
    elif expno==2:
        #exp1 [base case]        
        #for training deeponet
        #for training GA
        id_ga_train = [0, 1, 4, 6]        
        #for testing GA
        id_ga_test =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        cal_startDate = '1993/10/01'
        cal_endDate   = '2000/09/30'
        test_startDate= '1991/10/01'
        test_endDate  = '1993/09/30'

    elif expno==3:
        id_ga_train = [0, 7, 9]
        id_ga_test =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        cal_startDate = '1993/10/01'
        cal_endDate   = '2000/09/30'
        test_startDate= '1991/10/01'
        test_endDate  = '1993/09/30'

    elif expno==4:
        id_ga_train = [0, 1]
        id_ga_test =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        cal_startDate = '1993/10/01'
        cal_endDate   = '2000/09/30'
        test_startDate= '1991/10/01'
        test_endDate  = '1993/09/30'

    elif expno==5:
        id_ga_train = [0]
        id_ga_test =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        cal_startDate = '1993/10/01'
        cal_endDate   = '2000/09/30'
        test_startDate= '1991/10/01'
        test_endDate  = '1993/09/30'
    #
    #01222024, weights for GA loss function
    #                    
    all_gage_weights = [10, 5, 1, 1, 10, 1, 5, 5, 1, 5]    
    gage_weights = [all_gage_weights[id] for id in id_train]

    #ensemble startdate '1990-10-3' and enddate = '2000-10-01'
    startdate = cfg.custom.start_date 
    enddate   = cfg.custom.end_date   

    #11102023, add calibration/test period definitions
    pred_startDate= cfg.custom.pred_start_date
    pred_endDate  = cfg.custom.pred_end_date

    #These are taken from D:\Princeton_HiRes\neversink\neversinkgageloc_utm18.shp 
    #https://www.latlong.net/lat-long-utm.html
    #removed 
    gage_x = [534058.964434, 535267.218737, 537748.640447, 539611.662819, 541321.973647,
              542994.032181,548468.171454, 538144.921077, 541032.131468, 545722.512904] 

    gage_y = [4637647.10367, 4641018.68265, 4645171.17167, 4647365.05997,4649355.43552, 
              4648846.28256, 4651189.53034,4641636.11094, 4643041.40612,4646253.69601]

    #get distances relative to the most downstream gage
    gage_x = np.array(gage_x) - gage_x[0]
    gage_y = np.array(gage_y) - gage_y[0]
    
    #Location normalization [0, 1]
    gage_x = gage_x/np.max(gage_x)
    gage_y = gage_y/np.max(gage_y)

    #Load ensemble data [these are generated using readensemble.py in atsauto]
    paramDF, allDFDict, allForcingDFDict, allUSGSData = pkl.load(open(to_absolute_path(cfg.custom.data_file), 'rb'))

    rzList = paramDF['realization_id'].values.tolist()
    nRZ = len(rzList)

    if mode == 'train':
        #gages for deeponet training
        gage_ids_train = [gageids[i] for i in id_train]     
        gage_x_train = gage_x[id_train]
        gage_y_train = gage_y[id_train]
        nGages_train = len(gage_ids_train)

        gage_ids_test = gage_ids_train        
        nGages_test  = nGages_train
        gage_x_test = gage_x_train
        gage_y_test = gage_y_train
    else:    
        #gages for ga training
        gage_ids_train = [gageids[i] for i in id_ga_train]
        nGages_train = len(gage_ids_train)    
        gage_x_train = gage_x[id_ga_train]
        gage_y_train = gage_y[id_ga_train]

        #gages for ga testing
        gage_ids_test = [gageids[i] for i in id_ga_test]
        nGages_test = len(gage_ids_test)    
        gage_x_test = gage_x[id_ga_test]
        gage_y_test = gage_y[id_ga_test]

    #theta shape is (596, 13), 13 parameters, 596 realizations
    theta =np.stack([paramDF[paramDF['realization_id']==rz].iloc[:, 1:].to_numpy().squeeze() for rz in rzList])        

    #Split train/test for parameter domain
    #asun: 06252024, replace KF with realization number experiment
    # kf = KFold(n_splits=5, random_state=cfg.custom.ats_seed, shuffle=True)
    # for ifold, (train_indices, test_indices) in enumerate(kf.split(range(nRZ))):
    #     if ifold == cfg.custom.fold_no:
    #         train_indices = sorted(train_indices)
    #         test_indices  = np.array(list(sorted(set(range(nRZ)).difference(train_indices))))
    #         print (len(train_indices)+len(test_indices))
    #         break
    train_indices = np.array(sorted(np.random.choice(range(nRZ), int(0.8*nRZ), replace=False))) #account for 1-based index
    test_indices  = np.array(list(sorted(set(range(nRZ)).difference(train_indices))))

    #nTrain = 460
    #the last exp_no is the original setup    
    train_exps = [100, 200, 300, 400, len(train_indices)]
    train_split_no = cfg.custom.train_split_no

    train_indices = train_indices[:train_exps[train_split_no]]


    #get one realization for testing
    if not testid is None: 
        test_indices = [test_indices[testid]]

    if mode == 'train':
        #Form train/test for branch domain
        Qarr = getQarray(gage_ids_train)
        #separate train/test arrays for branchnet
        Q_train = Qarr[:, train_indices,:]
        Q_test  = Qarr[:, test_indices, :]         

    else:
        #ensemble for training GA 
        Qarr = getQarray(gage_ids_train)
        Q_train = Qarr[:, train_indices, :]

        #ensemble for testing GA on realizations not used in either surrogate modeling or GA calibration
        #                                     
        Qarr = getQarray(gage_ids_test)
        Q_test = Qarr[:, test_indices, :]

    _, _, nT = Qarr.shape

    #0210, get the median of predictions
    Q_pretrain = np.nanmedian(Qarr, axis=1)

    x_coord = np.arange(nT)/nT
    del Qarr

    timeSampling = 'all'
    if timeSampling=='random':    
        #randomly sample coordinates
        t_train_ind = sorted(np.random.choice(range(nT), 2000, replace=False))
        remain_ind = sorted(set(range(nT)).difference(t_train_ind))
        t_test_ind = sorted(np.random.choice(remain_ind, 500, replace=False))
    elif timeSampling == 'all':
        #split according to time>
        #original ATS simulation time: from 1990/10/3 to 2000/10/01
        full_daterng = pd.date_range(startdate, enddate, freq='1D')

        #use calibration/testing period
        #find index of calibration start/end
        tstart = None
        tend = None        
        t_test_start = None
        t_test_end = None

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

        #use a subset to train surrgate model
        t_obs_train_ind = list(range(tstart, tend))        
        #======assuming train dates are contiguous
        if mode == 'train':
            #if training, we only split realizations, but keep the time the same between train/test
            #train deeponet on the full time range because it does not extrapolate
            t_train_ind = list(range(nT))
            t_test_ind  = t_train_ind            
        else:            
            #if calibration, we split realizations and split the times             
            t_train_ind = list(range(tstart, tend+1))        
            t_test_ind = list(range(t_test_start, t_test_end+1))
            #t_test_ind  = list(set(range(nT)).difference(t_train_ind))

        t_train_ind = np.array(t_train_ind, dtype=int)
        t_test_ind  = np.array(t_test_ind, dtype=int)
        t_obs_train_ind = np.array(t_obs_train_ind, dtype=int)

    #note: for calibration, nT_train is different from nT_test
    nT_train = len(t_train_ind)
    nT_test  = len(t_test_ind)

    #LOAD OBS DATA
    #Note: this only matters to GA model calibration
    obsTrainDict = {}
    obsTestDict  = {}

    mu = None
    #must include the downstream gage so it can be used for normalization?
    assert ('01435000' in gage_ids_train)
    daterng = pd.to_datetime(pd.date_range(start=startdate, end=enddate, freq='1D', tz='UTC'))
    for gage_id in gageids:
        #obsdf   = loadUSGSObs(gageid=gage_id, startDate=startdate, endDate=enddate, returnDF=True)
        obsdf   = getEnsembleUSGSData(allUSGSData, gageid=gage_id, startDate=startdate, endDate=enddate, returnDF=True)        
        #this makes all DF the same length
        obsdf   = obsdf.reindex(daterng)
        #generate binary mask
        mask = np.ones(obsdf.shape[0])
        mask[obsdf['Q'].isna()] = 0
        obsdata = transformQ(obsdf.values, imethod=1)
        if gage_id == '01435000':
            #data normalization
            mu  = np.nanmean(obsdata[t_obs_train_ind])
            std = np.nanstd(obsdata[t_obs_train_ind])
        obsdata = (obsdata - mu)/std        
        obsdf = pd.DataFrame(obsdata, index=obsdf.index)
        if gage_id in gage_ids_train:
            obsTrainDict[gage_id]   = {'Q':obsdf.values[t_train_ind], 'mask':mask[t_train_ind]}
        if gage_id in gage_ids_test:
            obsTestDict[gage_id]    = {'Q':obsdf.values[t_test_ind],  'mask':mask[t_test_ind]}

    #normalize Q using '01435000' obs statistics
    Q_train= (Q_train - mu)/std
    Q_test = (Q_test  - mu)/std

    #for training/testing surrogate models (t_train_ind == t_test_ind)
    #[ngage, nEns, nT]
    Q_train = Q_train[:, :, t_train_ind]
    Q_test  = Q_test[:,:, t_test_ind]
    Q_pretrain = Q_pretrain[:, t_test_ind]
    #[ngages*nTrain,1]
    Q0 = Q_train[:,0,:].reshape(-1,1)

    Q_train = rearrange(Q_train, 'a b c ->(a b c) 1')
    Q_test  = rearrange(Q_test,  'a b c ->(a b c) 1')

    theta_train = theta[train_indices,:]
    theta_test  = theta[test_indices,:] 

    #Normalize parameters theta
    theta_scaling = cfg.custom.theta_scaling
    
    theta_param1=None
    theta_param2=None
    if theta_scaling in ['gaussian', 'normal']:    
        theta_param1 = np.mean(theta_train, axis=0)
        theta_param2 = np.mean(theta_train, axis=0)
        theta_train = (theta_train - theta_param1)/theta_param2
        theta_test = (theta_test - theta_param1)/theta_param2
    elif theta_scaling == 'minmax':
        # to range[-1,1]
        theta_param1 = np.min(theta_train, axis=0)
        theta_param2 = np.max(theta_train, axis=0)
        theta_train = (theta_train - theta_param1)/(theta_param2-theta_param1)*2 -1.0
        theta_test =  (theta_test - theta_param1)/(theta_param2-theta_param1)*2 -1.0
    else: 
        raise NotImplementedError('scaling method not implemented')

    print ('Theta min/max', np.min(theta_test, axis=0), np.max(theta_train,axis=0))

    theta_0     = np.mean(theta_train, axis=0).reshape(1, 1, -1) #initial guess?
    theta_train = np.expand_dims(theta_train, axis=0)
    theta_test_oo  = np.expand_dims(theta_test, axis=0)

    #Repeat theta to match row size of Q
    #the last dimension is number of parameters
    theta_train = repeat(theta_train, 'a b c -> (a rep1) (b rep2) c', rep1=nGages_train, rep2=nT_train)        
    theta_test  = repeat(theta_test_oo, 'a b c ->  (a rep1) (b rep2) c', rep1=nGages_test, rep2=nT_test)
    theta_0     = repeat(theta_0, 'a b c -> (a rep1) (b rep2) c',     rep1=nGages_train, rep2=nT_train)


    theta_train = rearrange(theta_train, 'a b c -> (a b) c')        
    theta_test  = rearrange(theta_test, 'a b c -> (a b) c')
    theta_0     = rearrange(theta_0, 'a b c -> (a b) c')

    #Repeat time coordinates
    #[nGages, nEns, nT]
    t_train0 =  np.reshape(x_coord[t_train_ind], (1, 1, nT_train)) #keep the original train vector
    t_test_oo=  np.reshape(x_coord[t_test_ind],  (1, 1, nT_test))    

    t_train = repeat(t_train0, 'a b c -> (a rep1) (b rep2) c',  rep1=nGages_train, rep2=len(train_indices))
    t_test  = repeat(t_test_oo, 'a b c ->  (a rep1) (b rep2) c',   rep1=nGages_test, rep2=len(test_indices))    
    t_train = rearrange(t_train, 'a b c -> (a b c) 1')
    t_test  = rearrange(t_test, 'a b c -> (a b c) 1')

    #Repeat x, y coordinates
    gagex_train = np.reshape(gage_x_train, (-1,1,1))
    gagey_train = np.reshape(gage_y_train, (-1,1,1))
    gagex_test = np.reshape(gage_x_test, (-1,1,1))
    gagey_test = np.reshape(gage_y_test, (-1,1,1))

    x_train = repeat(gagex_train, 'a b c -> a (b rep1) (c rep2)', rep1 =len(train_indices), rep2=nT_train)
    x_test = repeat(gagex_train, 'a b c  -> a (b rep1) (c rep2)', rep1 =len(test_indices), rep2=nT_test)
    y_train = repeat(gagey_train, 'a b c -> a (b rep1) (c rep2)', rep1 =len(train_indices), rep2=nT_train)
    y_test = repeat(gagey_train, 'a b c  -> a (b rep1) (c rep2)', rep1 =len(test_indices), rep2=nT_test)      
    x_train = rearrange(x_train, 'a b c -> (a b c) 1')
    x_test = rearrange(x_test, 'a b c -> (a b c) 1')
    y_train = rearrange(y_train, 'a b c -> (a b c) 1')
    y_test = rearrange(y_test, 'a b c -> (a b c) 1')

    t0 = repeat(t_train0, 'a b c -> (a rep1) b c',  rep1=nGages_train)
    x0 = repeat(gagex_train, 'a b c -> a b (c rep2)', rep2=nT_train)
    y0 = repeat(gagey_train, 'a b c -> a b (c rep2)', rep2=nT_train)

    t0 = rearrange(t0, 'a b c ->(a b c) 1')
    x0 = rearrange(x0, 'a b c ->(a b c) 1')
    y0 = rearrange(y0, 'a b c ->(a b c) 1')

    #for new sites
    tnew = repeat(t_test_oo,  'a b c -> (a rep1) b c',  rep1=nGages_test)
    xnew = repeat(gagex_test, 'a b c -> a b (c rep2)', rep2=nT_test)
    ynew = repeat(gagey_test, 'a b c -> a b (c rep2)', rep2=nT_test)
    tnew = rearrange(tnew,    'a b c ->(a b c) 1')
    xnew = rearrange(xnew,    'a b c ->(a b c) 1')
    ynew = rearrange(ynew,    'a b c ->(a b c) 1')

    print ('train x_train', x_train.shape, y_train.shape, t_train.shape, 'theta_train', theta_train.shape, 'u train', Q_train.shape)
    print ('test x_test', x_test.shape, y_test.shape, t_test.shape, 'theta_test', theta_test.shape, 'u test', Q_test.shape, 'train indices', len(t_train_ind))
    print ('a0 ', theta_0.shape, 't_0', t0.shape, 'x0', x0.shape, 'y0', y0.shape, 'u0', Q0.shape)
    print ('xnew', xnew.shape, 'ynew', ynew.shape, 'tnew', tnew.shape, nT_train, nT_test)

    data={
        't_train': t_train, 
        't_test': t_test,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'a_train': theta_train,
        'a_test' : theta_test,
        'u_train': Q_train,
        'u_test' : Q_test,
        'scaler' : (mu, std), 
        'time_axis': full_daterng,
        'train_ind': t_train_ind,
        'test_ind' : t_test_ind,
        'gage_id' : gage_id,
        'n_test' : len(test_indices),
        'n_gages': nGages_train,
        'n_gages_test': nGages_test,
        'nt_train': nT_train,
        'nt_test': nT_test,
        'obs_dict_train': obsTrainDict,
        'obs_dict_test': obsTestDict,
        'gage_weights': gage_weights,
        'a_0': theta_0,
        't_0': t0,
        'x_0': x0,
        'y_0': y0,
        'u_0': Q0,
        'id_train':id_train,
        'tnew': tnew,
        'xnew': xnew,
        'ynew': ynew,
        'theta_scaler': (theta_param1, theta_param2),
        'expno':expno,
        'gageids': gageids,
        'Q_pretrain': Q_pretrain,
        'fold_no': cfg.custom.fold_no,
    }
    return data

def genModel(cfg: ModulusConfig) -> DeepONetArch:
    # [init-model]
    trunk_net = instantiate_arch(
        cfg=cfg.arch.trunk,
        input_keys=[Key("t"),Key("x"),Key("y")],
        output_keys=[Key("trunk", cfg.arch.trunk.layer_size)],
    )
    # set any normalization value that you want to apply to the data. 
    # For this dataset, calculate the scale and shift parameters 
    # it's easier to do normalization outside modulus !!!
    branch_net = instantiate_arch(
        cfg=cfg.arch.branch,
        input_keys=[Key("a", 13)],
        output_keys=[Key("branch", cfg.arch.branch.layer_size)],
    )

    deeponet = instantiate_arch(
        cfg=cfg.arch.deeponet,
        output_keys=[Key("u")],
        branch_net=branch_net,
        trunk_net=trunk_net,
    )
    total_params = sum(p.numel() for p in deeponet.parameters())
    print(f"Number of parameters: {total_params}")

    return deeponet

def getNetwork_dir(trainGageList,fold_no):
    #set network_dir using train_id
    exp_str = ""
    for id in trainGageList:
        exp_str += f"_{id}"
    exp_str += f"fold_{fold_no}"        
    return exp_str

@modulus.main(config_path="conf", config_name="config_uns_uq_multi_rev")
def run(cfg: ModulusConfig) -> None:
    reTrain  = cfg.custom.retrain           
    dataDict = getData(cfg, mode='train')
  
    cfg.network_dir = getNetwork_dir(dataDict['id_train'], dataDict['fold_no'])
    cfg.network_dir += f'exp_{cfg.custom.train_split_no}'
    print ('Network dir is', cfg.network_dir)

    deeponet = genModel(cfg)
    nodes = [deeponet.make_node('deepo')]

    # [init-model]

    # [datasets]
    x_train = dataDict["x_train"]
    t_train = dataDict["t_train"]
    y_train = dataDict["y_train"]
    a_train = dataDict["a_train"]
    u_train = dataDict["u_train"]

    # load test dataset 1 [testing at training gage sites for test period]
    x_test = dataDict["x_test"]
    t_test = dataDict["t_test"]
    y_test = dataDict["y_test"]
    a_test = dataDict["a_test"]
    u_test = dataDict["u_test"]

    # [datasets]
    # [constraint]
    # make domain
    domain = Domain()
    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"a": a_train, "x": x_train, "t": t_train, "y":y_train},
        outvar={"u": u_train},
        batch_size=cfg.batch_size.train,
    )

    domain.add_constraint(data, "data")
    # [constraint]

    # [validator]
    # add validators
    # sites used during training
    invar_valid = {
        "a": a_test,
        "x": x_test,
        "t": t_test,
        "y": y_test
    }
    print ('x_test.shape', x_test.shape, 'a_test', a_test.shape, 'u_test', u_test.shape)

    outvar_valid = {"u": u_test}
    dataset = DictGridDataset(invar_valid, outvar_valid)

    validator = MyGridValidator(nodes=nodes, dataset=dataset, plotter=None)
    domain.add_validator(validator, "validator")

    #[validator]

    # make solver
    slv = Solver(cfg, domain)

    if reTrain:
        # start solver
        starttime = time.time()
        slv.solve()
        print ('training took ', time.time()-starttime)

    slv.eval()

    #[plotting]
    
    #=================do test dataset 1 ==================
    varr = np.load(to_absolute_path(f'outputs/{cfg.custom.validate_subfolder}/{cfg.network_dir}/validators/validator.npz'), allow_pickle=True)
    varr = np.atleast_1d(varr.f.arr_0)[0]
    
    x = varr['x'].flatten()
    y = varr['y'].flatten()
    t_axis = dataDict['time_axis'][dataDict['train_ind']]

    predU = varr['pred_u']
    trueU = varr['true_u']
    
    nt_test = dataDict['nt_train']
    nGages  = dataDict['n_gages']
    nTest   = dataDict['n_test']
    print (nt_test, nGages)
    predU = predU.reshape(nGages,nTest,nt_test)
    trueU = trueU.reshape(nGages,nTest,nt_test)

    allKGE = np.zeros((nGages,nTest))
    allNRMSE = np.zeros((nGages,nTest))

    for igage in range(nGages):
        for itest in range(nTest):
            pred_u = predU[igage, itest,:]
            true_u = trueU[igage, itest,:]
            pred_u = inverseTransformQ(dataDict['scaler'], pred_u, imethod=1).flatten()
            true_u = inverseTransformQ(dataDict['scaler'], true_u, imethod=1).flatten()

            allKGE[igage, itest] = HydroStats.kge_2012(pred_u,true_u )
            allNRMSE[igage, itest] = HydroStats.nrmse_range(pred_u,true_u )

    allKGE_mean   = np.mean(allKGE, axis=1)
    allKGE_median = np.median(allKGE, axis=1)     
    print ("Test data 1 ")
    print ('mean', allKGE_mean, 'median', allKGE_median)    
    #save the data
    np.save(to_absolute_path(f'outputs/{cfg.custom.validate_subfolder}/train_split_exp{cfg.custom.train_split_no}.npy'), allKGE)
    #plotViolinplot(allKGE.T, allNRMSE.T, dataDict)

def plotViolinplot(kge_arr, nrmse_arr, dataDict):
    """This generates Figure 3 in the manuscript
    """
    import seaborn as sns

    #form a DF
    df =  pd.DataFrame(kge_arr)
    
    fig,axes = plt.subplots(1,2, figsize=(12, 6), sharey=True)
    sns.boxplot(data=kge_arr, native_scale=True, ax=axes[0], orient='h', width=0.5)
    #!!!! need to first fake the ticks before modifying the tick labels
    axes[0].yaxis.set_ticks(range(kge_arr.shape[1]))
    axes[0].set_yticklabels(dataDict['gageids'], fontsize=12)
    axes[0].set_xlabel('KGE', fontsize=12)

    sns.boxplot(data=nrmse_arr, native_scale=True, ax=axes[1], orient='h', width=0.5)
    #!!!! need to first fake the ticks before modifying the tick labels
    axes[1].yaxis.set_ticks(range(nrmse_arr.shape[1]))
    axes[1].set_yticklabels(dataDict['gageids'], fontsize=12)
    axes[1].set_xlabel('NRMSE', fontsize=12)

    plt.tight_layout(w_pad=0.2)
    plt.savefig(to_absolute_path(f"outputs/surrogate_metrics_fold{dataDict['fold_no']}.eps"))
    plt.close()

@modulus.main(config_path="conf", config_name="config_uns_uq_multi_rev")
def compareEnsembleSize(cfg: ModulusConfig) -> None:
    """Plot KGE as a function of ensemble sizes, Figure S1
    """
    import matplotlib.pyplot as plt

    train_exps = [100, 200, 300, 400, '476 (this work)']
    markerlist = ['o', '^', '*', 'D', 'P']
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    for ix, rz_num in enumerate(train_exps):
        allKGE = np.load(to_absolute_path(f'outputs/{cfg.custom.validate_subfolder}/train_split_exp{ix}.npy'))
        allKGE_median   = np.median(allKGE, axis=1)
        ax.plot(allKGE_median, markerlist[ix], label=f'{rz_num}')
    
    gageids = [
        '01435000', '01434498',   '01434176',   '01434105', '01434025', 
        '0143402265', '01434021',  '01434017', '01434013',  '0143400680'
    ]

    ax.set_xticks(range(len(gageids)))
    ax.set_xticklabels(gageids)
    ax.set_xlabel('USGS Gage ID')
    ax.set_ylabel('Mean mKGE')

    plt.legend(loc='best', fontsize=12)
    #uncomment the following to generate EPS file
    #plt.savefig(to_absolute_path(f'outputs/{cfg.custom.validate_subfolder}/ensemblesize_comp.eps'))
    plt.savefig(to_absolute_path(f'outputs/{cfg.custom.validate_subfolder}/ensemblesize_comp.png'))
    plt.close()

def genAttr():
    """Generate uniform random numbers
    """
    return np.random.rand()*2-1.0

def getLoss(y, obsdict, loss_fun, gage_weights, device):
    """Multistation loss fun
    """
    if gage_weights is None:
        weights = np.zeros(len(obsdict.keys()))+1.0    
    else:
        weights = gage_weights
    loss = 0
    for ix, gage_id in enumerate(obsdict.keys()):     
        mask = torch.IntTensor(obsdict[gage_id]['mask']).to(device)
        obs_val = torch.FloatTensor(obsdict[gage_id]['Q'].squeeze()).to(device)       
        loss += weights[ix]*loss_fun(y[ix,0,:].squeeze()[mask==1], obs_val[mask==1])
    loss = loss/np.sum(weights)
    return loss.item()

def evalOneMin(individual, model, x0, y0, t0, obsdict, loss_fun, device, nGages, nT, return_y=False, gage_weights=None):
    a_in = torch.FloatTensor(np.array(individual)).unsqueeze(0)
    a_in = a_in.repeat(len(x0), 1)
    a_in = torch.FloatTensor(a_in).to(device)
    x0 = torch.FloatTensor(x0).to(device)    
    y0 = torch.FloatTensor(y0).to(device)
    t0 = torch.FloatTensor(t0).to(device)
    invar = {'a':a_in, 'x':x0, 'y':y0, 't': t0}
    out = model(invar)
    predU = out['u'].reshape(nGages,1, nT)

    if return_y:
        return predU.data.cpu().numpy()
    else:
        return getLoss(predU, obsdict, loss_fun, gage_weights, device),   

@modulus.main(config_path="conf", config_name="config_uns_uq_multi_rev")
def dodeap(cfg: ModulusConfig) -> None:
    from deap import base
    from deap import creator
    from deap import tools    
    import seaborn as sns
    import time

    starttime = time.time()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    dataDict = getData(cfg, mode='test')

    cfg.network_dir = getNetwork_dir(dataDict['id_train'], dataDict['fold_no'])
    cfg.network_dir += f'exp_{cfg.custom.train_split_no}'
    print ('Network dir is', cfg.network_dir)

    deeponet = genModel(cfg)
    nodes = [deeponet.make_node('deepo')]

    # [datasets for training GA]
    x_train = dataDict["x_train"]
    t_train = dataDict["t_train"]
    y_train = dataDict["y_train"]
    a_train = dataDict["a_train"]
    u_train = dataDict["u_train"]

    # make domain
    domain = Domain()
    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"a": a_train, "x": x_train, "t": t_train, "y":y_train},
        outvar={"u": u_train},
        batch_size=cfg.batch_size.train,
    )

    domain.add_constraint(data, "data")

    #load trained model
    slv = Solver(cfg, domain)
    #print Success loading model: outputs/uppersink_deeponet/deepo.0.pth if successful
    slv.eval()

    # load test dataset 1
    x_test = dataDict["x_test"]
    t_test = dataDict["t_test"]
    y_test = dataDict["y_test"]
    a_test = dataDict["a_test"]
    u_test = dataDict["u_test"]

    obs_dict = dataDict['obs_dict_train']
    gage_weights = dataDict['gage_weights']
    invar_valid = {
        "a": a_test,
        "x": x_test,
        "t": t_test,
        "y": y_test
    }

    outvar_valid = {"u": u_test}
    dataset = DictGridDataset(invar_valid, outvar_valid)

    model = Graph(
            nodes,
            Key.convert_list(dataset.invar_keys),
            Key.convert_list(dataset.outvar_keys),
    )

    device=torch.device("cuda:0")
    model.to(device)
    model.eval()

    x0 = dataDict['x_0']
    y0 = dataDict['y_0']
    t0 = dataDict['t_0']
    a0 = dataDict['a_0']
    nGages = dataDict['n_gages']
    nT = dataDict['nt_train']
    loss_func = setLossFun(itype=cfg.custom.ga.loss_fun)


    # Attribute generator 
    toolbox.register("attr_bool", genAttr)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 13)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalOneMin, model=model, loss_fun=loss_func, 
                    x0=x0, y0=y0, t0=t0, obsdict = obs_dict, nGages=nGages, nT=nT, gage_weights=gage_weights,
                    device=device )

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=cfg.custom.ga.population)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.46, 0.005
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    max_gen = int(cfg.custom.ga.generation)
    g = 0
    #asun 0627, plot GA convergence history
    allFits_mean=[]
    allFits_std =[]
    best_individual_fitness=[]
    # Begin the evolution    
    while max(fits) < 10000 and g < max_gen:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        #asun 06272024,
        allFits_mean.append(mean)
        allFits_std.append(std)
        best_ind_temp = tools.selBest(pop, 1)[0]
        best_individual_fitness.append(best_ind_temp.fitness.values[0])
    print("-- End of (successful) evolution --")
    print ('Wallclock training time', time.time()-starttime)
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    theta = np.array(best_ind)
    #save best parameter for UQ
    np.save(to_absolute_path(f"outputs/{cfg.custom.validate_subfolder}/{cfg.network_dir}/best_param_exp{dataDict['expno']}.npy"), theta)

    theta_min, theta_max = dataDict['theta_scaler']
    theta = 0.5*(theta+1.0)*(theta_max-theta_min)+theta_min
    printParams(theta.tolist())

    #Get best model results for training period
    best_u = evalOneMin(best_ind, model, x0, y0, t0, obs_dict, loss_func, device, nGages, nT, return_y=True)
    #get date range
    t_axis = dataDict['time_axis'][dataDict['train_ind']]
    best_u_dict_train = {}
    for iobs, gage_id in enumerate(obs_dict.keys()):
        obs_u = obs_dict[gage_id]['Q']
        mask = obs_dict[gage_id]['mask']

        pred_u = inverseTransformQ(dataDict['scaler'], best_u[iobs, 0, :]).flatten()
        obs_u  = inverseTransformQ(dataDict['scaler'], obs_u).flatten()

        kge_final = HydroStats.kge_2012(pred_u[mask==1], obs_u[mask==1])
        print ('gage_id', gage_id, f'KGE final {kge_final:4.3f}')

        best_u_dict_train[gage_id] = pred_u
    
    #Get best model results for testing period
    print ('*'*80)
    best_u_dict_test = {}
    obs_dict=dataDict['obs_dict_test']
    print (dataDict['n_gages_test'], dataDict['nt_test'])
    best_u = evalOneMin(best_ind, model, dataDict['xnew'], dataDict['ynew'], dataDict['tnew'], 
                        obs_dict, loss_func, device, dataDict['n_gages_test'], dataDict['nt_test'], return_y=True)

    #for latex
    latex_str = ''
    kge_arr = []
    for iobs, gage_id in enumerate(obs_dict.keys()):
        obs_u = obs_dict[gage_id]['Q']
        mask  = obs_dict[gage_id]['mask']

        pred_u = inverseTransformQ(dataDict['scaler'], best_u[iobs, 0, :]).flatten()
        obs_u  = inverseTransformQ(dataDict['scaler'], obs_u).flatten()

        kge_final = HydroStats.kge_2012(pred_u[mask==1], obs_u[mask==1])
        kge_arr.append(kge_final)

        print ('gage_id', gage_id, f'KGE final {kge_final:4.3f}')
        latex_str += f'{kge_final:4.3f} &'
        best_u_dict_test[gage_id] = pred_u
    print ('*'*10, dataDict['expno'])
    print (latex_str)
    #save the best solution
    pkl.dump([best_u_dict_train, best_u_dict_test], open(to_absolute_path(f"data/best_u_spacetime_{dataDict['expno']}.pkl"), 'wb'))
    pkl.dump(kge_arr, open(to_absolute_path(f"data/best_kge_spacetime_{dataDict['expno']}.pkl"), 'wb'))

    #0210, add pretrain results for table 2
    latex_str = ''
    Q_pretrain = dataDict['Q_pretrain']

    for iobs, gage_id in enumerate(obs_dict.keys()):
        obs_u = obs_dict[gage_id]['Q']
        mask  = obs_dict[gage_id]['mask']

        pred_u = Q_pretrain[iobs, :].flatten()
        obs_u  = inverseTransformQ(dataDict['scaler'], obs_u).flatten()

        kge_final = HydroStats.kge_2012(pred_u[mask==1], obs_u[mask==1])
        print ('gage_id', gage_id, f'Pretrain KGE final {kge_final:4.3f}')
        latex_str += f'{kge_final:4.3f} &'
    print ('*'*10, 'pretrain')
    print (latex_str)

    #asun 0627,plt mean fit values
    allFits_std = np.array(allFits_std)
    allFits_mean = np.array(allFits_mean)
    best_individual_fitness = np.array(best_individual_fitness)
    figure, axes = plt.subplots(1,1,figsize=(8,6))
    axes.plot(allFits_mean, color='#FF6347', linewidth=1.5, label='Mean')
    #axes.plot(best_individual_fitness, label='best fitness value')
    axes.fill_between(range(len(allFits_std)), allFits_mean-3.0*allFits_std, allFits_mean+3.0*allFits_std,color='#FFDAB9', label='Bound')
    axes.set_xlabel("Generation no.")
    axes.set_ylabel("GA fitness")
    plt.legend()
    plt.savefig(to_absolute_path(f"outputs/ga_converg_hist_exp{dataDict['expno']}.png"))
    plt.close()

@modulus.main(config_path="conf", config_name="config_uns_uq_multi_rev")

def plotGAExp(cfg: ModulusConfig) -> None:
    """This generates Figure 4 in the paper
    """
    import itertools
    import seaborn as sns

    nExp = 5
    dataDict = getData(cfg, mode='test')
    cfg.network_dir = getNetwork_dir(dataDict['id_train'], dataDict['fold_no'])
    theta_min, theta_max = dataDict['theta_scaler']
    
    #Experiments start with index 1
    allParam = []
    paramNames = [
    "PT-canopy",
    "PT-ground",
    "PT-snow",
    "PT-trans",
    "sm-rate",
    "sm-diff",
    "manning_n",
    "perm-S1",
    "perm_S2",
    "perm_S3",
    "perm_S4",
    "perm_S5",
    "perm_bedrock"]


    fig = plt.figure(figsize=(16,9))

    gs = fig.add_gridspec(1, 4,  width_ratios=(4,2,1,4), 
                left=0.05, right=0.95, bottom=0.15, top=0.9,
                wspace=0.25, hspace=0.0)            
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[0,3])

    palette = itertools.cycle(sns.color_palette('tab10'))

    for i in range(1,nExp+1):
        theta = np.load(to_absolute_path(f"outputs/{cfg.custom.validate_subfolder}/{cfg.network_dir}/best_param_exp{i}.npy"))
        best_params = 0.5*(theta+1.0)*(theta_max-theta_min)+theta_min
        print ('------Exp', i, '-------')
        print (printParams(best_params))
        expcolor = next(palette)
        markersize= 10
        ax0.plot(best_params[:4], 'o', color=expcolor, label=f'Exp{i}', markersize=markersize,alpha=0.75)        
        ax1.plot(best_params[4:6], 'o', color=expcolor, markersize=markersize,alpha=0.75)        
        ax2.plot(best_params[6],   'o', color=expcolor,markersize=markersize, alpha=0.75)        
        ax3.plot(best_params[7:],  'o', color=expcolor,markersize=markersize,alpha=0.75)        

    ax0.set_xticks(range(4))
    ax0.set_xticklabels(paramNames[:4],  rotation='vertical', fontsize=12)    

    ax1.set_xticks(range(2))
    ax1.set_xticklabels(paramNames[4:6],  rotation='vertical', fontsize=12)    

    ax2.set_xticks(range(1))
    ax2.set_xticklabels(paramNames[6:7],  rotation='vertical', fontsize=12)    

    ax3.set_xticks(range(6))
    ax3.set_xticklabels(paramNames[7:],  rotation='vertical', fontsize=12)    

    #handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(loc='best', fontsize=13)

    plt.savefig(to_absolute_path('outputs/ga_result_barplot.eps'))
    plt.close()

@modulus.main(config_path="conf", config_name="config_uns_uq_multi")
def doSimpleBound(cfg: ModulusConfig) -> None:
    """For Figure 5 parameter sensitivity
    """
    dataDict = getData(cfg, mode='test')

    cfg.network_dir = getNetwork_dir(dataDict['id_train'], dataDict['fold_no'])
    print ('Network dir is', cfg.network_dir)

    deeponet = genModel(cfg)
    nodes = [deeponet.make_node('deepo')]

    # [datasets for training GA]
    x_train = dataDict["x_train"]
    t_train = dataDict["t_train"]
    y_train = dataDict["y_train"]
    a_train = dataDict["a_train"]
    u_train = dataDict["u_train"]

    # make domain
    domain = Domain()
    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={"a": a_train, "x": x_train, "t": t_train, "y":y_train},
        outvar={"u": u_train},
        batch_size=cfg.batch_size.train,
    )

    domain.add_constraint(data, "data")

    #load trained model
    slv = Solver(cfg, domain)
    #print Success loading model: outputs/uppersink_deeponet/deepo.0.pth if successful
    slv.eval()

    # load test dataset 1
    x_test = dataDict["x_test"]
    t_test = dataDict["t_test"]
    y_test = dataDict["y_test"]
    a_test = dataDict["a_test"]
    u_test = dataDict["u_test"]

    obs_dict = dataDict['obs_dict_train']
    gage_weights = dataDict['gage_weights']
    invar_valid = {
        "a": a_test,
        "x": x_test,
        "t": t_test,
        "y": y_test
    }

    outvar_valid = {"u": u_test}
    dataset = DictGridDataset(invar_valid, outvar_valid)

    model = Graph(
            nodes,
            Key.convert_list(dataset.invar_keys),
            Key.convert_list(dataset.outvar_keys),
    )

    device=torch.device("cuda:0")
    model.to(device)
    model.eval()

    loss_func = setLossFun(itype=cfg.custom.ga.loss_fun)

    #load the best parameters
    best_param = np.load(to_absolute_path(f"outputs/{cfg.custom.validate_subfolder}/{cfg.network_dir}/best_param_exp{dataDict['expno']}.npy"))
    boundDict = {}
    bounds = [0.5,0.2]
    for bound in bounds:
        best_param_lb = best_param*(1-bound)
        best_param_ub = best_param*(1+bound)
        boundDict[bound]={'lb': best_param_lb, 'ub':best_param_ub}

    #Get best model results for testing period
    print ('*'*80)
    obs_dict=dataDict['obs_dict_test']
    print (dataDict['n_gages_test'], dataDict['nt_test'])
    pred_mean = evalOneMin(best_param, model, dataDict['xnew'], dataDict['ynew'], dataDict['tnew'], 
                        obs_dict, loss_func, device, dataDict['n_gages_test'], dataDict['nt_test'], 
                        return_y=True)
    pred_mean = pred_mean.reshape(dataDict['n_gages_test'],-1)
    predDict = {}
    for bound in bounds:
        best_param_lb = boundDict[bound]['lb']
        best_param_ub = boundDict[bound]['ub']
        pred_lb = evalOneMin(best_param_lb, model, dataDict['xnew'], dataDict['ynew'], dataDict['tnew'], 
                        obs_dict, loss_func, device, dataDict['n_gages_test'], dataDict['nt_test'], 
                        return_y=True)
        pred_ub = evalOneMin(best_param_ub, model, dataDict['xnew'], dataDict['ynew'], dataDict['tnew'], 
                        obs_dict, loss_func, device, dataDict['n_gages_test'], dataDict['nt_test'], 
                        return_y=True)
        pred_lb = pred_lb.reshape(dataDict['n_gages_test'],-1)
        pred_ub = pred_ub.reshape(dataDict['n_gages_test'],-1)
        predDict[bound]={'lb':pred_lb, 'ub':pred_ub}

    fig, axes = plt.subplots(2,2, figsize=(16,12))
    ax = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    plotStations=['01435000', '01434498', '0143402265', '0143400680']
    labels=['(a)', '(b)', '(c)', '(d)']
    counter=0
    for ix, station_id in enumerate(obs_dict.keys()):
        if station_id in plotStations:
            obs = obs_dict[station_id]['Q']
            plotBoundResult(ax[counter], labels[counter], dataDict, obs, pred_mean[ix,:], predDict, station_id,ix,counter)
            counter+=1
    plt.tight_layout(w_pad=0.2, h_pad=0.32)
    plt.savefig(to_absolute_path(f"outputs/simplebound_singlesite_plot_exp{dataDict['expno']}.eps"))
    plt.close()

def plotBoundResult(ax, plotlable, dataDict, obs_u, pred_u, predDict, station_id,ix,counter):
    """For Figure 5 parameter sensitivity
    """
    t_axis = dataDict['time_axis'][dataDict['test_ind']]  
    obs_u = inverseTransformQ(dataDict['scaler'], obs_u).flatten()
    pred_u = inverseTransformQ(dataDict['scaler'], pred_u).flatten()

    ax.plot(t_axis, pred_u, label='GA', alpha=0.9, color='darkgreen', linewidth=2.0)
    colors = ['#82E0AA', '#5DADE2']
    for bound,boundcolor in zip(predDict.keys(),colors):
        pred_lb = predDict[bound]['lb'][ix,:]
        pred_ub = predDict[bound]['ub'][ix,:]
        pred_lb = inverseTransformQ(dataDict['scaler'], pred_lb).flatten()
        pred_ub = inverseTransformQ(dataDict['scaler'], pred_ub).flatten()
        #ax.plot(t_axis, pred_lb, label='Lower', alpha=0.9, color='green', linestyle='--', linewidth=1.5)
        #ax.plot(t_axis, pred_ub, label='Upper', alpha=0.9, color='green', linestyle='--', linewidth=1.5)
        ax.fill_between(t_axis, pred_lb, pred_ub, color=boundcolor, alpha=0.8,label=f'{int(bound*100)}%')

    kge_final = HydroStats.kge_2012(pred_u, obs_u)

    ax.plot(t_axis, obs_u, label='Obs', alpha=0.8, color='#5D6D7E', linestyle='--', linewidth=2.0)
    ax.set_yscale('log', nonpositive='clip')        
    if (counter==0):
        ax.legend()
    if (counter in [0, 2]):
        ax.set_ylabel('Q (m$^3$/s)',fontsize=12)
    if (counter in [2,3]):
        ax.set_xlabel('Time', fontsize=12)
    ax.set_ylim([0.01, 100])
    ax.set_title(f'{plotlable} Gage{station_id}, mKGE={kge_final:4.3f}',fontsize=14)

if __name__ == "__main__":
    itask = 2
    if itask == 1:
        run()
    elif itask == 2:
        dodeap()        
    elif itask == 3:
        doSimpleBound()
    elif itask == 4:
        plotGAExp()    
    elif itask == 5:
        compareEnsembleSize()