#Author: Alex Sun
#Date: 01242023
#Purpose: parse ats ensemble 
#rev date: 02052023, add forcing data
#rev date: 02062023, note flow is log-transformed. The unit of flow is m3/d
#Notes from Pin Shuai
#1. ~400 have completed 10 years of simulations (1990-2000). @Alex, let me know if you need longer runs.
#2. The discharge results are organized by outlet gages. The column "obs discharge [m^3/d]” is observed streamflow from USGS.
#  The rest of the columns are ensembles. For example, "oneBedrock_ens.1001” is the 1001th ensemble and the corresponding model parameters used can be found 
# in “oneBedrock_ensemble_parameters.csv” with realization_id = 1001.
#rev date: 11072023, switch to new ATS ensemble sent by Peishi
#==========================================================-----------------
import os, sys
import pandas as pd
import numpy as np
import datetime
import pickle as pkl

def getEnsParam(fileName):
    """Get ensemble parameters for each realization
    Note: the valid realization start from 100x
    """
    df = pd.read_csv(os.path.join('data', fileName))
    return df

def parseDaymetForcingData(gageIDs):
    """parse CSV to form DF for each station
    """
    forcingDict={}
    forcingvar = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    for id in gageIDs:
        #collate all forcing data for each station
        stationDF = []
        for avar in forcingvar:
            daymet_file = f"data/daymet_forcing/Daymet_{avar}_{id}.csv"
            #first column is time
            df = pd.read_csv(daymet_file, index_col=0)
            stationDF.append(df)
        stationDF = pd.concat(stationDF, axis=1)
        forcingDict[id] = stationDF
    return forcingDict

def getEnsRun(gageIDs,paramDF,forcingDict):
    """Get 500 ensemble run outputs
    Note: only 478 are usable
    """
    allCols = []
    cols = None
    for id in gageIDs:
        filename = os.path.join('data', f'ensemble_daily_q_{id}-updated.csv')
        df = pd.read_csv(filename, index_col=0)
        if cols is None:
            cols = df.columns[1:].values
        #drop any column (realization ) with na values
        df = df.iloc[:, 1:]
        count = df.isna().sum()       
        count = count[count<4] # tolerate up to 4 NaN for each realization?
        allCols.append(count.index.values.tolist())    


    #loop through all columns to find valid columns
    goodRealizations = []
    for item in cols:
        goodItem = True
        for alist in allCols:
            if not item in alist:
                goodItem=False
                break
        if goodItem:
            goodRealizations.append(item)
    print ('Good realizations', len(goodRealizations))

    #save cleaned df's
    allDF = {}
    startDate = '1990-10-03'
    endDate = '2000-10-01'
    start_date = datetime.datetime.strptime(startDate, '%Y-%m-%d')
    end_date   = datetime.datetime.strptime(endDate,   '%Y-%m-%d')    

    drng = pd.date_range(start_date, end_date, freq='1D')
    for id in gageIDs:
        filename = os.path.join('data', f'ensemble_daily_q_{id}-updated.csv')
        df = pd.read_csv(filename, index_col=0)
        df = df.loc[startDate:endDate, goodRealizations]
        df.index = pd.to_datetime(df.index)
        df = df.reindex(drng)
        df = df.ffill()
        allDF[id] = df
    
    #filter paramDF
    goodRZ = []
    for item in goodRealizations:
        rz_num = int(item[-4:])
        goodRZ.append(rz_num)
    #get label based rz num
    paramDF = paramDF[paramDF['realization_id'].isin(goodRZ)]
    #trim forcing series
    allForcingDF = {}
    for id in gageIDs:
        df = forcingDict[id]
        df = df.loc[startDate:endDate, :]
        df.index = pd.to_datetime(df.index)
        df = df.reindex(drng)
        df = df.ffill()
        allForcingDF[id] =df 
    #check data length
    assert (allDF[gageIDs[0]].shape[0] == allForcingDF[gageIDs[0]].shape[0])
    return allDF, paramDF, allForcingDF  

def getEnsRunNew(gageIDs,paramDF,forcingDict):
    """Get ATS ensemble run [asun 11072023]
    600 ensemble run outputs
    Note: only 596? are usable
    11/09/2023 add observation data parsing
    """
    allCols = []
    cols = None
    for id in gageIDs:
        filename = os.path.join('data', f'atsensemblenew/ensemble_daily_q_{id}.csv')
        df = pd.read_csv(filename, index_col=0)
        if cols is None:
            cols = df.columns[1:].values
        #drop any column (realization ) with na values
        df = df.iloc[:, 1:]
        count = df.isna().sum()       
        count = count[count<4] # tolerate up to 4 NaN for each realization?
        allCols.append(count.index.values.tolist())    

    allGageObs={}
    for id in gageIDs:
        filename = os.path.join('data', f'atsensemblenew/ensemble_daily_q_{id}.csv')
        df = pd.read_csv(filename, index_col=0, usecols=['datetime', 'obs discharge [m^3/d]'])
        df.columns = ['Q']
        #convert to m3/s
        df['Q'] = df['Q'].apply(lambda x: x/(86400.0))
        allGageObs[id] = df

    #loop through all columns to find valid columns
    goodRealizations = []
    for item in cols:
        goodItem = True
        for alist in allCols:
            if not item in alist:
                goodItem=False
                break
        if goodItem:
            goodRealizations.append(item)
    print ('Good realizations', len(goodRealizations))

    #save cleaned df's
    allDF = {}
    startDate = '1990-10-03'
    endDate = '2000-10-01'
    start_date = datetime.datetime.strptime(startDate, '%Y-%m-%d')
    end_date   = datetime.datetime.strptime(endDate,   '%Y-%m-%d')    

    drng = pd.date_range(start_date, end_date, freq='1D')
    for id in gageIDs:
        filename = os.path.join('data', f'atsensemblenew/ensemble_daily_q_{id}.csv')
        df = pd.read_csv(filename, index_col=0)
        df = df.loc[startDate:endDate, goodRealizations]
        df.index = pd.to_datetime(df.index)
        df = df.reindex(drng)
        df = df.ffill()
        allDF[id] = df

    #Filter paramDF
    goodRZ = []
    for item in goodRealizations:
        rz_num = int(item[item.find('.')+1:])
        goodRZ.append(rz_num)
    #get label based rz num
    paramDF = paramDF[paramDF['realization_id'].isin(goodRZ)]
    #trim forcing series
    allForcingDF = {}
    for id in gageIDs:
        df = forcingDict[id]
        df = df.loc[startDate:endDate, :]
        df.index = pd.to_datetime(df.index)
        df = df.reindex(drng)
        df = df.ffill()
        allForcingDF[id] =df 
    #check data length
    assert (allDF[gageIDs[0]].shape[0] == allForcingDF[gageIDs[0]].shape[0])
    return allDF, paramDF, allForcingDF, allGageObs

def main():
    #good gages
    #['01434025', '01434017', '01435000', '01434498']
    gageids = [
        '01435000', '01434498', '01434176', '01434105', 
        '01434092', '01434025', '0143402265', '01434021', 
        '01434017', '01434013', '0143400680'
    ]
    #get forcing with key=stationID, val = combined daymet forcings
    forcingDict = parseDaymetForcingData(gageids)
    #asun 11072023, add branch for versions
    ensemble_version = 1 #valid choice are 0, 1
    if ensemble_version == 0:
        ensParamFile = 'oneBedrock_ensemble_parameters.csv'
        paramDF = getEnsParam(ensParamFile)
        #combine all data into a single DF
        allDF, paramDF,forcingDF = getEnsRun(gageids, paramDF=paramDF, forcingDict=forcingDict)
        #save cleaned df
        pkl.dump((paramDF, allDF, forcingDF),open(os.path.join('data', 'ats_run_clean.pkl'), 'wb'))

    elif ensemble_version == 1:
        #asun1107, switch to atsensemblenew
        ensParamFile = 'atsensemblenew/oneBedrock_ensemble_parameters.csv'
        paramDF = getEnsParam(ensParamFile)
        #copy this file to /home/suna/work/modulus/deeponet/data
        #combine all data into a single DF
        allDF, paramDF, forcingDF, allGageObs = getEnsRunNew(gageids, paramDF=paramDF, forcingDict=forcingDict)
        #save cleaned df
        #copy this file to /home/suna/work/modulus/deeponet/data
        pkl.dump((paramDF, allDF, forcingDF,allGageObs),open(os.path.join('data', 'atsensemblenew/ats_run_clean_v1.pkl'), 'wb'))

    else:
        raise Exception("invalid ensemble version")


if __name__ == '__main__':
    main()
