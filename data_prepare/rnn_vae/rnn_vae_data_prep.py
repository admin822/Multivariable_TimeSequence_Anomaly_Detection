import torch
from datetime import datetime,timedelta
import pandas as pd
import os
import sys
import torch
import numpy as np
import random
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ) 
sys.path.append(BASE_DIR)
from MTAD_util.dataframe_related_util import convert_to_datetime,convert_to_strtime,parse_time

# set seed for reporducibility
SEED=822
torch.manual_seed(SEED)

## 
# Input: One csv file, evenly distributed, no missing points, normalized;
# output: Three csv files: Training set, Validation set, testing_set
##


## This calss will serve as a custom_dataset, input will be a list of tensors with shape=seq_len*dim
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self,ls):
        self.input_list=ls
    def __len__(self):
        return len(self.input_list)
    def __getitem__(self,index):
        return self.input_list[index]

## 
# OFFLINE 
# input:(1) a tuple(starttime,endtime),both these two times should be in format datetime.datetime; (2) a dataframe with the first column being time(type==datetime.datetime); ; (3) save path for the new csv file
# output: (1) save a new csv file to the designated place, in the new file, it will contain all rows between starttime and endtime; (2) New dataframe with the selected rows carved out, this WILL directly change the original df
# what it does: find all rows between start and end, carve em out, group em up into a single csv file and save it, return the carved original dataframe.
##
def save_interval(start_time:datetime,end_time:datetime,df:pd.DataFrame,save_path:str):
    drop_rows=[]
    for i,row in df.iterrows():
        if(row[0]>=start_time and row[0]<=end_time):
            drop_rows.append(i)
        elif(row[0]>end_time):
            break
    if(len(drop_rows)==0):
        print("None of the rows in the file suffice the criteria")
    else:
        new_df=df.iloc[drop_rows,:]
        try:
            new_df.to_csv(save_path,index=False)
            df.drop(drop_rows,axis=0,inplace=True)
            df.reset_index(drop=True,inplace=True)
            print("SAVE_INTERVAL:   {}-{} time interval saved into file".format(start_time,end_time))
        except Exception as err:
            print("ERROR:SAVE_INTERVAL:   failed to save {}-{} time interval into file due to {}".format(start_time,end_time,err))
            raise Exception
            

##
# OFFLINE 
# description:this function will randomly pick n timestamps from the original dataframe and return the list of these points 
# input: (1) original csv path (2) n: n points to be picked (3) time_col_index: index of the time column
# output: a list of datetime.datetime
##
def randomly_pick(path_to_origin_csv:str,n:int,time_col_index:int):
    try:
        df=pd.read_csv(path_to_origin_csv)
    except Exception as err:
        print('ERROR RANDOMLY_PICK: failed to read in csv due to {}'.format(err))
    if(df.shape[0]<n):
        print("ERROR RANDOMLY_PICK:  cannot draw {} samples from {} rows".format(n,df.shape[0]))
        return None
    pot=[i for i in range(df.shape[0])]
    random.seed(SEED)
    straws=random.sample(pot,n)
    time_list=df.iloc[straws,time_col_index].to_list()
    for i,item in enumerate(time_list):
        time_list[i]=parse_time(item,"%Y-%m-%d %H:%M:%S")
    return time_list


## 
# OFFLINE 
# input:(1) the path to the anomaly trace file, anomaly trace file shold have the format of a string: 'machine_name'\t'feature_name'\t'time(in the format of '2019-01-04 13:53:58')\t'anomaly_reason'
# output: (1) a list that contains all the anomalous times(in the format of datetime.datetime)
# what it does: select the row that contains the timestamps in the trace file, and put em into a list, then return the list
##
def get_anomaly_times(path_to_file:str):
    times=[]
    with open(path_to_file,'rb') as f:
        for line in f:
            try:
                decoded_line=line.decode()
                anomaly_time=decoded_line.split('\t')[2]
                times.append(datetime.strptime(anomaly_time,"%Y-%m-%d %H:%M:%S"))
            except Exception as err:
                print("ERROR GET_ANOMALY_TIMES:    Parsing time encounters error due to {}".format(err))
                return None
        print("Parsing time complete, {} anomalous timestamps detected".format(len(times)))
        return times

## 
# OFFLINE
# input: (1) testing_times: a list that contains all the timestamps that need to be tested, (2) path to csv file, (3) test_csv_file_save_path: this is a path to a folder that holds all the test csvs, (4) training_dev_csv_save_path: this is a path to a folder that (5) interval(in hours)
# output: (1) save test csvS to test csv save path, t/d to td save path
# what it does: find all the testing dates in the csv file, take the data from (time-interval,time+interval) out from 
# original csv and write em into SEVERAL test csv(SEVERAL, EACH DATE HAS ONE FILE), save test csvS to designated path, remain of the 
# original csv to its own save path
##
def carve_out_testing_set(testing_times:list,original_path:str,test_save_folder_path:str,train_dev_save_folder_path:str,interval:int):
    if(len(testing_times)==0):
        print("No anomalous timestamps were found, double check your anoamly trace file")
        exit()
    if(testing_times!=None):
        if(os.path.exists(test_save_folder_path)==False):
            try:
                os.mkdir(test_save_folder_path)
            except:
                print("An error occurred during trying to make test save folder")
                exit()
        if(os.path.exists(train_dev_save_folder_path)==False):
            try:
                os.mkdir(train_dev_save_folder_path)
            except:
                print("An error occurred during trying to make train/dev folder")
                exit()
        try:
            df=pd.read_csv(original_path)
            print("CARVE_OUT_TESTING_SET:   csv reading complete, the shape of the dataframe is {}".format(df.shape))
        except Exception as err:
            print("Error: CARVE_OUT_TESTING_SET:    failure happened when reading in csv file due to {}".format(err))
        error_list=convert_to_datetime(df,0)
        print("Threre are {} timestamps cannot be parsed into datetime.datetime".format(len(error_list)))
        if(len(error_list)>0):
            exit()
        ##
        # TODO 
        # add parse fault recovery mechanism
        ##
        bound=timedelta(hours=interval)
        for atime in testing_times:
            upper_bound=atime+bound
            lower_bound=atime-bound
            save_path=os.path.join(test_save_folder_path,'test_{}-{}.csv'.format(convert_to_strtime(lower_bound),convert_to_strtime(upper_bound)))
            try:
                save_interval(lower_bound,upper_bound,df,save_path)
            except:
                pass
        td_save_path=os.path.join(train_dev_save_folder_path,'train_dev.csv')
        try:
            df.to_csv(td_save_path,index=False)
            print('CARVE_OUT_TESTING_SET:    train/dev dataset successfully saved')
        except Exception as err:
            print("Error: CARVE_OUT_TESTING_SET:    failed to save train/dev dataset due to {}".format(err))
        


## 
# ONLINE 
# input:(1)csv_path:str,(2) granularity_type: whether the granularity is in days,hours,minutes (3) granularity_value:int (4): time_col_index, the index of the timestamp column 
# output: (1) if the granularity in the csv file fits the description, then return df with time_col converted into datetime.datetime (2) if not, then gives an error
## 
def check_granularity(csv_path:str,granularity_type:str,granularity:int,time_col_index:int):
    df=pd.read_csv(csv_path)
    error_list=convert_to_datetime(df,time_col_index)
    print("Threre are {} timestamps cannot be parsed into datetime.datetime".format(len(error_list)))
    if(len(error_list)>0):
        exit()
    ##
    # TODO 
    # add parse fault recovery mechanism
    ## 
    if(granularity_type=='day'):
        interval=timedelta(days=granularity)
    elif(granularity_type=='hour'):
        interval=timedelta(hours=granularity)
    elif(granularity_type=='minute'):
        interval=timedelta(minutes=granularity)
    ##
    # TODO 
    # add more granularity types
    ## 
    else:
        print("Error: Granularity Type currently not supportted!")
        return None
    prev=None
    for i in df.shape[0]:
        if(i!=0):
            if(df.iloc[i,time_col_index]-prev!=interval):
                print("Error: At {} row, the interval between timestamp:{} and the previous row:{} does not fit the given granularity!".format(i+1,df.iloc[i,time_col_index],i,prev))
                return None
        prev=df.iloc[i,time_col_index]
    return df



##
# ONLINE
# input:(1) test_csv_path,(2) intervals:(number of rows), (3) all the inputs for function 'check_granularity'
# output: dataloader
# what it does: group all the rows in the input test csv into different intervals, create a torch.utils.data.dataloader 
# using these intervals
##
def get_test_dataset(test_csv_path:str,interval:int,granularity_type:str,granularity:int,time_col_index:int):
    df=check_granularity(test_csv_path,granularity_type,granularity,time_col_index)
    if(df==None):
        return None
    test_list=[]
    for i in range(df.shape[0]):
        if(df.shape[0]-i<=interval):
            break
        else:
            test_list.append(torch.from_numpy(np.array(df.iloc[i:i+6,:])))
    test_set=custom_dataset(test_list)
    return torch.utils.data.DataLoader(test_set,shuffle=False,batch_size=1)
    


##
# ONLINE
# input: (1)training_dev_csv_path,(2)ratio:ratio of the whole dataset that will be used as training set,(3) interval:number of rows that will be group into an interval,
# (4) dev_set_save_path: a path to save the dev_set(type:'custom_dataset:extend torch.utils.data.Dataset'),(5) training_set_save_path:a path to save the training set(type:'custom_dataset:extend torch.utils.data.Dataset') (6) batch_size: batch size for tranining and dev set 
# (7)all inputs for function 'check_granularity'
# Output: two dataloaders, one for training ,the other for dev 
# what it does: read the whole csv file, group rows into small sequences with a length==interval, randomly carve out ratio*len(whole_dataset) as training set and the rest as dev set 
# Considering that the Anomaly Detector module will require data in the dev set, we save dev set into the designated path
## 
def get_train_dev_dataset(train_dev_csv_path:str,interval:int,ratio:float,dev_save_path:str,training_set_save_path:str,granularity_type:str,granularity:int,time_col_index:int,batch_size=32):
    df=check_granularity(train_dev_csv_path,granularity_type)
    if(df==None):
        return None
    whole_list=[]
    for i in range(df.shape[0]):
        if(df.shape[0]-i<=interval):
            break
        else:
            whole_list.append(torch.from_numpy(np.array(df.iloc[i:i+6,:])))
    whole_dataset=custom_dataset(whole_list)
    training_set,dev_set=torch.utils.data.dataset.random_split(whole_dataset,[int(0.7*len(whole_dataset)),len(whole_dataset)-int(0.7*len(whole_dataset))])
    torch.save(training_set,training_set_save_path)
    torch.save(dev_save_path,dev_save_path)
    return torch.utils.data.DataLoader(training_set,shuffle=True,batch_size=32),torch.utils.data.DataLoader(dev_set,shuffle=True,batch_size=32)

# df=pd.DataFrame({'a':[1,2,3],'b':[3,4,5]})
# df.to_csv(os.path.join('../../data/suite_six/anomaly','test.csv'),index=False)