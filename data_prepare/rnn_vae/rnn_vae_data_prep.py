import torch


## 
# Input: One csv file, evenly distributed, no missing points, normalized;
# output: Three csv files: Training set, Validation set, testing_set
##

## 
# OFFLINE
# input: a list of testing times, csv file, test_csv_file_save_path, training_dev_csv_save_path, bound(in hours)
# output: save test csvS to test csv save path, t/d to td save path
# what it does: find all the testing dates in the csv file, take the data from (time-interval,time+interval) out from 
# original csv and write em into SEVERAL test csv(SEVERAL, EACH DATE HAS ONE FILE), save test csvS to designated path, remain of the 
# original csv to the its own save path
##
def carve_out_testing_path(testing_times:list,original_path:str,test_save_path:str,train_dev_save_path:str,interval:int):
    pass

## This calss will serve as a custom_dataset, input will be a list of tensors with shape=seq_len*dim
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass


##
# ONLINE
# input:test_csv_path,intervals:(number of rows)
# output: dataloader
# what it does: group all the rows in the input test csv into different intervals, create a torch.utils.data.dataloader 
# using these intervals
##
def get_test_dataset(test_csv_path:str,interval:int):
    pass


##
# ONLINE
# input: training_dev_csv_path,ratio,interval,dev_set_save_path
# Output: two dataloaders 
# what it does: read the whole csv file, group rows into small sequences with a length==interval, randomly carve out ratio*len(whole_dataset) as training set and the rest as dev set 
# Considering that the Anomaly Detector module will require data in the dev set, we save dev set into the designated path
## 
def get_train_dev_dataset(train_dev_csv_path:str,interval:int,ratio:float,dev_save_path:str):
    pass