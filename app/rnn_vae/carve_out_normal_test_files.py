import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
sys.path.append(BASE_DIR)
from data_prepare.rnn_vae.rnn_vae_data_prep import carve_out_testing_set,randomly_pick,get_anomaly_times

##
# OFFLINE 
# description:this script will serve as an interface for users to create test files(normal,multiple) and save em into a designated folder 
# input: 
# (1) path_to_test_folder: path to the folder where all test csvs will be saved 
# (2) path_to_train_dev_folder:path to the folder where train/dev csv will be saved 
# (3) path_to_original_path: path tot the csv file that holds the original data 
# (4) interval: rows in with timestamps that are in (center_point-interval,center_point+interval) will be selected from the original file
# output: test_csvs and train/dev csv will all be saved to the designated folders
##

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-train_folder',help='path to the folder where train/dev csv will be saved',default='../../data/suite_six/normal/training')
    parser.add_argument('-test_folder',help='path to the folder where all test csvs will be saved ',default='../../data/suite_six/normal/testing')
    parser.add_argument('-original_file_path',help='path tot the csv file that holds the original data',default='../../data/suite_six/normal/train_dev.csv')
    parser.add_argument('-interval',help='rows in with timestamps that are in (center_point-interval,center_point+interval) will be selected from the original file',default=24)
    args=parser.parse_args()
    testing_times=randomly_pick(args.original_file_path,12,0)
    carve_out_testing_set(testing_times,args.original_file_path,args.test_folder,args.train_folder,args.interval)