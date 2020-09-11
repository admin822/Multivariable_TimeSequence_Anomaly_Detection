import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
sys.path.append(BASE_DIR)
from data_prepare.rnn_vae.rnn_vae_data_prep import carve_out_testing_set,randomly_pick,get_anomaly_times

##
# OFFLINE 
# description:this script will serve as an interface for users to create test files(anomaly,multiple) and save em into a designated folder 
# input: 
# (1) path_to_error_trace_file 
# (2) path_to_test_folder: path to the folder where all test csvs will be saved 
# (3) path_to_train_dev_folder:path to the folder where train/dev csv will be saved 
# (4) path_to_original_path: path tot the csv file that holds the original data 
# (5) interval: rows in with timestamps that are in (center_point-interval,center_point+interval) will be selected from the original file
# output: test_csvs and train/dev csv will all be saved to the designated folders
##

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-error_file_path',help='path to the error trace file',default='../../anomaly_trace/suite_six.txt')
    parser.add_argument('-train_folder',help='path to the folder where train/dev csv will be saved',default='../../data/suite_six/normal')
    parser.add_argument('-test_folder',help='path to the folder where all test csvs will be saved ',default='../../data/suite_six/anomaly')
    parser.add_argument('-original_file_path',help='path tot the csv file that holds the original data',default='../../data/suite_six/min_max_suite_six_all_features.csv')
    parser.add_argument('-interval',help='rows in with timestamps that are in (center_point-interval,center_point+interval) will be selected from the original file',default=24)
    args=parser.parse_args()
    testing_times=get_anomaly_times(args.error_file_path)
    carve_out_testing_set(testing_times,args.original_file_path,args.test_folder,args.train_folder,args.interval)
    