# -*- coding: utf-8 -*-
"""e4_load_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GlkU4dtwyqOoQXpbHp5Vi6sTmCAzugyY

#e4_load_dataset.ipynb
This data set loader uses the e4_get_X_y_sub.py file generated by downloading the python version of the same name Jupyter notebook.

Important note:  The current data set is single subject, however there are
three subject numbers included {11,12,13} in order to perform the subject
based train/validate/test split.

Example usage:

    x_train, y_train, x_test, y_test = e4_load_dataset()
  

Developed and tested using colab.research.google.com  
To save as .py version use File > Download .py

Author:  [Lee B. Hinkle](https://userweb.cs.txstate.edu/~lbh31/), [IMICS Lab](https://imics.wp.txstate.edu/), Texas State University, 2021

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

TODOs:
*
"""


import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
import requests #for downloading zip file
import numpy as np
from tabulate import tabulate # for verbose tables, showing data
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # for one-hot encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys

#enter path to where the git repo was cloned
# Get project root directory path
my_path = os.path.abspath(os.path.dirname(__file__))

# use get_x_y_sub to get partially processed numpy arrays
full_filename = os.path.join(my_path, 'load_data_time_series', 'HAR', 'e4_wristband_Nov2019', 'e4_get_x_y_sub.py')
# Add the path to your pythonpath
sys.path.append(full_filename)

# shutil.copy(full_filename,'e4_get_x_y_sub.py')
from e4_get_x_y_sub import get_X_y_sub

def e4_load_dataset(
    verbose = True,
    incl_xyz_accel = False, # include component accel_x/y/z in ____X data
    incl_rms_accel = True, # add rms value (total accel) of accel_x/y/z in ____X data
    incl_val_group = False, # split train into train and validate
    split_subj = dict
                (train_subj = [11],
                validation_subj = [12],
                test_subj = [13]),
    one_hot_encode = True # make y into multi-column one-hot, one for each activity
    ):
    """calls e4_get_X_y_sub and processes the returned arrays by separating
    into _train, _validate, and _test arrays for X and y based on split_sub
    dictionary.  Note current dataset is single subject labeled as 11, 12, 13
    in order to exercise the code"""
    e4_flist = ['1574621345_A01F11.zip','1574622389_A01F11.zip', '1574624998_A01F11.zip']
    X, y, sub, xys_info = get_X_y_sub(zip_flist = e4_flist)
    log_info = 'Processing e4 files'+str(e4_flist)
    #remove component accel if needed
    if (not incl_xyz_accel):
        print("Removing component accel")
        X = np.delete(X, [0,1,2], 2)
    if (not incl_rms_accel):
        print("Removing total accel")
        X = np.delete(X, [3], 2)  
    #One-Hot-Encode y...there must be a better way when starting with strings
    #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

    if (one_hot_encode):
        # integer encode
        y_vector = np.ravel(y) #encoder won't take column vector
        le = LabelEncoder()
        integer_encoded = le.fit_transform(y_vector) #convert from string to int
        name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print("One-hot-encoding: category names -> int -> one-hot")
        print(name_mapping) # seems risky as interim step before one-hot
        log_info += "One Hot:" + str(name_mapping) +"\n\n"
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        print("One-hot-encoding",onehot_encoder.categories_)
        y=onehot_encoded
        #return X,y
    else:
        print("Not one-hot-encoding")
        y_vector = np.ravel(y) #encoder won't take column vector
        le = LabelEncoder()
        integer_encoded = le.fit_transform(y_vector) #convert from string to int
        name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(name_mapping) # seems risky as interim step before one-hot
        y = integer_encoded

    # split by subject number pass in dictionary
    sub_num = np.ravel(sub[ : , 0] ) # convert shape to (1047,)
    if (not incl_val_group):
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj'] + 
                                        split_subj['validation_subj']))
        x_train = X[train_index]
        y_train = y[train_index]
    else:
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj']))
        x_train = X[train_index]
        y_train = y[train_index]

        validation_index = np.nonzero(np.isin(sub_num, split_subj['validation_subj']))
        x_validation = X[validation_index]
        y_validation = y[validation_index]

    test_index = np.nonzero(np.isin(sub_num, split_subj['test_subj']))
    x_test = X[test_index]
    y_test = y[test_index]
    if (incl_val_group):
        return x_train, y_train, x_validation, y_validation, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test


        if(verbose):
            headers = ("Reshaped data","shape", "object type", "data type")
            mydata = [("x_train:", x_train.shape, type(x_train), x_train.dtype),
                    ("y_train:", y_train.shape ,type(y_train), y_train.dtype),
                    ("x_test:", x_test.shape, type(x_test), x_test.dtype),
                    ("y_test:", y_test.shape ,type(y_test), y_test.dtype)]
            print(tabulate(mydata, headers=headers))

        return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    print("Downloading and processing e4 dataset")
    x_train, y_train, x_test, y_test = e4_load_dataset()
    print("\nreturned arrays without validation group:")
    print("x_train shape ",x_train.shape," y_train shape ", y_train.shape)
    print("x_test shape  ",x_test.shape," y_test shape  ",y_test.shape)

    x_train, y_train, x_validation, y_validation, x_test, y_test = e4_load_dataset(incl_val_group=True)
    print("\nreturned arrays with validation group:")
    print("x_train shape ",x_train.shape," y_train shape ", y_train.shape)
    print("x_validation shape ",x_validation.shape," y_validation shape ", y_validation.shape)
    print("x_test shape  ",x_test.shape," y_test shape  ",y_test.shape)