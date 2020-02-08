
import pandas as pd
import tensorflow as tf
from tqdm import tqdm , tqdm_notebook
import numpy as np
import category_encoders as ce
import re
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler , RobustScaler , MinMaxScaler
from sklearn.compose import ColumnTransformer
import warnings , sys
warnings.filterwarnings('ignore')
from copy import deepcopy
sys.path.append('/home/advice/Python/SR/Paper/')
import matplotlib.pyplot as plt
sys.path.append('/home/advice/Python/SR/Paper/KDD/')
from GAINUtils import splitData , DataHadling
from Common import calculate_rmse_pfc

import argparse
parser = argparse.ArgumentParser(prog='GAIN For KDD')
parser.add_argument('--filename', type=str)
parser.add_argument('--gpu', type=int , default=0)
parser.add_argument('--batch_size', type=int, default=1000)
args = parser.parse_args()

not_missing_data = pd.read_csv("/home/advice/Python/SR/Data/kdd/uci/uci_creditcard-train-0.0-0.0.csv",
                               sep = ",")

not_missing_data = not_missing_data[not_missing_data.sep_idx == 1]

in_var = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
          "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
          "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
num_var = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
           "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
           "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
fac_var = sorted(['EDUCATION' , 'SEX', 'MARRIAGE'])
in_var =  num_var + fac_var
in_var = sorted(in_var)
not_missing_data = not_missing_data[in_var]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler2', MinMaxScaler(feature_range=(-1,1))),])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))])

not_missing_data[num_var] = numeric_transformer.fit_transform(not_missing_data[num_var])
not_missing_data[fac_var] = categorical_transformer.fit_transform(not_missing_data[fac_var])


import os
static = "/home/advice/Python/SR/Data/kdd/uci"
file_path =  args.filename
missing_path = os.path.join(static , file_path )
missing_data = pd.read_csv(missing_path ,sep = ",")
missing_data = missing_data[missing_data.sep_idx == 1]
missing_data = missing_data[in_var]
missing_matrix = missing_data[in_var].isna().values
missing_data[num_var] = numeric_transformer.fit_transform(missing_data[num_var])
missing_data[fac_var] = categorical_transformer.fit_transform(missing_data[fac_var])
comp_pfc , comp_rmse = calculate_rmse_pfc(not_missing_data , missing_data , missing_matrix,
                                          in_var , num_var , fac_var)
print(file_path , comp_pfc , comp_rmse)
raw_result_pfc_rmse = {"pfc" : comp_pfc ,
                       "rmse" : comp_rmse}

missing_path = os.path.join(static , file_path)
missing_data = pd.read_csv(missing_path ,sep = ",")
train , valid = splitData(missing_data , [1,0] , key="sep_idx")
DataHandle = DataHadling(in_var , num_var , NAColDrop=True)
DataHandle.start(train)
DataHandle.evaluate(valid)
DataHandle.setting()
DataHandle.FindColumnsIndex()
train_onehot = DataHandle.train_result
valid_onehot = DataHandle.valid_result

train_onehot = DataHandle.select(train_onehot)
valid_onehot = DataHandle.select(valid_onehot)
train_missing = DataHandle.missing_matrix(train_onehot)
valid_missing = DataHandle.missing_matrix(valid_onehot)
DataHandle.WeightInfo(train_onehot,None)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler(feature_range=(-1,1))
minmax.feature_range
train_onehot = DataHandle.numscaler(train_onehot , minmax , eval=False )
valid_onehot = DataHandle.numscaler(valid_onehot , minmax , eval=True )
information = {}
info = DataHandle.__dict__

for i in ["fac_var","in_var", "num_var", "ce",
          "weight_info", "scaler","key_store","store",
          "train_onehot_col" , "ForComp" , "ForComp_missing_matrix"
         ] :
    information[i] = info[i]

information["train_data"]    = train_onehot
information["train_missing"] = train_missing
information["valid_data"]    = valid_onehot
information["valid_missing"] = valid_missing

from GAIN import GAIN

mg = GAIN(gen_dim = [100, 90, 75 , 65 , 55 ]  ,
        dis_dim = [100, 90, 75 , 65 , 55 ] ,
        information =  information ,
        gpu = args.gpu ,)

comp_result = {}
comp_result["not_missing_data"] = not_missing_data
comp_result["comparision"] = raw_result_pfc_rmse
comp_result
mg.update_dict_(comp_result)


def tf_mish(x) :
    return x * tf.nn.tanh(tf.nn.softplus(x))
mb_size = args.batch_size
mg.set_env((train_onehot.fillna(100), train_missing),
       (valid_onehot.fillna(100), valid_missing) ,
       mb_size ,
       hint = 0.5 ,
       Gact = tf.nn.selu ,
       Dact = tf.nn.selu ,
       epoch = 1000,
       alpha = 3.0 ,
       weight_regularizer = 0 ,
       SN = True ,
       max_norm = False ,
       lr = 1e-4 ,
       patience = 20 ,
       cutoff = 1e-20
      )

folder = file_path.split(".csv")[0]
mg.run_parallel(folder , 5)

