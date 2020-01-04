import pandas as pd
import numpy as np
import category_encoders as ce
import re
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import dill

"""
===============================================
  Missing 없는 데이터 생성하는 Class
===============================================
"""


class Gene_Missing :
    def __init__(self,  data , obj_col , ord_col , num_col , p_miss , num_scaler ) :
        self.data = data
        self.obj_col = obj_col
        self.ord_col = ord_col
        self.re_col = [col + "_nan" for col in self.ord_col]
        self.num_col = num_col
        self.usecols = data.columns.tolist()
        self.p_miss = p_miss
        self.scaler = num_scaler
        self.ce_one_hot = ce.OneHotEncoder(cols = obj_col , 
            drop_invariant= True , 
            handle_unknown = "value" , 
            handle_missing = "return_nan" , 
            use_cat_names= True)
        self.ce_one_hot.fit(X = self.data)
        self.scaler.fit(self.data[self.ord_col + self.num_col])
        self.container = {}
        self.container["num_col"] = num_col
        self.container["ord_col"] = ord_col
        self.container["obj_col"] = obj_col
        self.ori_missing = None

    def cat_num_(self, data) :
        """
        Category + Numerical
        return : Missing Matrix
        """
        No , ori = data.shape
        self.data_len = No
        RAW_Dim = self.store[-1][1]
        p_miss_vec = self.p_miss * np.ones((No,1)) 
        Missing = np.zeros((No,RAW_Dim))
        ori_Missing = np.zeros((No,ori))
        for idx , st in enumerate(self.store) :
            len_ = st[1] -st[0]
            A = np.random.uniform(0., 1., size = [No,])
            A1 = np.expand_dims(A , axis = 1)
            B_ = A > p_miss_vec[idx]
            ori_Missing[:,idx] = 1.*B_
            B = A1 > p_miss_vec
            Missing[:,st[0]:st[1]] = 1.* np.tile(B , (len_))
        self.ori_missing = 1-ori_Missing
        rMiss = 1-Missing
        return rMiss

    def num_(self , data) :
        """
        Only Numeric Matrix
        return : Missing Matrix
        """
        No , Dim = data.shape
        self.data_len = No
        p_miss_vec = self.p_miss * np.ones((Dim,1)) 
        Missing = np.zeros((No,Dim))
        for i in range(Dim):
            A = np.random.uniform(0., 1., size = [No,])
            B = A > p_miss_vec[i]
            Missing[:,i] = 1.*B
        self.ori_missing = 1-Missing
        return self.ori_missing


    def missing_matrix(self, data , re) :
        """
        return : Missing Matrix
        """
        if re == True :
            self.ori_missing = None
        else :
            pass
        if self.obj_col is None : 
            self.ord_col2 = [idx for idx , col in enumerate(self.usecols) if col in self.ord_col ]
            if self.ori_missing is None :
                miss_mat = self.num_(data)
            else :
                miss_mat = self.miss_mat
        else :
            self.obj_col2 = [idx for idx , col in enumerate(self.usecols) if col in self.obj_col ]
            self.ord_col2 = [idx for idx , col in enumerate(self.usecols) if col in self.ord_col ]
            start_idx = 0
            store = []
            for idx , col in enumerate(self.usecols) :
                if col in self.obj_col :
                    nn = data[col].nunique()
                    aa = [start_idx , start_idx + nn]
                    start_idx += nn
                    store.append(aa)
                else :
                    aa = [start_idx , start_idx +1]
                    store.append(aa)
                    start_idx += 1
            self.store = store 
            if self.ori_missing is None :
                miss_mat = self.cat_num_(data)
            else :
                miss_mat = self.miss_mat
        self.miss_mat = miss_mat
        return miss_mat

    def scale_trans(self, ) :
        """
        Return Numerical Normalization
        """
        result = deepcopy(self.data)
        result[self.ord_col + self.num_col] = self.scaler.transform(result[self.ord_col + self.num_col])
        return result

    def show_data(self ,scale = True ) :
        """
        Return : Data( scaling , onehot 처리 된 것)
        """
        if self.obj_col is None : 
            if scale == True :
                result = self.scale_trans()
            else :
                result = self.data
        else :
            if scale == True :
                result = self.scale_trans()
                result = self.ce_one_hot.transform(result)
            else :
                result = self.ce_one_hot.transform(self.data)
        return result

    def generate(self , scale , re= True ) :
        """
        Return : misiing data , missing matrix
        """
        if scale == True :
            result11 = self.scale_trans()
        else :
            result11 = self.data
        result = result11.dropna(axis = 0)
        if self.obj_col is None : 
            Missing = self.missing_matrix(result , re)
            missing_data = deepcopy(result)
            #missing_data[Missing ==1] = np.nan
        else :
            Missing = self.missing_matrix(result , re)
            missing_data = self.ce_one_hot.transform(result)
            #missing_data[Missing == 1] = np.nan
        return missing_data , Missing
    
    def train_test_split(self, train_prob , scale , na_gene , 
                         save = False , numpy = True ) : 
        """
        Return : trainX , testX , trainM , testM
        """
        if na_gene == True :
            Data , _ = self.generate(scale)
        else :
            Data = self.show_data(scale)
        idx = np.random.permutation(self.data_len)
        Train_No = int(self.data_len * train_prob)
        Test_No = self.data_len - Train_No
        self.trainX = Data.iloc[idx[:Train_No],:]
        self.testX = Data.iloc[idx[Train_No:],:]
        self.trainM = self.miss_mat[idx[:Train_No],:]
        self.testM = self.miss_mat[idx[Train_No:],:]
        if numpy == True :
            self.trainX = self.trainX.values
            self.testX = self.testX.values
        else :
            pass
        if save == True :
            self.container["trainX"] = self.trainX 
            self.container["testX"] = self.testX 
            self.container["trainM"] = self.trainM 
            self.container["testM"] =  self.testM
        else :
            pass
        return self.trainX , self.testX , self.trainM , self.testM
        
    def save_info(self, file_name) :
        """
        SAVE Missing Information
        """
        self.container["scaler"] = self.scaler
        self.container["missing_matrix"] = self.miss_mat
        self.container["ori_missing_matrix"] = self.ori_missing
        if self.obj_col is None : 
            self.container["columns"] = self.usecols
        else :
            self.container["columns"] = self.ce_one_hot.feature_names
            self.container["ce_encoder"] = self.ce_one_hot
            mapping = self.ce_one_hot.ordinal_encoder.mapping
            cat_2_ord = {}
            ord_2_cat = {}
            for idx , col_dict in enumerate(mapping) :
                col , map_ = col_dict["col"] , col_dict["mapping"] 
                cat_2_ord[col] = dict(zip(map_.index.tolist() , map_.values))
                ord_2_cat[col] = dict(zip( map_.values.astype(float) , map_.index.tolist()))
            self.container["cat2ord"] = cat_2_ord
            self.container["(ord+1)2cat"] = ord_2_cat
            obj_info = {}
            obj_range = []
            obj_start = []
            coool = self.ce_one_hot.feature_names
            for obj in  self.obj_col :
                find =\
                [idx for idx , col in enumerate(coool) if re.search("^{}_".format(obj) , col)]
                start_idx = min(find)
                col_n = len(find)
                obj_start.append(start_idx)
                obj_range.extend(find)
                obj_info[obj] = {"start_idx" : start_idx , "n" : col_n}
            self.container["obj_info"] = obj_info
            self.container["obj_range"] = obj_range
            not_obj_idx = list(set(np.arange(len(coool))) - set(obj_range))
            cond = list(np.sort(not_obj_idx + obj_start))
            self.container["cat_num_idx_info"] = cond
        with open(file_name, 'wb') as f:
            dill.dump(self.container, f)
        return print("{} missing 정보 저장 완료".format(file_name))
    
    def load_info(self, file_name) : 
        """
        load missing infro
        """
        with open(file_name, 'rb') as f:
            result = dill.load(f)
        return result



            


