import category_encoders as ce
import dill , re , numpy as np
import pandas as pd
from copy import deepcopy
"""
===============================================
  Missing 있는 데이터에서 Missing 관련 정보 얻기
===============================================
"""
import warnings , seaborn as sns
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(15.7,10.27)})

class MissInfo :
    def __init__(self ,X ,  obj_col , num_col , ord_col , num_scaler) :
        self.X = X
        self.rawT = deepcopy(X)
        self.TotalColumns = X.columns.tolist()
        self.info = {}
        self.obj_col = obj_col
        self.ord_col = ord_col
        self.num_col = num_col
        self.scaler = num_scaler
        self.RAW_missing = deepcopy(X)
        self.RAW_missing_matrix = X.isna().values
        self.scaler.fit(self.X[self.ord_col + self.num_col])
        self.usecols = self.X.columns.tolist()
        self.obj_idx =\
        [idx for idx , col in enumerate(self.usecols) if col in self.obj_col ]
        self.ord_idx =\
        [idx for idx , col in enumerate(self.usecols) if col in self.ord_col ]
        self.info["X"] = self.X
        self.info["original_column"] = self.usecols
        self.info["scaler"] = self.scaler
        self.info["obj_col"] = self.obj_col
        self.info["ord_col"] = self.ord_col
        self.info["num_col"] = self.num_col
        self.info["ce_replace"] = [col +"_nan"  for col in obj_col ]
        self.info["obj_idx"] = self.obj_idx
        self.info["ord_idx"] = self.ord_idx
#         self.info["missing_matrix"] = self.RAW_missing_matrix
        self.category =\
         ce.OneHotEncoder(cols = obj_col , 
                          verbose = 1 , 
            drop_invariant= True , 
            handle_unknown = "value" , 
            handle_missing = "return_nan" , 
            use_cat_names= True)

    def fit(self , ) :
        self.category.fit(self.X)
        self.info["ce_encoder"] = self.category
        self.info["columns"] = self.category.feature_names
        start_idx = 0
        store = []
        for idx , col in enumerate(self.usecols) :
            if col in self.obj_col :
                nn = self.X[col].dropna().nunique()
                aa = [start_idx , start_idx + nn]
                start_idx += nn
                store.append(aa)
            else :
                aa = [start_idx , start_idx +1]
                store.append(aa)
                start_idx += 1
        self.store = store 
        obj_info = {}
        obj_range = []
        obj_start = []
        coool = self.category.feature_names
        for obj in  self.obj_col :
            find =\
            [idx for idx , col in enumerate(coool) if re.search("^{}_".format(obj) , col)]
            start_idx = min(find)
            col_n = len(find)
            obj_start.append(start_idx)
            obj_range.extend(find)
            obj_info[obj] = {"start_idx" : start_idx , "n" : col_n}
        self.info["obj_info"] = obj_info
        self.info["obj_range"] = obj_range
        not_obj_idx = list(set(np.arange(len(coool))) - set(obj_range))
        cond = list(np.sort(not_obj_idx + obj_start))
        self.info["cat_num_idx_info"] = cond
        obj_weight_info = []
        for col in self.TotalColumns :
            if col in self.obj_col :
                weight_class = pd.DataFrame(self.rawT[col].value_counts(normalize=True)).T
                ck = "{}_".format(col)
                order =[re.sub(ck , "" , col)  for col in self.info["columns"] if re.search(ck , col)]
                weight_info = np.squeeze(weight_class[order].values).tolist() 
                obj_weight_info.append(weight_info)
            else :
                obj_weight_info.append([1])
        self.info["obj_weight_info"] = obj_weight_info
        
        return print("Missing 정보 함축")
    
    def smooth(self , X , degree=0.1 ) :
        obj_info = self.info["obj_info"]
        for col , dict_ in obj_info.items() :
            start = dict_["start_idx"]
            n = dict_["n"]
            r =  X.iloc[: , start:(start+n)]
            rr = r + np.random.uniform(0,degree,size=r.shape)
            X.iloc[: , start:(start+n)] = rr.values / np.expand_dims(np.sum(rr.values , axis = 1), axis = 1)
        return X
    
    
    def DL_train_test_split(self, train_prob , p_miss , 
                            scale = True , save = False , numpy = True , 
                            reproduce = True , smooth = True, degree = 0.2) :
        self.p_miss = p_miss
        Data = self.X.dropna(axis = 0)
        ##
        if scale == True :
            Data[self.ord_col + self.num_col] = self.scaler.transform(Data[self.ord_col + self.num_col])
        else : pass
        Missing = self.missing_matrix(Data , reproduce)
        self.ori_raw_pd = Data
        self.info["ori_raw_pd"] = self.ori_raw_pd
        Data =self.category.transform(Data)
        self.ori_transform_pd = Data
        self.info["ori_transform_pd"] = self.ori_transform_pd
        ##
        if smooth == True :
            Data = self.smooth(X = Data , degree = degree)
        else : pass
        data_len , Dim = Data.shape
        print(Dim)
        idx = np.random.permutation(data_len)
        Train_No = int(data_len * train_prob)
        Test_No = data_len - Train_No
        self.trainX = Data.iloc[idx[:Train_No],:]
        self.testX = Data.iloc[idx[Train_No:],:]
        self.miss_mat = Missing
        self.trainM = self.miss_mat[idx[:Train_No],:]
        self.testM = self.miss_mat[idx[Train_No:],:]
        ##
        if numpy == True :
            self.trainX = self.trainX.values.astype(float)
            self.testX = self.testX.values.astype(float)
        else : pass
        ##
        if save == True :
            self.info["DL_trainX"] = self.trainX 
            self.info["DL_testX"] = self.testX 
            self.info["DL_trainM"] = self.trainM 
            self.info["DL_testM"] =  self.testM
        else : pass
        return self.trainX , self.testX , self.trainM , self.testM
        
    
    def train_test_split(self, 
                         train_prob , scale = True , 
                         onehot = True , dropna = True ,
                         save = False , numpy = True ) : 
        if dropna == True :
            Data = self.X.dropna(axis = 0)
        else :
            Data = self.X
        data_len , Dim = Data.shape
        idx = np.random.permutation(data_len)
        Train_No = int(data_len * train_prob)
        Test_No = data_len - Train_No
        if scale == True :
            Data[self.ord_col + self.num_col] = self.scaler.transform(Data[self.ord_col + self.num_col])
        else : pass
        
        if onehot == True :
            Data =self.category.transform(Data)
            self.trainX = Data.iloc[idx[:Train_No],:]
            self.testX = Data.iloc[idx[Train_No:],:]
            self.miss_mat = Data.isna().values
        else :
            self.trainX = Data.iloc[idx[:Train_No],:]
            self.testX = Data.iloc[idx[Train_No:],:]
            self.miss_mat = Data.isna().values
        self.trainM = self.miss_mat[idx[:Train_No],:]
        self.testM = self.miss_mat[idx[Train_No:],:]
        if numpy == True :
            self.trainX = self.trainX.values.astype(float)
            self.testX = self.testX.values.astype(float)
        else : pass
        if save == True :
            self.info["trainX"] = self.trainX 
            self.info["testX"] = self.testX 
            self.info["trainM"] = self.trainM 
            self.info["testM"] =  self.testM
        else : pass
        return self.trainX , self.testX , self.trainM , self.testM
    
    def summary(self,) :
        missing_sum = pd.DataFrame([np.sum(self.trainM , axis = 0) , 
                                    np.sum(self.testM, axis = 0)])
        m = pd.concat([missing_sum , pd.DataFrame(missing_sum.sum()).T]).T
        m.columns = ["train","test", "sum"]
        m["total"] = self.trainM.shape[0] + self.testM.shape[0]
        m["missing rate"] =np.round(m["sum"] / m["total"] * 100  , 2 )
        m["missing_train_mean"] = np.mean(self.trainM , axis = 0)
        m["missing_test_mean"] = np.mean(self.testM , axis = 0)
        m = m.T
        m.columns = self.info["columns"]
        return m 
    
    def save(self, file) :
        with open(file, 'wb') as f:
            dill.dump(self.info, f)

    def load(self, file) :
        with open(file, 'rb') as f:
            result = dill.load(f)
            
            
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
        rMiss = 1-Missing
        self.ori_missing = 1-ori_Missing
        self.ori_transform_missing = rMiss
        self.info["ori_missing_matrix"] = self.ori_missing
        self.info["ori_transform_missing_matrix"] = self.ori_transform_missing
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
        self.info["ori_missing_matrix"] = self.ori_missing
        return self.ori_missing


    def missing_matrix(self, data , re) :
        """
        return : Missing Matrix
        """
        if re == True : self.ori_missing = None
        else : pass
        
        if self.obj_col is None : 
            if self.ori_missing is None : miss_mat = self.num_(data)
            else : miss_mat = self.miss_mat
        else :
            if self.ori_missing is None :
                miss_mat = self.cat_num_(data)
            else :
                miss_mat = self.miss_mat
        self.miss_mat = miss_mat
        return miss_mat
    
    def information(self, ) :
        return self.info
        

