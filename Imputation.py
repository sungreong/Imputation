from autoimpute.imputations import SingleImputer, MultipleImputer
from autoimpute import imputations
import impyute as impy
import numpy as np
from missingpy import MissForest
from tqdm import tqdm , tqdm_notebook
import matplotlib.pyplot as plt
import pandas as pd , dill
import matplotlib , os
from time import time
from copy import deepcopy
from sklearn.model_selection import KFold
import seaborn as sns  , multiprocessing
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from renom.utility import completion
import re 
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier as GBM , GradientBoostingRegressor as GBR
from sklearn.metrics import f1_score , mean_squared_error as MSE , accuracy_score as acc
import category_encoders as CatEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def make_dir(directory ) :
    directory = "/".join(directory.split("/")[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

class Evaluate :
    """
    X         = data
    mask      = missing index
    algo      = [ "MissForest","mean", "median" , "knn", "MICE"  , "EM" , "MultipleImputer" , "mode" , "random"]
    kf        = split (cross validation)
    notobj    = not object columns
    obj       = object columns
    target    = target label
    miss_info = missing information 정보 모음
    """
    def __init__(self , T , mask , algo , miss_info , kf , notobj , obj , target ) :    
        try :
            self.miss_info = miss_info
            self.columns = notobj
            self.ord_num_col = self.miss_info["ord_col"] + self.miss_info["num_col"]
            metric = {"rmse" : {}, "nrmse" : {}}
            self.rawT = T 
            self.target = target
            if target is not None : self.target_y = T[target]
            else : self.target_y = None 
            self.cv = {}
            self.cv.update(deepcopy(metric))
            self.kf = kf 
            self.MSE = {}
            self.MSE.update(deepcopy(metric))
            self.result = {}
            self.time_ck = {}
            X = deepcopy(T)
            mask = pd.DataFrame(mask, columns = T.columns.tolist())
            self.rawmask = mask
            X[(mask == 1).values] = np.nan
            if obj in [None , []] : obj = None 
            else : pass 
            ##########################################
            self.X = X[notobj]
            self.T = T[notobj]
            self.mask = mask[notobj]
            self.notobj = notobj
            ##########################################
            if obj is not None : 
                ############ Numeric + Category  #################
                cat_impute = SimpleImputer(strategy="most_frequent")
                X[obj] = cat_impute.fit_transform( X[obj])
                self.true_obj = T[obj]
                self.pd_obj = X[obj]
                ###################################################
                TT = deepcopy(T)
                cat_encoder = miss_info["ce_encoder"]
                for k in cat_encoder.category_mapping :
                    col , map_ = k["col"] , k["mapping"]
                    TT[col] = TT[col].replace(dict(zip(k["mapping"].index , k["mapping"].values )))       
                self.full_miss_data = TT
                self.full_miss_data[(mask == 1).values] = np.nan
                mice_data = deepcopy(T)
                for obj_col in obj :
                    mice_data[obj_col] = "Cols_" + mice_data[obj_col] 
                self.full_mice_data = mice_data
                self.full_mice_data[(mask == 1).values] = np.nan
            else :
                ########## Numeric  ###############################
                num_data = deepcopy(self.X)
                num_data[(self.mask == 1).values] = np.nan
                self.full_miss_data = deepcopy(num_data)
                self.full_mice_data = deepcopy(num_data)
                ###################################################
            self.algo = algo
            self.method = {
                "MissForest" : lambda x : MissForest(verbose = 0, n_jobs  = -1 ).fit(x) ,
                "mean" : lambda x : impy.mean(x) , 
                "median" : lambda x : impy.median(x) , 
                "mode" : lambda x : impy.mode(x) , 
                "knn" : lambda x : impy.fast_knn(x) ,
                "MICE" : lambda x : impy.mice(x) ,
                "EM" : lambda x : impy.em(x),
                "MultipleImputer" : lambda x : MultipleImputer(n=1, return_list = True).\
                fit_transform(pd.DataFrame(x)).values,
            }
        except Exception as e :
            print(e)
            pass
        

    def Mice_Imputation(self, test_index, final ) :
        miss_info = self.miss_info
        obj_col = deepcopy(miss_info["obj_col"])
        if final :
            if obj_col == [] :
                self.numMICE = completion(self.full_mice_data.values , 
                                          mode="mice", impute_type="col")
                sample = self.numMICE
            else :
                MI = completion(self.full_mice_data.values , mode="mice", impute_type="col")
                MICE_pd= pd.DataFrame(MI, columns= miss_info["original_column"])
                self.numMICE = MICE_pd[self.notobj].values
                sample = MICE_pd.values
        else :
            if obj_col == [] :
                self.numMICE = completion(self.full_mice_data.iloc[test_index,:].values ,
                                          mode="mice", impute_type="col")
                sample = self.numMICE
            else :
                numMICE = completion(self.full_mice_data.iloc[test_index,:].values ,
                                        mode="mice", impute_type="col")
                MICE_pd= pd.DataFrame(numMICE, columns= miss_info["original_column"])
                self.numMICE = MICE_pd[self.notobj].values
                sample = MICE_pd.values
        return sample
        
    def Missforest_Imputation(self , train_index, test_index, final ) :
        miss_info = self.miss_info
        obj_col = deepcopy(miss_info["obj_col"])
        cat_var = [idx for idx , i in enumerate(miss_info["original_column"]) if i in obj_col]
        if final :
            if obj_col == [] :
                self.numMI = MissForest(max_depth=5).fit_transform(X = self.full_miss_data.values )
                sample = self.numMI
            else :
                MI = MissForest(verbose = 0, 
                                n_jobs  = -1 ,
                                max_depth=5).fit_transform(X = self.full_miss_data.values , 
                                                         cat_vars= cat_var)
                MI_pd= pd.DataFrame(MI, columns= miss_info["original_column"])
                self.MI_pd = MI_pd
                sample = self.MI_pd
        else :
            if obj_col == [] :
                MISS = MissForest(max_depth=5).\
                fit(X = self.full_miss_data.iloc[train_index,:].values)
                self.numMI = MISS.transform(X = self.full_miss_data.iloc[test_index,:].values )
                sample = self.numMI
            else :
                MIss = MissForest(verbose = 0, n_jobs  = -1 ,
                                  max_depth=5).\
                fit(X = self.full_miss_data.iloc[train_index,:].values , 
                                                   cat_vars= cat_var)
                MI = MIss.transform(self.full_miss_data.iloc[test_index,:].values)
                MI_pd= pd.DataFrame(MI, columns= miss_info["original_column"])
                self.numMI = MI_pd[self.notobj].values
                sample = MI_pd.values
        return sample
    
    def PFC(self ,  Real , Imputed , save_file_path )  :
        miss_info = self.miss_info
        obj_col = miss_info["obj_col"]
        ##
        ck = ~(Real[obj_col] == Imputed[obj_col])
        real_missing = miss_info["ori_missing_matrix"]
        missing_n = pd.DataFrame(real_missing, 
                                 columns = Real.columns.tolist())[obj_col].sum(axis = 0)
        ratio = ck.sum(axis = 0) / missing_n
        if self.target is not None :
            ratio = ratio.drop(index = [self.target])
        ratio.plot.barh(title= "Object Columns Missing Predict Ratio")
        png_file = "{}.png".format(save_file_path)
        plt.savefig(png_file)
        plt.show()
        result = pd.concat([ck.sum(axis = 0) , missing_n], axis = 1 ).\
        rename( columns = {0 :"N_not_correct" , 1 : "N_missing"})
        if self.target is not None :
            result = result.drop(index = [self.target])
        ## Proportion of Falsely Classified entries
        result["PFC"] = result["N_not_correct"] / result["N_missing"] 
        result.to_csv("{}.csv".format(save_file_path) , index = True)
        return result
        
    def objectType_comparision(self, name , save_file_path ) :
        if name in ["MICE" , "AutoEncoder" , "GAIN"] :
            pass
        else :
            print("MICE, AutoEncoder, GAIN 만 가능")
            raise
        miss_info = self.miss_info
        obj_col = miss_info["obj_col"]
        cat_encoder = miss_info["ce_encoder"]
        Real = deepcopy(miss_info["ori_raw_pd"])
        Imputed =  self.Outcome_Dict[name]
        Real = Real.reset_index(drop = True)
        result = self.PFC(Real, Imputed , save_file_path) 
        return result
        
        
    def Missforest_object_comparision(self  ,save_file_path) :
        miss_info = self.miss_info
        obj_col = miss_info["obj_col"]
        cat_encoder = miss_info["ce_encoder"]
        Real = deepcopy(miss_info["ori_raw_pd"])
        if obj_col == [] : return "object columns not exist"
        else :
            for k in cat_encoder.category_mapping :
                col , map_ = k["col"] , k["mapping"]
                if col == self.target :
                    continue
                self.Outcome_Dict["MissForest"][col] = self.Outcome_Dict["MissForest"][col].astype(str)
                try :
                    self.Outcome_Dict["MissForest"][col] = self.Outcome_Dict["MissForest"][col].\
                astype(float).replace(dict(zip(k["mapping"].values, k["mapping"].index )))       
                except :
                    pass
            obj_col = miss_info["obj_col"]
            Imputed = self.Outcome_Dict["MissForest"]
            Real = Real.reset_index(drop = True)
            result = self.PFC(Real, Imputed , save_file_path) 
            return result
            
    def MICE_object_comparison(self , save_file_path ) :
        miss_info = self.miss_info
        obj_col = miss_info["obj_col"]
        Real = deepcopy(miss_info["ori_raw_pd"])
        if obj_col == [] : return "object columns not exist"
        else :
            obj_col = miss_info["obj_col"]
            Imputed = self.Outcome_Dict["MICE"]
            Real = Real.reset_index(drop = True)
            result = self.PFC(Real, Imputed , save_file_path) 
            return result
        
    def object_most_commom(self , save_file_path) :
        most_common = deepcopy(self.pd_obj)
        miss_info = deepcopy(self.miss_info)
        obj_col = miss_info["obj_col"]
        Real = deepcopy(miss_info["ori_raw_pd"])
        if obj_col == [] : return "object columns not exist"
        else :
            obj_col = miss_info["obj_col"]
            Imputed = most_common
            Real = Real.reset_index(drop = True)
            result = self.PFC(Real, Imputed , save_file_path) 
            return result
        
        
    def select_algo(self,) :
        self.algo2 = {}
        for name in self.algo :
            self.algo2[name] = self.method[name]
    
    def NRMSE(self , true , imp ) :
        n = np.mean( np.square(imp - true) , axis = 0)
        d = np.mean(np.square( true - np.mean(true, axis =0 ) ) )
        return np.mean(n/d)
    
    def RMSE(self, true , imp ) :
        n = np.mean( np.square(imp - true ) , axis = 0)
        n1 = (n)**(0.5)
        return np.mean(n1)
        
    def metric(self , imputed ,test_index ,  standardize = True, metric_type = "rmse") :
        T = deepcopy(self.T)
        if standardize : T[self.ord_num_col] = self.miss_info["scaler"].transform(T[self.ord_num_col])
        else : pass
        if metric_type == "rmse" :    Metric = self.RMSE(T.iloc[test_index,:].values , imputed)
        elif metric_type == "nrmse" : Metric =self.NRMSE(T.iloc[test_index,:].values , imputed)
        return Metric
    
    def cv_metric(self , name ) :
        """
        name : 분모 type 결정
        1) minmax  2) quantile 3) root
        """
        def denum(name , true) :
            
            
            if name == "minmax" : 
                value = np.max(true , axis = 0 ) - np.min(true_cv , axis = 0 )
                value = np.where( value == 0.0 , value + 1e-5 , value)
            elif name == "quantile" :
                quantile = np.quantile(true , q = [0.1 , 0.9] , axis = 0)
                value = quantile[1] - quantile[0]
                value = np.where( value == 0.0 , value + 1e-1 , value)
            elif name == "root" :
                value = np.sqrt(np.mean(np.square( true - np.mean(true, axis =0 ) )  , axis = 0 ))
            elif name == "var" :
                value = np.mean(np.square( true - np.mean(true, axis =0 ) ) , axis = 0)
            elif name == "mean" :
                value = np.mean(true, axis =0 )
            elif name == "rmse" : 
                value = 1.0
            return value    
        
        metric_result = {}
        # cv_imputed
        for key , imputes in self.Outcome_Dict.items() : # cv_imputed
            cv_result = []
            for idx , test_index in enumerate(self.cv_test_index) :
                #test_index = self.cv_test_index[idx]
                true_cv    = self.T.iloc[test_index  ,][self.columns].values
                imputed_cv = imputes.iloc[test_index ,][self.columns].values
                numer = np.sqrt(np.mean((true_cv - imputed_cv)**2 , axis = 0))
                denom = denum(name =name , true = true_cv )
#                 if name in ["var", "root" ] :
#                     print("{} 분자 : {} 분모 : {}".format(key , numer , denom))
                result = np.mean(numer/ denom)
                cv_result.append(result)
            metric_result[key] = cv_result
        return metric_result
    
    def re_metric(self, mode) :
        self.Before_MSE = deepcopy(self.MSE)
        result = deepcopy(self.result)
        for name , imputed in result.items() :
            if type(imputed) == pd.core.frame.DataFrame : imputed = deepcopy(imputed[self.columns])
            else : imputed = pd.DataFrame(imputed, columns = self.columns )          
            if mode == "standardize" :
                standardize = True
                if self.standardize : pass
                else : imputed[self.ord_num_col] = self.miss_info["scaler"].transform(imputed[self.ord_num_col])
            elif mode == "inverse" :
                standardize = False
                if self.standardize : 
                    imputed[self.ord_num_col] = self.miss_info[f"scaler"].inverse_transform(imputed[self.ord_num_col])
                else : pass
            Metric = self.metric(imputed.values , standardize = standardize)
            self.MSE[name] = Metric

    def parallel(self,name, X , standardize , obj_col , 
                 train_index, test_index) :
        if name == "MissForest" : 
            imputed = self.Missforest_Imputation(train_index, test_index , final = False)
            numimputed = self.numMI
        elif name == "MICE"  :
            imputed = self.Mice_Imputation(test_index , final = False) 
            numimputed = self.numMICE
        else : 
            imputed = self.method[name](X.iloc[test_index,:].values)
            numimputed = deepcopy(imputed)
        
        
        NRMSE = self.metric(imputed = deepcopy(numimputed) ,
                             test_index = deepcopy(test_index) , 
                             standardize = standardize,
                             metric_type = "nrmse")
        
        RMSE = self.metric(imputed = deepcopy(numimputed) ,
                             test_index = deepcopy(test_index) , 
                             standardize = standardize,
                             metric_type = "rmse")
        
        print(name , imputed.shape)
        return imputed , RMSE , NRMSE
    
    def evaluate(self,standardize = False ) :
        self.standardize = standardize
        self.select_algo()
        ck_algo = deepcopy(self.algo2)
        algo = deepcopy(self.algo2)
        obj_col = self.miss_info["obj_col"]
        ck = list(self.MSE["nrmse"].keys())
        X = deepcopy(self.X)        
        for ai in ck_algo : 
            if ai in ck :
                print("{} : 이미 학습 완료".format(ai))
                del algo[ai]
            else : pass
            
        self.cv_test_index = []
        self.cv_imputed = {}
        for name in tqdm_notebook(algo) : 
            try :
                start = time()
                print("{} 진행중....".format(name))
                RMSE_cv = []
                NRMSE_cv = []
                cv_test_index = []
                cv_imputed_array = []
                cv_iter = 1
                lst = (Parallel(n_jobs=self.kf.n_splits ,
                                backend = "threading",
                                prefer="threads")
                       (delayed(self.parallel)
                        (name , X , standardize, obj_col , 
                         train_index, test_index) 
                        for train_index, test_index in self.kf.split(X)))
                imputed_cv , RMSE_cv, NRMSE_cv = [list(t) for t in zip(*lst)]
                cv_test_index = [test_index for _ , test_index in self.kf.split(X)]
                spend_time = time() - start
                print("{} | RMSE : {} | NRMSE : {} | {:.3f}초".\
                      format(name , np.mean(RMSE_cv) ,  np.mean(NRMSE_cv) , spend_time ))
                self.cv_imputed[name] = imputed_cv
                self.cv_test_index = deepcopy(cv_test_index)
                self.cv["rmse"][name] = deepcopy(RMSE_cv)
                self.cv["nrmse"][name] = deepcopy(NRMSE_cv)
                self.MSE["rmse"][name] = np.mean(RMSE_cv)
                self.MSE["nrmse"][name] = np.mean(NRMSE_cv)
                print(name , np.concatenate(imputed_cv).shape)
                self.result[name] = np.concatenate(imputed_cv)
                self.time_ck[name] = spend_time 
            except Exception as e :
                print(e)
                del self.algo2[name]
        return None
    
    def cv_boxplot(self, metric = "rmse",path = None ,  **kwargs) :
        try :
            cv = deepcopy(kwargs["cv"])
        except :
            cv  = deepcopy(self.cv[metric])
        cv_melting = pd.DataFrame(cv).melt()
        #sns.set(rc={'figure.figsize':(15.7,10.27)})
        fig , ax = plt.subplots(figsize =(15.7,10.27)) 
        plt.subplots_adjust(left = 0.18, bottom = 0.1, right = 0.95 , top = 0.95 , hspace = 0, wspace = 0)
        sns.boxplot(y="variable", x="value", data=cv_melting , showfliers=False)
        ax.set_ylabel("Methods", fontsize = 25)
        ax.set_title('Metric : {}'.format(metric) ,fontsize= 30) # title of plot
        if path is None :pass
        else : 
            make_dir(path)
            plt.savefig(path)
        plt.show()
    
    def cv_boxplot_all_run(self , 
                           methods = ["mean", "var", "root", "quantile", "minmax" , "rmse" ] ,
                           path = None ) :
        for method in methods :
            cv = self.cv_metric(name = method)
            if method != "rmse" :
                self.cv_boxplot( metric = "NRMSE ({})".format(method) , cv = cv ,
                                path = "{}/{}.png".format(path , method))
            else :
                self.cv_boxplot(metric = method , cv = cv ,
                                path = "{}/{}.png".format(path , method))
            
        
    
    def method_append(self, name , excute ) :
        imputed = excute
        Metric = self.metric(imputed)
        self.MSE[name] = Metric
        
    def append_algo(self , name , rmse,  nrmse , imputed) :
        self.cv["rmse"][name] = rmse
        self.cv["nrmse"][name] = nrmse
        self.MSE["rmse"][name] = np.mean(rmse) 
        self.MSE["nrmse"][name] = np.mean(nrmse) 
        cv_ = []
        for index in self.cv_test_index  :
            if type(imputed) == pd.DataFrame : result= imputed.iloc[index,:].values
            else : result = imputed.iloc[index,:]
            cv_.append(result)
        self.cv_imputed[name] = cv_
        imputed = deepcopy(imputed)
        self.result[name] = imputed
        return print("{} 업데이트".format(name))
    
    def save_result(self, filename) :
        Result = {}
        Result["MSE"] = self.MSE
        Result["RESULT"] = self.result
        try :
            Result["Method_Outcome"] = self.Method_Outcome
            Result["Method_Outcome_Pred"] = self.Method_Outcome_Pred
            Result["Method_Outcome_True"] = self.Method_Outcome_True
        except :
            pass
        Result["Outcome_Dict"] = self.Outcome_Dict
        Result["target_y"] = self.target_y
        Result["target"] = self.target
        Result["miss_info"] = self.miss_info
        Result["cv"] = self.cv
        try :
            Result["pd_obj"] = self.pd_obj
        except :
            pass
        
        with open(filename , "wb" ) as w :
            dill.dump(Result , w )
    
    def load_result(self, filename) :
        try :
            with open(filename , "rb" ) as r :
                Result = dill.load(r)
        except Exception as e :
            print(e)
            raise
        self.MSE = Result["MSE"]
        self.result = Result["RESULT"]
        try :
            self.Method_Outcome = Result["Method_Outcome"]
            self.Method_Outcome_Pred = Result["Method_Outcome_Pred"]
            self.Method_Outcome_True = Result["Method_Outcome_True"]
        except :
            pass
        self.Outcome_Dict = Result["Outcome_Dict"]
        self.target_y = Result["target_y"]
        self.target = Result["target"]
        self.miss_info = Result["miss_info"]
        self.cv= Result["cv"]
        try :
            self.pd_obj = Result["pd_obj"]
        except :
            pass
        print(">> Reload Result : \n")
    
    def plot(self, figsize = (9,9) ,metric="rmse" ,  img_name = None , title = None , **kwargs ) :
        try :
            RESULT = deepcopy(kwargs["MSE"])
        except :
            RESULT  = deepcopy(self.MSE[metric])
            
        fig, ax = plt.subplots(figsize = figsize)  
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        out = pd.DataFrame({"ALGO" : RESULT }).reset_index()
        plt.barh(range(len(out)), out["ALGO"], color=plt.cm.Paired(np.arange(len(out))))
        plt.yticks(range(len(out)) , tuple(out["index"]) )
        for i, v in enumerate(list( out["ALGO"].values )):
            ax.text(0, i-0.1, str(v), color='black', fontweight='bold')
        plt.title(title)
        if img_name is None :
            plt.show()
        else :
            plt.savefig(img_name)
            plt.show()
    ### 모델링 
    def parallel_cv(self, Method ,  train_index, test_index ) :
        try :
            X_imp = deepcopy(self.Outcome_Dict[Method])
            XX_imp = X_imp.drop([self.target], axis = 1)
            XXX_imp = deepcopy(XX_imp)
            #print(Method)
            #print("before : " , XX_imp.shape)
            #XXX_imp = CatEncoder.OneHotEncoder().fit_transform(XX_imp).values
            #print("after : " , XXX_imp.shape)
            if len(np.unique(self.target_y)) > 10 :
                y = self.target.values
            else :
                y = LabelEncoder().fit_transform(self.target_y).astype(int)
            lgb_train = lgb.Dataset(XXX_imp.iloc[train_index , : ]  , y[train_index])
            lgb_eval = lgb.Dataset(XXX_imp.iloc[test_index, :] , y[test_index] ,
                                   reference=lgb_train)
            if len(np.unique(y)) > 4 :
                params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'max_depth' : 7 ,
                'bagging_freq': 5,
                'verbose': -1,
                'nthread': 5,
                }
                lgbm = lgb.train(params, lgb_train , num_boost_round=20,
                                 valid_sets=lgb_eval,
                                 early_stopping_rounds=5,
                                )
                #Model = GBR(max_depth = 7).fit(X = XXX_imp[train_index , : ] , y= y[train_index] )
                y_pred = lgbm.predict(XXX_imp.iloc[test_index, :] ,
                                      num_iteration= lgbm.best_iteration )
                metric = MSE(y[test_index] , y_pred)
            else :
                params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'max_depth' : 7 ,
                'bagging_freq': 5,
                'verbose': -1 ,
                'nthread': 5,
                }
#                 lgbm = lgb.train(params, lgb_train , num_boost_round=20,
#                                  valid_sets=lgb_eval,
#                                  early_stopping_rounds=20,)
#                 y_pred = lgbm.predict(XXX_imp.iloc[test_index, :] , 
#                                       num_iteration= lgbm.best_iteration )
                lgbm = lgb.LGBMClassifier(**params )
                lgbm.fit( XXX_imp.iloc[train_index , : ]  , y[train_index]) 
                y_pred = lgbm.predict(XXX_imp.iloc[test_index , : ])
                metric = acc(y[test_index] , y_pred)
        except Exception as e :
            print("{}, Error : {}".format(Method , e))
            raise
        return metric , y[test_index] , y_pred
    
    def Predict_Comparision(self , ) :
        if self.target_y is None :
            print("Target이 없으므로 작동하지 않습니다.")
        else :
            Methods = list(self.Outcome_Dict.keys())
            kf = KFold(n_splits = 7) # Define the split - into 2 folds 
            Indexs = list(kf.split(self.result["mean"]))
            Method_Outcome = {}
            Method_Outcome_Pred = {}
            Method_Outcome_True = {}
            for Method in Methods :
                X_imp = deepcopy(self.Outcome_Dict[Method])
                cv_result = (Parallel(n_jobs= kf.n_splits ,backend = "threading",prefer="threads")
                                   (delayed(self.parallel_cv)
                                    (Method , train_index, test_index) 
                                    for train_index, test_index in Indexs))
                Metrics , y_true, y_pred = [list(t) for t in zip(*cv_result)]
                cv_SCORE = np.mean(list(Metrics))
                print("{:11}, score : {:7.5f}".format(Method , cv_SCORE))
                Method_Outcome[Method] = list(Metrics)
                Method_Outcome_Pred[Method] = list(y_pred)
                Method_Outcome_True[Method] = list(y_true)
            self.Method_Outcome = Method_Outcome
            self.Method_Outcome_Pred = Method_Outcome_Pred
            self.Method_Outcome_True = Method_Outcome_True
        return None
    
    def Outcome(self, ) :
        Result = deepcopy(self.result)
        Methods = list(Result.keys())
        total_col = self.miss_info["original_column"]
        objcols = self.miss_info["obj_col"]
        self.objcols = objcols
        for Method in Methods :
            if objcols != [] :
                Imputed = Result[Method]
                if len(total_col) != Imputed.shape[1] :
                    print(" 가공 {} Category Most Common Imputation".format(Method))
                    Imputed = pd.DataFrame(Imputed, columns = self.notobj)
                    Imputed = pd.concat([Imputed , self.pd_obj ] , axis = 1)
                    for obj in self.miss_info["obj_col"] :
                        Imputed[obj] =  Imputed[obj].apply(lambda x : re.sub("Cols_","", x))
                    Imputed[objcols] = Imputed[objcols].astype('category')
                    if self.target is not None :
                        Imputed[self.target] = self.target_y.values
                    Result[Method] = Imputed[total_col]
                else :
                    print(" 가공 {}".format(Method))
                    Imputed = pd.DataFrame(Imputed, columns = total_col)
                    Imputed[objcols] = Imputed[objcols].astype(str)
                    
                    Imputed[self.notobj] = Imputed[self.notobj].astype(float)
                    for obj in objcols :
                        Imputed[obj] =  Imputed[obj].apply(lambda x : re.sub("Cols_","", x))
                    Imputed[objcols] = Imputed[objcols].astype('category')
                    if self.target is not None :
                        Imputed[self.target] = self.target_y.values
                    Result[Method] = Imputed
            else :
                Imputed = pd.DataFrame(Result[Method], columns = self.notobj)
                Result[Method] = Imputed
        self.Outcome_Dict = Result
        return None    
    
    def Measure_Vis(self , metric ) :
        result = {}
        for pred, true in zip(self.Method_Outcome_Pred.items() , 
                              self.Method_Outcome_True.items()) :
            method , pred2 = pred
            method , true2 = true    
            store2 = []
            for pred3 , true3 in zip(pred2 , true2) :
                store2.append( metric(true3 , pred3) )
            result[method] = store2        
        cv_melting = pd.DataFrame(result).melt()
        cv_mean  = cv_melting.groupby(["variable"])["value"].mean()
        pos = range(len(cv_mean))
        cv_mean_label = [str(np.round(s, 2)) for s in cv_mean.values] 
        sns.set(rc={'figure.figsize':(15.7,10.27) }) 
        sns.set_context("paper", 
                        rc={"font.size":20,"axes.titlesize":30,"axes.labelsize":20},
                        font_scale = 2.0)  
        ax = sns.boxplot(y="variable", x="value", data=cv_melting , showfliers=False)
        ax.set(xlabel='Score', ylabel='Method',
               title = "CrossValidation BoxPlot [ {} ]".format(metric.__name__))
        list_ = list(ax.get_yticklabels())
        xticklables_ = []
        for tick , label in enumerate(list_) :
            method = label.get_text()
            value  = cv_mean[method]
            xticklables_.append("{:12}({:5.2f})".format(method , value*100,3))
        ax.set_yticklabels(xticklables_, rotation = 0)
        plt.show()
        
    
    def Modeling_Measure_Vis(self, path = None ) :
        cv_melting = pd.DataFrame(self.Method_Outcome).melt()
        cv_mean  = cv_melting.groupby(["variable"])["value"].mean()
        pos = range(len(cv_mean))
        cv_mean_label = [str(np.round(s, 2)) for s in cv_mean.values]
        
        #sns.set(rc={'figure.figsize':(15.7,10.27) }) 
        fig , ax = plt.subplots(figsize = (15.7,10.27) )
        plt.subplots_adjust(left = 0.21, bottom = 0.1, right = 0.95 , top = 0.95 , hspace = 0, wspace = 0)
        sns.set_context("paper", 
                        rc={"font.size":20,"axes.titlesize":30,"axes.labelsize":20},
                        font_scale = 2.0)  
        ax = sns.boxplot(y="variable", x="value", data=cv_melting , showfliers=False)
        ax.set(xlabel='Score', ylabel='Method',
               title = "Model CV Score Comparision")
        list_ = list(ax.get_yticklabels())
        xticklables_ = []
        for tick , label in enumerate(list_) :
            method = label.get_text()
            value  = cv_mean[method]
            xticklables_.append("{:12}({:5.2f})".format(method , value*100,3))
        ax.set_yticklabels(xticklables_, rotation = 0)
        if path is None : pass
        else : 
            make_dir(path)
            plt.savefig(path) 
        plt.show()
    
    def show_result(self, row_m = 3 , n = 2 ) :
        obj_col = self.miss_info["obj_col"]
        ord_num_col = self.miss_info["ord_col"] + self.miss_info["num_col"]
        store = pd.DataFrame()
        col_index = []
        miss_idx =\
        [idx for idx , i in enumerate((self.rawmask.sum(axis = 1 )>row_m).values.tolist()) if i==True ]
        try :
            select_idx = np.random.choice(miss_idx , n , replace = False)
        except IndexError as e :
            select_idx = miss_idx
        result = deepcopy(self.Outcome_Dict)
        masking = self.rawmask.iloc[select_idx,:]
        for key , imputed in result.items() :
            try :
                if type(imputed) == pd.core.frame.DataFrame : aa = deepcopy(imputed)
                else :
                    aa = pd.DataFrame(imputed,columns = self.miss_info["original_column"] )                          
                    if self.standardize :
                        aa[ord_num_col] =self.miss_info["scaler"].inverse_transform(aa[ord_num_col])
                        aa = pd.DataFrame(aa)
                if obj_col == [] : pass
                else : aa[obj_col] = aa[obj_col].astype(str)
                store = pd.concat([store , aa.iloc[select_idx,:] *masking.astype(int)  ])
                index = [key] * n
                col_index.extend(index)    
            except Exception as e :
                print("{} , Error {}".format(key , e))
        true = deepcopy(self.rawT)
        store = pd.concat([masking , true.iloc[select_idx,:]*masking.astype(int) , store]).reset_index(drop=True)
        df2 = store.reset_index(drop=True)
        
        algo = pd.DataFrame(["missing"] * n + ["True"] * n + col_index, 
                            columns =["algo"])
        df2 = pd.concat([algo , df2] , axis = 1)
        # colorlist
        # https://digitalsynopsis.com/design/beautiful-color-palettes-combinations-schemes/
        color = ["#fe4a49" , "#2ab7ca" , "#fed766" , "#e6e6ea" , 
                 "#ead5dc" , "#eec9d2" , "#f4b6c2" , "#f6abb6", 
                 "#4a4e4d" , "#0e9aa7" , "#3da4ab" , "#f6cd61" , "#fe8a71" ,
                 "#fe9c8f","#feb2a8","#fec8c1","#fad9c1","#f9caa7"
                 "#ee4035","#f37736","#fdf498","#7bc043","#0392cf"
                 "#96ceb4","#ff6f69","#ffcc5c","#88d8b0"] 
        colorlist = np.random.choice(color , 
                                     len(self.Outcome_Dict.keys()) , 
                                     replace = False)
        self.colorlist = dict( zip( list( self.Outcome_Dict.keys() ) , colorlist) ) 
        df3  = df2.style.apply(self.row_highlight , axis =1 ).\
        applymap(self.color_negative_red , 
                 subset =pd.IndexSlice[0:(len(select_idx)-1), : ] ).\
        applymap(self.algo_color , 
                 subset =pd.IndexSlice[:, ["algo"] ] ).\
        applymap(self.font_style).bar(subset=pd.IndexSlice[0:(len(select_idx)-1),
                                                           self.miss_info["original_column"]  ], 
                                      color='#ffeead')
        return df3
    
    
    def font_style(self, val):
        if type(val) == str :
            color = 'font-weight: bold;' # font-size:larger
        else :
            color = 'font-weight: normal'
        return color 
    
    def row_highlight(self , s):
        if s.algo == "True" :
            return ['background-color: yellow']* ( len(self.miss_info["original_column"]) + 1 )
        else:
            return ['background-color: white']* ( len(self.miss_info["original_column"]) + 1 )
    
    def color_negative_red(self , val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
        if val == 1  :
            color = 'font-weight: bold;color:red'
        elif val == 0 :
            color = "color:white"
        else :
            color = "color:blue"    
        return color
    
    def algo_color(self , val ) :
        if val == "True" :
            color = 'background-color: yellow;color:red' 
        elif val == "missing" :
            color = 'background-color: palegreen' 
        elif val in ["MICE", "mean", "median" , "MissForest" , "knn", "mode" , "EM" , "GAIN" , "AutoEncoder" ] :
            color = 'background-color: {}'.format(self.colorlist[val])
        else :
            color = 'background-color: white' 
        return color
        
        
    
            

        