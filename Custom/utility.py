from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from sklearn.metrics import roc_curve

print(
    """
    사용하면 좋은 패키지
    st / train_validation_test_split / cat2dict / code2cat / colorlist / onehotencoding / roccurve  / ProbDensity
    pycm import *  
    ConfusionMatrix(actual_vector=``test_y``, predict_vector= ``pred`` ) 
    사용하면 통계량값 다 얻을 수 있음
    """
)

def st(obj):
    if isinstance(obj, (tuple, list, set, dict)):
        print(type(obj))
        print(obj)
        return
    elif isinstance(obj, pd.DataFrame):
        print("%s : dimension of %s" % (type(obj), obj.shape))

        temp = list(obj.index.values[:10])
        temp = ', '.join(list(map(str, temp)) )
        print("Index: %s ...  : %s \n" % (temp, obj.index.dtype))

        if obj.shape[0] > 100: obj_short = obj.iloc[:100, :]

        res = pd.concat([obj_short.dtypes, obj_short.apply(lambda x: [x.unique()]) ], axis=1)
        print(pd.DataFrame.to_string(res, header=False))
        return
    try:
        print(type(obj))
        print(obj)
    except:
        return "Something's broken!"
    

def printProgress(iteration, total, 
                  prefix = '', suffix = '', 
                  decimals = 4, barLength = 50): 
    formatStr = "{0:." + str(decimals) + "f}" 
    percent = formatStr.format(100 * (iteration / float(total))) 
    filledLength = int(round(barLength * iteration / float(total))) 
    bar = '#' * filledLength + '-' * (barLength - filledLength) 
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)), 
    if iteration == total: 
        sys.stdout.write('\n') 
    sys.stdout.flush()
        
def ProbDensity(Output , rainidx , subplot, title) :
    """
    Output = predict_proba()
    f, ax = plt.subplots( 2,  2 , figsize=(18, 7))
    axx = ax.flatten()
    """
    sns.distplot(Output[rainidx , 1], label = "rain" , ax = subplot) 
    sns.distplot(Output[~rainidx , 1], label = "not rain" , ax = subplot ).set_title( title , fontsize = 15) 
    subplot.legend()

def rocvis(true , prob , label ) :
    from sklearn.metrics import roc_curve
    if type(true[0]) == str :
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true = le.fit_transform(true)
    else :
        pass
    fpr, tpr, thresholds = roc_curve(true, prob)
    plt.plot(fpr, tpr, marker='.', label = label  )
    

def train_validation_test_split(
    X, y, train_size=0.8, val_size=0.1, test_size=0.1, 
    random_state=None, shuffle=True):
    """
    X,y 는 Numpy 형태로 해야한다.
    """
    assert type(X) == np.ndarray and type(y) == np.ndarray
    assert int(train_size + val_size + test_size + 1e-7) == 1
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,    test_size=val_size/(train_size+val_size), 
        random_state=random_state, shuffle=shuffle)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def cat2dict(data) :
    assert type(data) == pd.core.frame.DataFrame
    try :
        object_names = data.select_dtypes("category").columns
    except Exception as e :
        print("Category 변수 넣어주세요")
        print(e)
    value_to_code = {}
    code_to_value = {}
    for object_name in object_names:
        col = data[object_name].dropna()
        keys = np.sort(col.cat.codes.unique()).astype(float)
        values = np.sort(col.unique()).astype(str)
        keys = np.append(np.nan , keys)
        values = np.append(str(-1) , values)
        dic = dict(zip(values, keys))
        value_to_code[object_name] = dic
        dic = dict(zip(keys, values))
        code_to_value[object_name] = dic
        
    return value_to_code , code_to_value

"""
class로 multiprocessing으로 replace 
그냥 loop/ map이 더 빠르다..머지....
"""

def code2cat(data , dictionary ) :
    #col = data.columns.tolist()
    col = list(dictionary.keys())
    start = time.time()
    for i in col :
        data[i] = data[i].map(dictionary.get(i))
    print("소요시간(초) : " , time.time() - start )
    return data
        
def cat2code(data , na_show = False ) :
    col = data.columns.tolist()
    start = time.time()
    categorycol = data.select_dtypes("category").columns.tolist()
    data[categorycol]  = data.select_dtypes("category").apply(lambda x: x.cat.codes)
    if na_show :
        pass
    else :
        minus2na = {}
        for i in categorycol :
            dic = {-1 : np.nan}
            minus2na[i] = dic
        data[categorycol]  = data[categorycol].replace(minus2na)
    print("소요시간(초) : " , time.time() - start )
    return data

def colorlist(size) :
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)# Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    color = np.random.choice(sorted_names, size, replace = False)
    return color

def onehotencoding(df, columns ) :
    ## onehot을 할 컬럼만 빼서 onehot 시키기
    category = df[columns]
    df.drop(columns , inplace = True , axis = 1)
    category = category.astype(str)
    onehot = pd.concat([pd.get_dummies(category[col] , prefix = str(col), dummy_na =False ) for col in category ], axis=1)
    #Output = pd.concat([onehot , df ], axis = 1)
    return onehot

"""
고생해서 만들었지만 쓸모없는 걸로....ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ
"""
class multiprocessing_replace_value :
    def __init__(self , num_partitions ,  num_cores , dictionary , data , func) :
        self.num_partitions = num_partitions
        self.num_cores      = num_cores
        self.dictionary     = dictionary
        self.data           = data
        self.func           = func
        
    def value_to_coding(self , column ):
        self.data[column] = self.data[column].map(self.dictionary.get(column))
        return self.data[column]
        #df_split = np.array_split(data , self.num_partitions)    

    def parallelize_dataframe_v2(self ) :
        df_split = np.array_split(self.data , self.num_partitions)
        pool = Pool(self.num_cores)
        Output = pd.concat(pool.map(self.func , df_split))
        pool.close()
        pool.join()
        return Output
    
    ## better 
    def run_v2(self ) :
        print("Partion : {} , Cores : {}".format(self.num_partitions , self.num_cores )) 
        start = time.time()
        df    = self.parallelize_dataframe_v2()
        print("소요시간(초) : " , time.time() - start )
        return df

    
    def parallelize_dataframe(self):
        col = self.data.columns.tolist()
        pool = Pool(self.num_cores)
        Output = pool.map(self.value_to_coding , col)
        pool.close()
        pool.join()
        return Output

    def run(self) :
        print("Partion : {} , Cores : {}".format(self.num_partitions , self.num_cores )) 
        start = time.time()
        df    = self.parallelize_dataframe()
        print("소요시간(초) : " , time.time() - start )
        return pd.concat(df, axis = 1)
