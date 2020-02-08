import pandas as pd
import category_encoders as ce
import re, numpy as np
from copy import deepcopy
from sklearn.utils.class_weight import compute_class_weight

table = pd.DataFrame


def splitData(input: table, sep: list, key: str) -> (table, table):
    tr_data = input[input[key] == sep[0]]
    tr_data.reset_index(drop=True, inplace=True)
    te_data = input[input[key] == sep[1]]
    te_data.reset_index(drop=True)
    print(f"Shape Train : {tr_data.shape} / Valid : {te_data.shape}")
    return tr_data, te_data


class DataHadling:
    def __init__(self, in_var: list, num_var: list, NAColDrop: bool = True):
        self.fac_var = sorted(list(set(in_var).difference(set(num_var))))
        self.in_var = sorted(in_var)
        self.num_var = num_var
        self.ce = ce.OneHotEncoder(cols=self.fac_var,
                                   verbose=0,
                                   drop_invariant=NAColDrop,
                                   handle_unknown="value",
                                   handle_missing="return_nan",
                                   use_cat_names=True)

    def start(self, input: table) -> table:
        self.ForComp = input[self.in_var]
        self.ForComp_missing_matrix = self.ForComp.isna().values * 1
        totalcol = input.columns.tolist()
        Remaincol = list(set(totalcol).difference(set(self.in_var)))
        RemainTable = input[Remaincol]
        table = input.drop(Remaincol, axis=1)
        train_result = self.ce.fit_transform(input[self.in_var])
        train_result = train_result[np.sort(self.ce.feature_names)]
        self.train_onehot_col = train_result.columns.tolist()
        self.train_result = pd.concat([train_result, RemainTable], axis=1)
        return None

    def evaluate(self, input: table) -> table:
        totalcol = input.columns.tolist()
        Remaincol = list(set(totalcol).difference(set(self.in_var)))
        RemainTable = input[Remaincol]
        table = input.drop(Remaincol, axis=1)
        valid_result = self.ce.transform(input[self.in_var])
        valid_result = valid_result[np.sort(self.ce.feature_names)]
        self.valid_onehot_col = valid_result.columns.tolist()
        self.valid_result = pd.concat([valid_result, RemainTable], axis=1)
        return None

    def setting(self, ):
        self.only_tr = list(set(self.train_onehot_col).difference(set(self.valid_onehot_col)))
        self.only_va = list(set(self.valid_onehot_col).difference(set(self.train_onehot_col)))
        if (self.only_tr == []) & (self.only_va == []):
            print("이상없음")
        if len(self.only_va) > 0:
            for fac_col in self.only_va:
                original_var = None
                for ori_col in self.fac_var:
                    if re.search(f"^{ori_col}", fac_col) is not None:
                        original_var = ori_col
                        break
                self.valid_result[original_var + "_|zbin"] = \
                    self.valid_result[fac_col] + self.valid_result[original_var + "_|zbin"]
        if len(self.only_tr) > 0:
            for fac_col in self.only_tr:
                self.valid_result[fac_col] = 0
        return None

    def select(self, input: table, ) -> table:
        return input[self.train_onehot_col]

    def numscaler(self, input: table, scaler, eval: bool = True) -> table:
        if eval == False:
            self.scaler = scaler
            input[self.num_var] = self.scaler.fit_transform(input[self.num_var])
            self.ForComp[self.num_var] = self.scaler.transform(self.ForComp[self.num_var])
        else:
            input[self.num_var] = self.scaler.transform(input[self.num_var])
        return input

    def missing_matrix(self, input: table) -> np.array:
        return input.isna().values * 1

    def class_weight(self, onehotset, balanced):
        array = onehotset.values
        y = np.argmax(array, axis=1)
        classes = np.unique(y)
        output = compute_class_weight(class_weight=balanced, classes=classes, y=y)
        # output = np.append(output,1)
        return output

    def WeightInfo(self, data: table, balanced: str = "balanced") -> list:
        store = self.store
        weight_info = []
        for idx, i in enumerate(store):
            start, end = i[0], i[1]
            diff = end - start
            if diff == 1:
                weight_info.append([1])
            else:
                onehotset = data.iloc[:, i[0]: i[1]]
                weight_info.append(self.class_weight(onehotset, balanced))
        self.weight_info = weight_info
        return None

    def FindColumnsIndex(self, ) -> (dict, list):
        in_var = self.in_var
        num_var = self.num_var
        one_hot_var = self.train_onehot_col
        start_idx = 0
        key_store = {}
        store = []
        for idx, col in enumerate(in_var):
            if col in num_var:
                aa = [start_idx, start_idx + 1]
                store.append(aa)
                start_idx += 1
            else:
                find = (
                    [idx2 for idx2, ck in enumerate(one_hot_var)
                     if re.search(f"^{col}_", ck)]
                )
                nn = len(find)
                aa = [start_idx, start_idx + nn]
                store.append(aa)
                start_idx += nn
            key_store[col] = aa
        self.key_store, self.store = key_store, store
        return None


def calculate_rmse_pfc(not_missing_data, imputed_data, missing_matrix,
                       in_var, num_var, fac_var):
    results = []
    for idx, col in enumerate(in_var):
        if col in num_var:
            only_imputed = imputed_data.loc[missing_matrix[:, idx], col].values
            only_real = not_missing_data.loc[missing_matrix[:, idx], col].values
            try:
                result = np.mean(np.square(only_real - only_imputed))
                a = np.sqrt(np.mean(np.square(only_real - only_imputed)))
            except Exception as e:
                a = 0
                pass
        else:
            only_imputed = imputed_data.loc[missing_matrix[:, idx], col].values
            only_real = not_missing_data.loc[missing_matrix[:, idx], col].values
            a = np.sum(only_real != only_imputed) / len(only_real)
        #             print(f"PFC Loss : {col}, {a}")
        if np.isnan(a):
            a = 0
        else:
            pass
        results.append(a)
    pfcs = results[len(num_var):]
    rmses = results[:len(num_var)]
    return sum(pfcs), sum(rmses)
