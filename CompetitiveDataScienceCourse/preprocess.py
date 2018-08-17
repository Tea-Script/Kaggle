import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
DATA_DIR = "../"

# TODO:
# read in csv file, remove headers and date, sort by date_block_num by item_id by store_id with custom sort criteria
# aggregate variable item_cnt_day in last column for each store in date_block_num as new row item_cnt_month
train_set = pd.read_csv(DATA_DIR + "sales_train.csv")  
train_set=train_set.iloc[: , 1:]
train_set = train_set.sort_values(by=["date_block_num", "shop_id","item_id"])
#train_set = train_set.agg(["sum"])
#print(train_set)


#test = pd.DataFrame({"year": [0,0,0,0,1,1,1,2,2,2,2,3,3,3,3], "month" : [0,0,0,0,1,1,1,2,2,2,3,3,3,4,4], "day" : [0,0,0,1,1,1,2,2,2,2,3,3,4,4,5], "count" : [7,4,3,2,1,5,4,2,3,2,5,3,2,1,3]})
#test = test[["year", "month", "day", "count"]]
#print(test)

def agg_multiple(df, labels, aggvar, repl=None):
    if(repl is None): repl = aggvar
    return df.groupby(labels)[(aggvar)].sum().to_frame(repl).reset_index()

#used for file creation
train_set = agg_multiple(train_set, ["date_block_num", "shop_id", "item_id"], "item_cnt_day", repl="item_cnt_mnth")
train_set[train_set["item_cnt_mnth"] > 20] = 20
train_set[train_set["item_cnt_mnth"] < 0] = 0
train_set.to_csv(DATA_DIR + "train_data.csv")
print("Training set created")
