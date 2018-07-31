import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

DATA_DIR = "~/.kaggle/competitions/competitive-data-science-final-project/"

# TODO:
# read in csv file, remove headers and date, sort by date_block_num by item_id by store_id with custom sort criteria
# aggregate variable item_cnt_day in last column for each store in date_block_num as new row item_cnt_month
#train_set = pd.read_csv(DATA_DIR + "sales_train.csv")  
#train_set=train_set.iloc[: , 1:]
#train_set = train_set.sort_values(by=["date_block_num", "shop_id","item_id"])
#train_set = train_set.agg(["sum"])
#print(train_set)


#test = pd.DataFrame({"year": [0,0,0,0,1,1,1,2,2,2,2,3,3,3,3], "month" : [0,0,0,0,1,1,1,2,2,2,3,3,3,4,4], "day" : [0,0,0,1,1,1,2,2,2,2,3,3,4,4,5], "count" : [7,4,3,2,1,5,4,2,3,2,5,3,2,1,3]})
#test = test[["year", "month", "day", "count"]]
#print(test)

def agg_multiple(df, labels, aggvar, repl=None):
    if(repl is None): repl = aggvar
    return df.groupby(labels)[(aggvar)].sum().to_frame(repl).reset_index()

#used for file creation
#train_set = agg_multiple(train_set, ["date_block_num", "shop_id", "item_id"], "item_cnt_day", repl="item_cnt_mnth")
#train_set[train_set["item_cnt_mnth"] > 20] = 20
#train_set.to_csv(DATA_DIR + "train_data.csv")

train_set = pd.read_csv(DATA_DIR + "train_data.csv")

train_set = train_set.sample(100000) #I don't have time to train on 3mil

X_ = np.array(train_set[["date_block_num", "shop_id","item_id"]])
y_ = np.array(train_set[["item_cnt_mnth"]])

tests = train_set.sample(5000)
X_test = np.array(tests[["date_block_num", "shop_id","item_id"]])
y_test = np.array(tests[["item_cnt_mnth"]])


#regr = SVR(C=1, epsilon = .1, gamma=.001, kernel="sigmoid")
#regr.fit(X_, y_.ravel())

regr = SVR(C=50, epsilon = .1, gamma=.001, kernel="rbf")
regr.fit(X_, y_.ravel())

preds = np.round(regr.predict(y_test))

true = np.sum((preds == y_test.ravel() )) / len(y_test) 
print("true accuracy is ", true)




pickle.dump(regr, open("finalized_model.sav", 'wb'))



file = "finalized_model.sav"
with open(file, "rb") as f:
    regr = pickle.load(f)

print("reading in test set")
data = pd.read_csv(DATA_DIR +"test.csv")
data["date_block_num"] = 34
X = np.array(data[["date_block_num", "shop_id","item_id"]])
print("creating predictions")
y_hat = np.round(regr.predict(X))
data["item_cnt_month"] = y_hat
print("generating submission file")
subm = data[["ID", "item_cnt_month"]]
subm.to_csv(DATA_DIR + "final.csv", index=False)
#print("Hyper Parameter Tuning")

#Par1 = {'kernel' : ['sigmoid', 'rbf'], 'C': [.001, .1, 100, 1000], 'epsilon':[.1], 'gamma': [.0001, .001, .01, .1]}

#clf1 = GridSearchCV(regr, Par1)
#clf1.fit(X, y.ravel())

#print(clf1.score(X_test, y_test.ravel()))
#print(clf1.best_estimator_)

#preds = np.round(clf1.predict(X_test))



