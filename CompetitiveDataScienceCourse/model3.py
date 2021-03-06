import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import xgboost as xgb
from sklearn.metrics import accuracy_score
DATA_DIR = "../"


dataset = pd.read_csv(DATA_DIR + "train_data.csv")

train_set = dataset.sample(frac=.001) #I don't have time to train on 3mil
X_ = np.array(train_set[["date_block_num", "shop_id","item_id"]])
y_ = np.array(train_set[["item_cnt_mnth"]])

tests = dataset.sample(5000)
X_test = np.array(tests[["date_block_num", "shop_id","item_id"]])
y_test = np.array(tests[["item_cnt_mnth"]])


#regr = SVR(C=1, epsilon = .1, gamma=.001, kernel="sigmoid")
#regr.fit(X_, y_.ravel())

regr = xgb.XGBRegressor()
#regr.fit(X_,y_)

Params = {"learning_rate": [.001, .01], "colsample_bytree" : [.2, .4, .7], "subsample": [.4, .6,.8, .9], "n_estimators" : [300,500, 700, 1000],
          "max_depth" : [20, 100, 150, 200], "objective": ["reg:tweedie", "survival:cox"]

}

clf = GridSearchCV(regr, Params)
clf.fit(X_, y_.ravel())

print(clf.score(X_test, y_test.ravel()))
print(clf.best_params_)

train_preds = np.round(clf.predict(X_))
test_preds = np.round(clf.predict(X_test))

train_acc = accuracy_score(train_preds, y_.ravel())
print("train accuracy is ", train_acc)

test_acc = accuracy_score(test_preds, y_test.ravel())
print("test accuracy is ", test_acc)



pickle.dump(clf.best_estimator_, open("finalized_model3.sav", 'wb'))



file = "finalized_model3.sav"
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
subm.to_csv(DATA_DIR + "final3.csv", index=False)
#print("Hyper Parameter Tuning")




