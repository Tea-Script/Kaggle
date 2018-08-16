import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

DATA_DIR = "../"

dataset = pd.read_csv(DATA_DIR + "train_data.csv")
train_set = dataset.sample(frac=1) #I don't have time to train on 3mil

X_ = np.array(train_set[["date_block_num", "shop_id","item_id"]])
y_ = np.array(train_set[["item_cnt_mnth"]])

tests = dataset.sample(5000)

#regr = SVR(C=1, epsilon = .1, gamma=.001, kernel="sigmoid")
#regr.fit(X_, y_.ravel())
print("Beginning Fitting")
regr = SVR(C=50, epsilon = .1, gamma=.001, kernel="rbf")
regr.fit(X_, y_.ravel())
print("Creating Predictions")

train_preds = np.round(regr.predict(X_))
test_preds = np.round(regr.predict(X_test))

train_acc = accuracy_score(train_preds, y_.ravel())
print("train accuracy is ", train_acc)

test_acc = accuracy_score(test_preds, y_test.ravel())
print("test accuracy is ", test_acc)



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




