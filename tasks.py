from dis import dis
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def compute_metrics(y_pred, y_test):
    y_pred[y_pred<0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    return y_pred

def kf_predict(X, Y):

    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def predict_regression(embs, labels, display=False):
    y_pred, y_test = kf_predict(embs, labels)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    if display:
        print("MAE: ", mae)
        print("RMSE: ", rmse)
        print("R2: ", r2)
    return mae, rmse, r2


def lu_classify(emb, display=False):
    lu_label_filename = "./Data/mh_cd.json"
    cd = json.load(open(lu_label_filename))
    cd_labels = np.zeros((180))
    for i in range(180):
        cd_labels[i] = cd[str(i)]

    n = 12
    kmeans = KMeans(n_clusters=n, random_state=3)
    emb_labels = kmeans.fit_predict(emb)

    nmi = normalized_mutual_info_score(cd_labels, emb_labels)
    ars = adjusted_rand_score(cd_labels, emb_labels)
    if display:
        print("emb nmi: {:.3f}".format(nmi))
        print("emb ars: {:.3f}".format(ars))
    return nmi, ars

def do_tasks(embs, ):
    display=True
    print("Crime Count Prediction: ")
    crime_count_label = np.load("./Data/crime_counts_label.npy")
    crime_count_label = crime_count_label[:, 0]
    mae, rmse, r2 = predict_regression(embs, crime_count_label, display=display)

    print("Check-in Prediction: ")
    check_in_label = np.load("./Data/check_in_label.npy")
    mae, rmse, r2 = predict_regression(embs, check_in_label, display=display)

    print("Land Usage Prediction: ")
    lu_classify(embs, display=display)

