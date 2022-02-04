from ntpath import realpath
import os
import pandas as pd
import numpy as np
import string

def leastSquareMultipleMethod(X, Y):
    X_T = X.transpose()
    B = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_T, X)), X_T), Y)
    return B

def residualSquared(X, Y, B):
    Y_pred = np.matmul(X, B)
    u = sum(np.power((Y - Y_pred), 2))
    v = sum(np.power((Y - np.mean(Y)), 2))
    R = 1 - u/v
    return R

def realpath_folder():
    curfile = os.path.realpath(__file__)
    index = curfile.rfind("\\")
    curfolder = curfile[0:index]
    return curfolder

# normalisation by standard deviation?
def normalize_by_std(df):
    for col in df.columns:
        if col == 'price':
            continue
        scale_factor = df[col].std()
        df[col] -= df[col].mean()
        df[col] /= scale_factor
    return df

def averageAbsoluteLoss(X, Y, B):
    Y_pred = np.matmul(X, B)
    return np.mean(abs(Y_pred - Y))

dataCSV = pd.read_csv(realpath_folder()+"\\"+"data.csv")
dataDF = pd.DataFrame(dataCSV)
dataDF = dataDF.select_dtypes(include=np.number)
correlations = dataDF.corr('pearson')

cols_to_remove = list()
for col in correlations.columns:
    if abs(correlations.loc['price',col]) < 0.1:
        cols_to_remove.append(col)

dataDF = dataDF.drop(columns = cols_to_remove)
dataDF = dataDF.drop('view', 1)
dataDF = normalize_by_std(dataDF)

# print(dataDF)

training_coeff = 0.8

num_rows = dataDF.shape[0]
num_coll = dataDF.shape[1]
num_rows_train = int(num_rows * training_coeff)


dataDF_train = dataDF.iloc[:num_rows_train]
dataDF_test = dataDF.iloc[num_rows_train::]

x_train = dataDF_train.drop(columns = 'price')
y_train = dataDF_train['price']

x_test = dataDF_test.drop(columns = 'price')
y_test = dataDF_test['price']


x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()
x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()

B = leastSquareMultipleMethod(x_train_np, y_train_np)
R = residualSquared(x_test_np, y_test_np, B)

AvgLoss = averageAbsoluteLoss(x_test_np, y_test_np, B)

print("R: ", R)
print("AvgLoss: ", AvgLoss)
print("AvgPrice: ", np.mean(dataDF['price']))