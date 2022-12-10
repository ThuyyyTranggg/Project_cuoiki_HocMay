import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
one = np.ones((1, X.shape[1]))
print(X.shape[1])
print(one)
Xbar = np.concatenate((one, X), axis = 0) # each point is one row
y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
A = np.dot(Xbar, Xbar.T)
b = np.dot(Xbar, y)
w = np.dot(np.linalg.pinv(A), b)

w_0, w_1 = w[0], w[1]
print(w_0, w_1)
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0
print(y1)
# st.write(pd.DataFrame({
#     'Chiều cao':X,
#     'Cân nặng dự đoán': y1,
# }))
