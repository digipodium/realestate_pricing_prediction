import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ml related
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import warnings
from scipy import stats
from scipy.stats import norm, skew

warnings.filterwarnings('ignore')

train_path = "datasets/train.csv"
test_path = "datasets/test.csv"

@st.cache
def load_train_data(train_path):
    return pd.read_csv(train_path)

@st.cache
def load_test_data(test_path):
    return pd.read_csv(test_path)


st.title("REAL ESTATE PRICING PREDICTION")

train_data = load_train_data(train_path)
test_data = load_test_data(test_path)

if st.checkbox("view dataset colum description"):
    st.subheader('displaying the column wise stats for the dataset')
    st.write(train_data.columns)
    st.write(train_data.describe())

st.subheader('Correlation b/w dataset columns')
corrmatrix = train_data.corr()
f,ax = plt.subplots(figsize=(20,9))
sns.heatmap(corrmatrix,vmax = .8, annot=True)
st.pyplot()

st.subheader("most correlated features")
top_corr = train_data.corr()
top_corr_feat = corrmatrix.index[abs(corrmatrix['SalePrice'])>.5]
plt.figure(figsize=(10,10))
sns.heatmap(train_data[top_corr_feat].corr(), annot=True, cmap="RdYlGn")
st.pyplot()

st.subheader("Comparing Overall Quality vs Sale Price")
sns.barplot(train_data.OverallQual, train_data.SalePrice)
st.pyplot()

st.subheader("Pairplot visualization to describe correlation easily")
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], size=2.5)
st.pyplot()

st.subheader("Analyis of Sale Price column in dataset")
sns.distplot(train_data['SalePrice'] , fit=norm);# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_data['SalePrice'])
st.write( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
st.pyplot()

fig = plt.figure(figsize=(10,10))
res = stats.probplot(train_data['SalePrice'], plot=plt,)
st.pyplot()

