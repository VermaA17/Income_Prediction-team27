#Importing Libraries:

import pandas as pd
import numpy as np
from missingpy import KNNImputer
from catboost import CatBoostRegressor
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
#import xgboost as xgb
import category_encoders as ce
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer


#Loading Files:
df=pd.read_csv("C:/Users/ADMIN/PycharmProjects/tcd-ml-1920-group-income-train.csv",na_values =['#N/A','nA','#NUM!'],encoding = "ISO-8859-1",low_memory=False)
df5=pd.read_csv("C:/Users/ADMIN/PycharmProjects/tcd-ml-1920-group-income-test.csv",na_values =['#N/A','nA','#NUM!'],encoding = "ISO-8859-1",low_memory=False)




#Dropping the irrelevant columns:
df.drop(['Instance','Wears Glasses','Hair Color','Satisfation with employer','Body Height [cm]'],axis=1,inplace=True)
df5.drop(['Instance','Wears Glasses','Hair Color','Satisfation with employer','Body Height [cm]'],axis=1,inplace=True)

#Imputation:
df['Gender']=df['Gender'].replace('f','female')
df['Country']=df['Country'].replace('0',np.nan)
df['University Degree']=df['University Degree'].replace('0',np.nan)


df5['Gender']=df5['Gender'].replace('f','female')
df5['Country']=df5['Country'].replace('0',np.nan)
df5['University Degree']=df5['University Degree'].replace('0',np.nan)


df['Profession']=df['Profession'].fillna(method='ffill')
df5['Profession']=df5['Profession'].fillna(method='ffill')


imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis = 0)
df['Work Experience in Current Job [years]']= imputer.fit_transform(df['Work Experience in Current Job [years]'].values.reshape(-1,1))
df['Work Experience in Current Job [years]']=pd.DataFrame(df['Work Experience in Current Job [years]'])
df5['Work Experience in Current Job [years]']= imputer.fit_transform(df5['Work Experience in Current Job [years]'].values.reshape(-1,1))
df5['Work Experience in Current Job [years]']=pd.DataFrame(df5['Work Experience in Current Job [years]'])

df['Year of Record']= imputer.fit_transform(df['Year of Record'].values.reshape(-1,1))
df['Year of Record']=pd.DataFrame(df['Year of Record'])
df5['Year of Record']= imputer.fit_transform(df5['Year of Record'].values.reshape(-1,1))
df5['Year of Record']=pd.DataFrame(df5['Year of Record'])


#Reducing cardinality for Gender & University Degree :

def convert_sparse_values(df, col, threshold):
    xxx = df[col].value_counts()[
        df[col].value_counts().cumsum() < df[col].value_counts().sum() * threshold].index.values
    df[col] = df[col].map(lambda x: x if x in xxx else 'other')


convert_sparse_values(df, col='Gender', threshold=.91)


def convert_sparse_values(df, col, threshold):
    xxx = df[col].value_counts()[
        df[col].value_counts().cumsum() < df[col].value_counts().sum() * threshold].index.values
    df[col] = df[col].map(lambda x: x if x in xxx else 'No')


convert_sparse_values(df, col='University Degree', threshold=.98)


def convert_sparse_values(df5, col, threshold):
    xxx = df5[col].value_counts()[
        df5[col].value_counts().cumsum() < df5[col].value_counts().sum() * threshold].index.values
    df5[col] = df5[col].map(lambda x: x if x in xxx else 'other')


convert_sparse_values(df5, col='Gender', threshold=.91)


def convert_sparse_values(df5, col, threshold):
    xxx = df5[col].value_counts()[
        df5[col].value_counts().cumsum() < df5[col].value_counts().sum() * threshold].index.values
    df5[col] = df5[col].map(lambda x: x if x in xxx else 'No')


convert_sparse_values(df5, col='University Degree', threshold=.98)




#imputer = KNNImputer(n_neighbors=4, weights="uniform")

#df1 = df['Year of Record'].values.reshape(-1,1)
#df1 = imputer.fit_transform(df1)
#df1=pd.DataFrame(df1)
#df=pd.concat([df1,df],axis=1)

#df2 = df['Work Experience in Current Job [years]'].values.reshape(-1,1)
#df2 = imputer.fit_transform(df2)
#df2=pd.DataFrame(df2)
#df=pd.concat([df2,df],axis=1)

#df6 = df5['Year of Record'].values.reshape(-1,1)
#d#f6 = imputer.fit_transform(df6)
##df6=pd.DataFrame(df6)
#df5=pd.concat([df6,df5],axis=1)

#df7 = df5['Work Experience in Current Job [years]'].values.reshape(-1,1)
#df7 = imputer.fit_transform(df7)
#df7=pd.DataFrame(df7)
#df5=pd.concat([df7,df5],axis=1)

#Encoding of the categorical columns:

onehotencoder = OneHotEncoder(sparse=False)

enc1 = onehotencoder.fit_transform(df[{'University Degree'}])
enc1 = pd.DataFrame(enc1)
df = pd.concat([df, enc1], axis=1)

enc2 = onehotencoder.fit_transform(df[{'Gender'}])
enc2 = pd.DataFrame(enc2)
df = pd.concat([df, enc2], axis=1)


enc3 = onehotencoder.fit_transform(df5[{'University Degree'}])
enc3 = pd.DataFrame(enc3)
df5 = pd.concat([df5, enc3], axis=1)

enc4 = onehotencoder.fit_transform(df5[{'Gender'}])
enc4 = pd.DataFrame(enc4)
df5 = pd.concat([df5, enc4], axis=1)



#enc3 = onehotencoder.fit_transform(df[{'Satisfation with employer'}])
#enc3 = pd.DataFrame(enc3)
#df = pd.concat([df, enc3], axis=1)

mean_encode1 =df.groupby('Profession')['Total Yearly Income [EUR]'].mean()
df.loc[:,'Profession_mean_enc']=df['Profession'].map(mean_encode1)
df5.loc[:,'Profession_mean_enc']=df5['Profession'].map(mean_encode1)

mean_encode2 =df.groupby('Country')['Total Yearly Income [EUR]'].mean()
df.loc[:,'Country_mean_enc']=df['Country'].map(mean_encode2)
df5.loc[:,'Country_mean_enc']=df5['Country'].map(mean_encode2)

df['Country_mean_enc']=df['Country_mean_enc'].fillna(method='ffill')
df5['Country_mean_enc']=df5['Country_mean_enc'].fillna(method='ffill')

#encoder = ce.CatBoostEncoder(cols=[[df['Profession']],[df5['Profession']],[df['Country']],[df5['Country']]]



##,'Year of Record','Work Experience in Current Job [years]'

df.drop(['Housing Situation','University Degree','Profession','Country','Gender'],axis=1,inplace=True)
df5.drop(['Housing Situation','University Degree','Profession','Country','Gender'],axis=1,inplace=True)



#Assigning Predictors and Target Value for training :
y = df['Total Yearly Income [EUR]'].values
x = df.drop(['Total Yearly Income [EUR]'],axis=1).values
x1=df5.drop(['Total Yearly Income [EUR]'],axis=1).values


#Train-Test split:
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

#print(df.isna().any())

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#lgb_train = lgb.Dataset(x_train, y_train)
#lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


#regressor = LinearRegression()
#regressor.fit(x_train, y_train)
#y_pred = regressor.predict(x_test)

gbm = lgb.LGBMRegressor(n_estimators=100000,learning_rate=0.01)
gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)],eval_metric='l1')
y_pred= gbm.predict(x_test, num_iteration=gbm.best_iteration_)

#model = model = xgboost.XGBRegressor()
#fit_model = model.fit(x_train, y_train)
#y_pred =fit_model.predict(x_test)
y1=regressor.predict(x1)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))
y1=pd.DataFrame(y1)
y1.to_csv("Income_Upload.csv")