import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score


# parameters

n_estimators=200, 
max_depth=10, 
min_samples_leaf=3
output_file = 'model_rf.bin'


# data preparation
columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

df1 = pd.read_csv('adult.data',names=columns)
df2 = pd.read_csv('adult.test',names=columns)

df2.drop(0, inplace=True)
df2['age'] = df2['age'].astype('int64')

df = pd.concat([df1, df2])

df['income'] = df['income'].map({' <=50K':0,' <=50K.':0, ' >50K':1,' >50K.':1})

#Removing unknown records " ?" from the columns workclass, occupation
df.drop(df[df['workclass'] == " ?"].index, inplace=True)
df.drop(df[df['occupation'] == " ?"].index, inplace=True)
df.drop(df[df['native-country'] == " ?"].index, inplace=True)

categorical = list(df.columns[df.dtypes == "object"])

numerical = list(df.columns[df.dtypes != "object"])


# training 

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.income.values
y_val = df_val.income.values
y_test = df_test.income.values

del df_train['income']
del df_val['income']
del df_test['income']

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
auc = roc_auc_score(y_val, y_pred)
print(f'auc of train data is {auc}')


#Computing Final RMSE

y_pred = rf.predict(X_test)
auc_rf_test = roc_auc_score(y_test, y_pred)
#rmse_rf_test = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'auc of test data is {auc_rf_test}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is saved to {output_file}')