#!/usr/bin/env python
# coding: utf-8

# In[406]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[407]:


# Importing the datasets
train_dat = pd.read_csv('train.csv')
test_dat = pd.read_csv('evaluation.csv', header = None)
test_dat.columns = ['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
'passenger_count']


# In[408]:


# Check for presence of null values

train_dat.isnull().sum()
test_dat.isnull().sum()


# In[409]:


# Taking a glimpse at the data
test_dat.head()


# ### Exploratory data analysis

# In[410]:


train_dat.describe()


# In[411]:


train_dat['fare_amount'].hist()
plt.xlabel('fare')


# In[412]:


plt.scatter(train_dat['passenger_count'], train_dat['fare_amount'])
plt.xlabel('passenger count')
plt.ylabel('fare')


# In[413]:


train_dat['pickup_latitude'].hist()
plt.xlabel('pickup_latitude')


# In[414]:


train_dat['pickup_longitude'].hist()
plt.xlabel('pickup_longitude')


# In[415]:


train_dat['dropoff_latitude'].hist()
plt.xlabel('dropoff_latitude')


# In[416]:


train_dat['dropoff_longitude'].hist()
plt.xlabel('dropoff_longitude')


# ### Feature engineering

# In[417]:


# Extracting a feature indicating the time-phase in the day
# The phase is obtained by binning the day times as follows:
# 00:00 - 06:00 (phase 1), 06:00 - 12:00 (phase 2), 12:00 - 18:00 (phase 3), 18:00 - 00:00 (phase 4)

pick_up = train_dat['pickup_datetime'].values
time_ = []
time_binned = []
for t in pick_up:
    hrs = int(t.split()[1].split(':')[0])
    time_.append(hrs)
    
for d in time_:
    if d >= 0 and d < 6:
        time_binned.append('P1')
    elif d >= 6 and d <= 12:
        time_binned.append('P2')
    elif d > 12 and d <= 18:
        time_binned.append('P3')
    else:
        time_binned.append('P4')
        
pick_up1 = test_dat['pickup_datetime'].values
time1_ = []
time1_binned = []
for t in pick_up1:
    hrs = int(t.split()[1].split(':')[0])
    time1_.append(hrs)
    
for d in time1_:
    if d >= 0 and d < 6:
        time1_binned.append('P1')
    elif d >= 6 and d <= 12:
        time1_binned.append('P2')
    elif d > 12 and d <= 18:
        time1_binned.append('P3')
    else:
        time1_binned.append('P4')        


# In[418]:


# Add this time phase column to the data sets

train_dat['TIME'] = time_
test_dat['TIME'] = time1_


# In[419]:


# Extracting a feature indicating the season of the year
# The season is obtained by binning the months as follows:
# 1 - 3 (season 1), 4 - 6 (season 2), 7 - 9 (season 3), 10 - 12 (season 4)

# Implementation on the train set
pick_up = train_dat['pickup_datetime'].values
mnth_ = []
mnth_binned = []
for t in pick_up:
    mn = int(t.split()[0].split('-')[1])
    mnth_.append(mn)
    
for d in mnth_:
    if d >= 1 and d <= 3:
        mnth_binned.append('S1')
    elif d > 3  and d <= 6:
        mnth_binned.append('S2')
    elif d > 6 and d <= 9:
        mnth_binned.append('S3')
    else:
        mnth_binned.append('S4')
        
# Implementation on the test set
pick_up1 = test_dat['pickup_datetime'].values
mnth1_ = []
mnth1_binned = []
for t in pick_up1:
    mn = int(t.split()[0].split('-')[1])
    mnth1_.append(mn)
    
for d in mnth1_:
    if d >= 1 and d <= 3:
        mnth1_binned.append('S1')
    elif d > 3  and d <= 6:
        mnth1_binned.append('S2')
    elif d > 6 and d <= 9:
        mnth1_binned.append('S3')
    else:
        mnth1_binned.append('S4')        


# In[420]:


# Add the season column to the datasets

train_dat['MNTH'] = mnth_
test_dat['MNTH'] = mnth1_


# In[421]:


# Define the rmse loss
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[422]:


# # Baseline model 1: Predict the mean fare for each ride

# pred1 = np.mean(train_dat['fare_amount'])
# baseline1_pred = np.repeat(pred1,y_test.shape[0])
# rmse(baseline1_pred, y_test)


# In[423]:


# # Baseline model 2: Predict the mean fare for each ride based on the phase in the day
# baseline_pred2 = []
# mean_byphase = {}

# train_dat['TIME_BIN'] = time_binned

# for p in ['P1', 'P2', 'P3', 'P4']:
#     mean_byphase[p] = np.mean(train_dat.loc[train_dat['TIME_BIN'] == p]['fare_amount'])
# for i in range(len(train_dat)):
#      baseline_pred2.append(mean_byphase[train_dat.iloc[i]['TIME_BIN']])
# train_dat.drop(['TIME_BIN'], axis = 1, inplace = True)
# rmse(baseline_pred2, train_dat['fare_amount'])


# In[424]:


from sklearn.preprocessing import OneHotEncoder
oce = OneHotEncoder()
ocem = OneHotEncoder()

# A good modelling choice is to one-hot encode the phase in the day since it is a categorical variable with no sense of order 
time_binned1 = np.reshape(time_binned, (-1, 1))
oh_time = oce.fit_transform(np.reshape(time_binned1, (-1, 1)))
oh_time1 = oh_time.toarray()


mnth_binned1 = np.reshape(mnth_binned, (-1, 1))
oh_mnth = ocem.fit_transform(np.reshape(mnth_binned1, (-1, 1)))
oh_mnth1 = oh_mnth.toarray()


# Transforming the test data similarly
time1_binned1 = np.reshape(time1_binned, (-1, 1))
oh1_time = oce.transform(np.reshape(time1_binned1, (-1, 1)))
oh1_time1 = oh1_time.toarray()


mnth1_binned1 = np.reshape(mnth1_binned, (-1, 1))
oh1_mnth = ocem.transform(np.reshape(mnth1_binned1, (-1, 1)))
oh1_mnth1 = oh1_mnth.toarray()


# In[425]:


add_feat_train = pd.DataFrame(np.concatenate([oh_time1, oh_mnth1], axis = 1))
add_feat_test = pd.DataFrame(np.concatenate([oh1_time1, oh1_mnth1], axis = 1))


# In[426]:


# Add column names for train data
add_feat_train.columns =  ['P1', 'P2', 'P3', 'P4', 'S1', 'S2', 'S3', 'S4']

# Considering n-1 columns when the number of categories is n to avoid correlated columns
add_feat_train.drop(['P4', 'S4'], axis = 1, inplace = True)

# Add column names for test data
# Considering n-1 columns when the number of categories is n to avoid correlated columns
add_feat_test.columns =  ['P1', 'P2', 'P3', 'P4', 'S1', 'S2', 'S3', 'S4']
add_feat_test.drop(['P4', 'S4'], axis = 1, inplace = True)


# In[427]:


train_final = pd.concat([train_dat, add_feat_train], axis = 1)
test_final = pd.concat([test_dat, add_feat_test], axis = 1)


# In[428]:


# Getting rid of unnecessary columns in train and test datasets

X_tr = train_final.drop(['key', 'fare_amount', 'pickup_datetime', 'TIME', 'MNTH'], axis = 1)
Y_tr = train_final['fare_amount']

X_te = test_final.drop(['key', 'pickup_datetime', 'TIME', 'MNTH'], axis = 1)


# In[429]:


# Keep a track on the changes beig made
X_tr.head()


# In[430]:


# Feature scaling to bring all features to the same scale
cols_tbs = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
tb_scaled_train = X_tr[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
tb_scaled_test = X_te[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]


# In[431]:


from sklearn.preprocessing import StandardScaler

sl = StandardScaler()

scaled_feat_tr = sl.fit_transform(tb_scaled_train)
scaled_feat_te = sl.fit_transform(tb_scaled_test)

X1_tr = pd.DataFrame(scaled_feat_tr, columns = cols_tbs)
X2_tr = X_tr.drop(cols_tbs, axis = 1)

X1_te = pd.DataFrame(scaled_feat_te, columns = cols_tbs)
X2_te = X_te.drop(cols_tbs, axis = 1)

Xtr_fin = pd.concat([X1_tr, X2_tr], axis = 1)
Xte_fin = pd.concat([X1_te, X2_te], axis = 1)


# ### MODELLING

# In[432]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xtr_fin, Y_tr)


# ### Hyperparameter tuning
# 
# ##### Note, there is much more to tuning that I could have done than what I have done (due to time constraints)

# In[ ]:


# Hyper parameter tuning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 

# Define pipeline for SVC
pipe_rf = Pipeline([('clf', RandomForestRegressor(random_state=1))])

# Define parameter range 
# The other hyperparametes are not being tuned due to time constraint
param_range1 = [300, 500]
# Define parameter grid for different combinations of classifier kernel and parameters
param_grid = [{'clf__n_estimators' : param_range1}] 

# Perform grid search to find the scores for each combination
gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
gs = gs.fit(X_train, y_train) 

print(gs.best_score_) 
print(gs.best_params_)


# In[433]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs = -1)

# # fit the estimator
model.fit(X_train, y_train)


# In[434]:


model.score(X_train, y_train)


# In[435]:


model.score(X_test, y_test)


# In[437]:


# This is a significant improvement over the 2 base line models
rmse(model.predict(X_test), y_test)


# In[438]:


# Then, generate a feature importance plot
plt.figure(num=None, figsize=(20, 22), dpi=80, facecolor='w', edgecolor='k')

feat_importances = pd.Series(model.feature_importances_, index= X_train.columns)
feat_importances.nlargest(15).plot(kind='barh')


# In[439]:


pred_test = model.predict(Xte_fin)
final_pred = pd.DataFrame({'key' : list(range(len(Xte_fin))), 'predictions' : pred_test})
final_pred.to_csv('AkshayKhare_predictions.csv', index = False)


# In[ ]:


# Additional factor that can be included and where to get that data from...
# Number of cabs in the vicintiy of that location and current demand for rides in that area

