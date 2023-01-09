

import csv
import os
import pandas as pd
import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import catboost
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sklearn.metrics as metrics

train = pd.read_csv('/Users/edwinhuang/Desktop/CS everything/科研/cis代码/input/train.csv', usecols=[ 'imdb_id', 'budget', 'popularity','runtime', 'release_date', 'revenue'])

train = pd.merge(train, pd.read_csv('/Users/edwinhuang/Desktop/CS\ everything/科研/cis代码/input/TrainAdditionalFeatures.csv '), how='left', on=['imdb_id'])


train1 = train.drop(train[train['budget'] < 1000].index)

train2 = train1.drop(train1[train1['runtime'] < 60].index)

train2[['release_month','release_day','release_year']]=train2['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)
train2['release_year'] = train2['release_year']
train2.loc[ (train2['release_year'] <= 18) & (train2['release_year'] < 100), "release_year"] += 2000
train2.loc[ (train2['release_year'] > 18)  & (train2['release_year'] < 100), "release_year"] += 1900

train2 = train2.dropna()


result = train2.revenue

trainfeature = train2.drop(labels="revenue", axis=1)
trainfeature1 = trainfeature.drop(labels="release_date",axis=1)
trainfeature2 = trainfeature1.drop(labels="imdb_id",axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    trainfeature2, result, test_size=0.20, random_state=42)


params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          }




my_model_2 = ensemble.GradientBoostingRegressor(**params)
my_model_2.fit(X_train, y_train)

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(my_model_2.staged_predict(X_test)):
    test_score[i] = my_model_2.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, my_model_2.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()




print(my_model_2.score(X_test, y_test))

#lightgbm model set up

params_lgb = {'objective':'regression',
         'num_leaves' : 6,
         'min_data_in_leaf' : 10,
         'max_depth' : 5,
         'learning_rate': 0.001,
         'feature_fraction':0.2,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         'lambda_l1': 0.2,
         "metric": 'rmse',
         'subsample':.8, 
         'colsample_bytree':.9,
         "random_state" : 2020,
         "verbosity": -1}

val_x = X_train.values
val_y = y_train.values

record = dict()
lgb_1 = lgb.train(params_lgb
                , lgb.Dataset(X_train, y_train)
                , num_boost_round = 100000
                , valid_sets = [lgb.Dataset(val_x, val_y)]
                , early_stopping_rounds = 500
                , callbacks = [lgb.record_evaluation(record)]
                     )


y_lgb_pre=lgb_1.predict(X_test)

s = r2_score(y_test, y_lgb_pre)
print("R2 score for lgb model:",s)



    
print('Plotting feature importances...')
ax = lgb.plot_importance(lgb_1, max_num_features=10)
plt.show()

              


#cat model set up

print("setting up cat model")

cat_model = CatBoostRegressor(iterations=100000,
                                 learning_rate=0.004,
                                 depth=5,
                                 eval_metric='RMSE',
                                 colsample_bylevel=0.8,
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200
                                )
cat_model.fit(X_train, y_train,
                 eval_set=(val_x, val_y),
                 use_best_model=True,
                 verbose=False)

print("set up finish")


y_cat_pre=cat_model.predict(X_test)
s2 = r2_score(y_test, y_cat_pre)
print("R2 score for cat model:", s2)



feature_importances = cat_model.get_feature_importance()
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))



#create heat map
train2 = train2[['budget','rating','totalVotes','popularity','runtime','release_year','release_month','release_day','revenue']]
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(train2.corr(), annot=True)
plt.show()


