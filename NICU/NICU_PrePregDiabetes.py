#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import pickle
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


#Import dataset
cdc3yearswolabor = pd.read_csv('cdc3yearswolabor.csv')

#get rid of 'U' of AB_NICU
cdc3yearswolabor =  cdc3yearswolabor[cdc3yearswolabor['AB_NICU'].isin(['Y', 'N'])]

cdc3yearswolabor.columns

def create_random_column(df):
    '''
    this creates a list of random numbers between 1 and 1000
    of the same lenght as each column in the dataframe, appends
    a column named "RANDOM" to the dataframe
    '''
    import random
    mylist = []
    for i in range(0,df.shape[0]):
        x = random.randint(1,1000)
        mylist.append(x)
    df['RANDOM'] = mylist
    
    return df

cdc3yearswolabor = create_random_column(cdc3yearswolabor)


# create 2 new datasets 
# 1. mothers that have risk factor diabetes
diabetes_df = cdc3yearswolabor[cdc3yearswolabor['RF_PDIAB'] == 'Y']
print(diabetes_df.shape)


#LabelEncoding Function. Thanks Ira!
def LabelEncoding(dataframe):
    '''
    Function that takes a dataframe and transforms it with label encoding on all the categorical features.
    '''
    
    #create a list using object types since dataframe.dtypes.value_counts() only shows objects and int64
    objlist = list(dataframe.select_dtypes(include=['object']).columns)
    
    #change type then transform column using cat codes
    for col in objlist:
        dataframe[col] = dataframe[col].astype('category')
        dataframe[col] = dataframe[col].cat.codes
    
    return dataframe

diabetes_df = LabelEncoding(diabetes_df)

# function to split out holdout test set
def split_sets(dataframe, seed, test_prop=0.1): 
    '''
    - A function that splits specifically a dataframe into a train and test portion
    - Requires multiple assignment: train, test
    ---------------
    - dataframe: dataframe to be split
    - seed: set seed for reproducability
    - test_prop: takes a float - proportion of dataframe that should be allocated to the test set
    '''

    np.random.seed(seed)
    testIdxes = np.random.choice(range(0,dataframe.shape[0]), size=round(dataframe.shape[0]*test_prop), replace=False)
    trainIdxes = list(set(range(0,dataframe.shape[0])) - set(testIdxes))

    train = dataframe.iloc[trainIdxes,:]
    test  = dataframe.iloc[testIdxes,:]
    
    return train, test

train_dia, test_dia = split_sets(diabetes_df, 0, test_prop=0.1)

X_train_dia = train_dia.drop('AB_NICU', axis=1)
y_train_dia = train_dia['AB_NICU']
X_test_dia = test_dia.drop('AB_NICU', axis=1)
y_test_dia = test_dia['AB_NICU']

#XGBoost initial fit 
xgb = XGBClassifier()
xgb.set_params(random_state=0)
xgb.fit(X_train_dia, y_train_dia)
print("The training error is: %.5f" % (1 - xgb.score(X_train_dia, y_train_dia)))
print("The test error is: %.5f" % (1 - xgb.score(X_test_dia, y_test_dia)))

#initial confusion matrix
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test_dia, xgb.predict(X_test_dia))
cm_test

#weight 2 fit
xgb_2 = XGBClassifier()
xgb_2.set_params(random_state=0, scale_pos_weight = 2)
xgb_2.fit(X_train_dia, y_train_dia)
print("The training error is: %.5f" % (1 - xgb_2.score(X_train_dia, y_train_dia)))
print("The test error is: %.5f" % (1 - xgb_2.score(X_test_dia, y_test_dia)))

#weight 2 confusion matrix
cm_test_2 = confusion_matrix(y_test_dia, xgb_2.predict(X_test_dia))
cm_test_2

#weight 1 fit
xgb_3 = XGBClassifier()
xgb_3.set_params(random_state=0, scale_pos_weight = 1)
xgb_3.fit(X_train_dia, y_train_dia)
print("The training error is: %.5f" % (1 - xgb_3.score(X_train_dia, y_train_dia)))
print("The test error is: %.5f" % (1 - xgb_3.score(X_test_dia, y_test_dia)))

#weight 1 confusion matrix
cm_test_3 = confusion_matrix(y_test_dia, xgb_3.predict(X_test_dia))
cm_test_3

# set the parameter grid
xgb_param_grid ={'learning_rate': [0.001, 0.01, 0.05],
                 'max_depth': [4,5,6],
                 'min_child_weight': [4,5,6],
                 'n_estimators': [200, 300, 400, 500]}

#grid search
grid_search_xgb_2 = GridSearchCV(xgb_2, xgb_param_grid, scoring='precision', cv= 3, n_jobs=-1, return_train_score = True)
%time grid_search_xgb_2.fit(X_train_dia, y_train_dia)


# get the best parameters
print(grid_search_xgb_2.best_params_)
print(grid_search_xgb_2.best_score_)

#confusion matrix
confusion_matrix(y_test_dia, grid_search_xgb_2.best_estimator_.predict(X_test_dia))

# set the parameter grid
xgb_param_grid ={'learning_rate': [0.01],
                 'max_depth': [2,3,4],
                 'min_child_weight': [6,7,8],
                 'n_estimators': [500,600,700]}

#grid search
grid_search_xgb_2 = GridSearchCV(xgb_2, xgb_param_grid, scoring='precision', cv= 3, n_jobs=-1, return_train_score = True)
%time grid_search_xgb_2.fit(X_train_dia, y_train_dia)

# get the best parameters
print(grid_search_xgb_2.best_params_)
print(grid_search_xgb_2.best_score_)

#confusion matrix
confusion_matrix(y_test_dia, grid_search_xgb_2.best_estimator_.predict(X_test_dia))

# set the parameter grid
xgb_param_grid ={'learning_rate': [0.01],
                 'max_depth': [3],
                 'min_child_weight': [6],
                 'n_estimators': [700, 900, 1100]}

#grid search
grid_search_xgb_2 = GridSearchCV(xgb_2, xgb_param_grid, scoring='precision', cv= 3, n_jobs=-1, return_train_score = True)
%time grid_search_xgb_2.fit(X_train_dia, y_train_dia)

# get the best parameters
print(grid_search_xgb_2.best_params_)
print(grid_search_xgb_2.best_score_)

confusion_matrix(y_test_dia, grid_search_xgb_2.best_estimator_.predict(X_test_dia))

# Get numerical feature importances
importances_xgb = list(grid_search_xgb_2.best_estimator_.feature_importances_)

# List of tuples with variable and importance
feature_importances_xgb = [(feature, round(importance, 5)) for feature, importance in zip(X_train_dia, importances_xgb)]

# Sort the feature importances by most important first
xgb_feature_importances = sorted(feature_importances_xgb, key = lambda x: x[1], reverse = True )

# Print out the feature and importances 
[print('Variable: {:10} Importance: {}'.format(*pair)) for pair in xgb_feature_importances]

xgb_feature_importances_top20 = xgb_feature_importances[:20]
featureNames, featureScores = zip(*list(xgb_feature_importances_top20))
xgb_feature_importances_top20

plt.barh(range(len(featureScores)), featureScores, tick_label=featureNames)
plt.gca().invert_yaxis()
plt.title('feature importance')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importances')
plt.savefig('xgbFI.png')
