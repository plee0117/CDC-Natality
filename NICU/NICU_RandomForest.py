# -*- coding: utf-8 -*-
"""RandomForest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16Izqr0sIEcAHfNOjHaQL6HD_ncA6Oojk
"""

#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
import re

#Import dataset

# trimmed2018 = pd.read_csv('cdc2018trimmed.csv') #(84)

# csv2018 = pd.read_csv('CSV2018.csv') # with bettina's function as well, (98 lol)

# trimwolabor2018 = pd.read_csv('cdc2018wolabor.csv') #(68)

# trimmed2 = pd.read_csv('cdc2018trimmed2.csv') #(84)

# trimmed2016 = pd.read_csv("CDC2016trimmed.csv") #84.9

# trimwolabor2016 = pd.read_csv('CDC2016wolabor.csv') #67.3

trimmed2017 = pd.read_csv('CDC2017trimmed.csv') #84.8
#predictors
# [b'OEGest_R10',
#  b'AB_AVEN1',
#  b'GESTREC10',
#  b'BWTR12',
#  b'AB_SURF',
#  b'APGAR5',
#  b'DOB_TT',
#  b'ME_ROUT',
#  b'DOB_MM',
#  b'PREVIS_REC']

dsample = trimmed2017

#Downsample function, thanks Bettina and Aron!
def downsample_df (df):

    '''
    Remove undefined information on NICU admissions (AB_NICU == 'U'),
    create a binary target vector, and create a "balanced" dataframe
    with all NICU admissions and matching numbers of randomly selected non-NICU admissions.
    '''

    import pandas as pd
    import numpy as np

    # remove unknown class from df
    df_no_unknown = df[df['AB_NICU'].isin(['Y', 'N'])]

    # Create binary target vector, NICU = yes classified as class 0
    df_y_n = np.where((df_no_unknown['AB_NICU'] == 'Y'), 0, 1)

    # Get indicies of each class' observations
    index_class0 = np.where(df_y_n == 0)[0]
    index_class1 = np.where(df_y_n == 1)[0]

    # Get numbers of observations in class 0
    n_class0 = len(index_class0)

    # Randomly sample the same number of observations from class 1 as in class 0, without replacement
    np.random.seed(0)
    index_class1_downsampled = np.random.choice(index_class1, size=n_class0, replace=False)

    # Create dataframes for NICU and downsampled non-NICU
    df_defect = df_no_unknown.iloc[index_class0]
    df_adj_NONdefect = df_no_unknown.iloc[index_class1_downsampled]

    # Append into 1 dataframe
    df_downsampled = df_defect.append(df_adj_NONdefect)

    return df_downsampled

#Downsampled
dsample = downsample_df(dsample)

def create_reduced_df (df, list_to_drop):

    '''
    Function to choose columns from dataframe for label encoding. Takes the data frame and the columns to drop
    as a list.
    '''

    import pandas as pd
    import numpy as np

    # create list of flags, territory info and imputed info likely to be dropped together
    flags = list(filter(lambda i: re.search('\AF_',i) , df.columns))
    territory_info = ['OCTERR','OCNTYFIPS', 'OCNTYPOP', 'MBCNTRY', 'MRCNTRY', 'MRTERR', 'RCNTY', 'RCNTY_POP', 'RCNTY_POP',
                 'RCITY_POP', 'RECTYPE']
    imputed_info = ['MAGE_IMPFLG', 'MAGE_REPFLG', 'MRACEIMP','MAR_IMP', 'FAGERPT_FLG', 'IMP_PLUR', 'IMP_SEX',
                'COMPGST_IMP', 'OBGEST_FLG', 'LMPUSED']

    # create a copy of dataframe
    df2 = df.copy()

    # compare columns in case they have already been dropped in the input df
    

    # drop columns
    for feature in list_to_drop:
        #if ~feature.isin(df2.colunms):
        #    except ValueError:
        #    print("Column name does not exist")
        if feature == 'flag':
            df2.drop(flags, inplace = True, axis=1)
        elif feature == 'territory':
            df2.drop(territory_info, inplace = True, axis=1)
        elif feature == 'imputed':
            df2.drop(imputed_info, inplace = True, axis=1)
        else:
            df2.drop(feature, inplace = True, axis=1)

    return df2

# create list of flags, territory info and imputed info likely to be dropped together
#only to be used with FULL data sets (ie: CSV2018)
flags = list(filter(lambda i: re.search('\AF_',i) , dsample.columns))
territory_info = ['OCTERR','OCNTYFIPS', 'OCNTYPOP', 'MBCNTRY', 'MRCNTRY', 'MRTERR', 'RCNTY', 'RCNTY_POP',
                 'RCITY_POP', 'RECTYPE']
imputed_info = ['MAGE_IMPFLG', 'MAGE_REPFLG', 'MRACEIMP','MAR_IMP', 'FAGERPT_FLG', 'IMP_PLUR', 'IMP_SEX',
                'COMPGST_IMP', 'OBGEST_FLG', 'LMPUSED']
dsample = create_reduced_df(dsample,flags)
dsample = create_reduced_df(dsample,territory_info)
dsample = create_reduced_df(dsample,imputed_info)

#LabelEncoding Function. Thanks Ira!
def LabelEncoding(dataframe):
    '''
    Function that takes a dataframe and transforms it with label encoding on all the categorical features.
    '''
    
    import pandas as pd
    
    #create a list using object types since dataframe.dtypes.value_counts() only shows objects and int64
    objlist = list(dataframe.select_dtypes(include=['object']).columns)
    
    #change type then transform column using cat codes
    for col in objlist:
        dataframe[col] = dataframe[col].astype('category')
        dataframe[col] = dataframe[col].cat.codes
    
    return dataframe

#Label Encoded
dsample = LabelEncoding(dsample)

def targetchoice(column,dataframe):
    '''
    Takes a column and a dataframe, returns four values for;
    x_train, X_test, y_train, and y_test
    '''
    from sklearn.model_selection import train_test_split
    
    #Cutting the data and target dataframes
    sample_data = dataframe.loc[:, dsample.columns != column ]
    sample_target = dataframe.loc[:,column]
    
    #assigning to variables
    X_train, X_test, y_train, y_test = train_test_split(sample_data, sample_target, test_size=0.2, random_state=0)
    
    #appending to a list to return for multi-assignment
    varlist = []
    varlist.append(X_train)
    varlist.append(X_test)
    varlist.append(y_train)
    varlist.append(y_test)
    return varlist

#format is "X_train, X_test, y_train, y_test = targetchoice()"

X_train, X_test, y_train, y_test = targetchoice('AB_NICU',dsample)

#RANDOM FOREST INITIAL FIT-
randomForest = ensemble.RandomForestClassifier()
randomForest.set_params(random_state=0)
randomForest.fit(X_train, y_train) 
print("The training error is: %.5f" % (1 - randomForest.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - randomForest.score(X_test, y_test)))

# Commented out IPython magic to ensure Python compatibility.
# set the parameter grid
grid_para_forest = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 16),
    'n_estimators': range(10, 50, 30)
}
# grid search
grid_search_forest = ms.GridSearchCV(randomForest, grid_para_forest, scoring='accuracy', cv=5, n_jobs=-1,)
# %time grid_search_forest.fit(X_train, y_train)

# Best Params so far: {'criterion': 'gini', 'max_depth': 15, 'n_estimators': 40}
# 0.8431166186317793
print(grid_search_forest.best_params_)
grid_search_forest.best_score_

# get the training/test errors
print("The training error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_test, y_test)))

#list of feature importance
feature_importance = list(zip(dsample.columns, randomForest.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_importance = np.array(feature_importance, dtype=dtype)
feature_sort = np.sort(feature_importance, order='importance')[::-1]
[i for (i, j) in feature_sort[0:10]]

#Sorting feature importance
sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
sorted_features

# Plot
features_top10 = sorted_features[:10]
featureNames, featureScores = zip(*list(features_top10))
plt.barh(range(len(featureScores)), featureScores, tick_label=featureNames)
plt.title('feature importance')

#Thanks Drucila!
feature_importance = 100.0 * (randomForest.feature_importances_ / randomForest.feature_importances_.max())
important_features = X_train.columns[feature_importance >= 10]
unimportant_features = X_train.columns[feature_importance < 5]

important_features

unimportant_features