import Functions.py

# Baseline Dataset
variables = {'nominal_categorical':['MBSTATE_REC','MRACEHISP','MAR_P','DMAR','MEDUC','FRACEHISP',\
                                    'FEDUC','WIC','RF_PDIAB','RF_GDIAB','RF_PHYPE','RF_GHYPE',\
                                    'RF_EHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR',\
                                  'IP_GON','IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'PAY', 'SEX'],\
           'ordinal_categorical':['PRECARE', 'DOB_MM'],\
           'continuous':['MAGER', 'FAGECOMB','PRIORTERM','PRIORLIVE','PRIORDEAD','LBO_REC','TBO_REC',\
                         'ILLB_R','ILOP_R','ILP_R','PREVIS','CIG_0','CIG_1','CIG_2','CIG_3','M_Ht_In','BMI',\
                         'WTGAIN','RF_CESARN','OEGest_Comb'],\
            'target':['CA_CCHD']}

#create missingness types:
missing_dict = {'cont9': ['LBO_REC', 'TBO_REC'],\
                'cont99': ['FAGECOMB', 'PRIORTERM','PRIORLIVE', 'PRIORDEAD', 'PRECARE', 'PREVIS',\
                         'CIG_0', 'CIG_1', 'CIG_2', 'CIG_3', 'M_Ht_In', 'WTGAIN', 'RF_CESARN', 'OEGest_Comb'],\
                'cont999':['ILLB_R', 'ILP_R', 'ILOP_R'],\
                'cont99.9': ['BMI'],\
                'cat3': ['MBSTATE_REC'],\
                'cat8': ['MRACEHISP'],\
                'cat9': ['MEDUC', 'FEDUC', 'PAY', 'FRACEHISP', 'DMAR'],\
                'catU': ['WIC','RF_PDIAB','RF_GDIAB','RF_PHYPE',\
                        'RF_GHYPE','RF_EHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR','IP_GON',\
                        'IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'MAR_P']}
missing_vals = [9,99,999,99.9,3,8,9,'U']


#pull selected variables from 2016-2018 databases in SQL and append to a single dataframe
query18 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2018"

query17 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2017"

query16 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2016"


queries = [query18, query17, query16]
            
cchd = pd.DataFrame()
test_cchd = pd.DataFrame()

for query in queries:
    temp = create_table_from_SQL('root','cdc',sql_pw, query)
    train, test = split_sets(temp, 0, test_prop=0.1)
    train = downsample_df(train, 'CA_CCHD')
    cchd = cchd.append(train)  
    test_cchd = test_cchd.append(test)



chd=cchd.copy()

chd_test = test_cchd.copy()



chd = mlp_all_of_the_above(chd,chd,'CA_CCHD')

chd['lrg_miss_imp'] = chd.FAGECOMB_IMP | chd.FRACEHISP_IMP | chd.ILOP_R_IMP | chd.ILP_R_IMP | chd.FEDUC_IMP
chd.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd = LabelEncoding(chd)

chd = add_random_column_to_df(chd)



chd_test = chd_test.loc[(chd_test[target]=='Y')|(chd_test[target]=='N'),:]

chd_test = mlp_all_of_the_above(chd_test,chd,'CA_CCHD')

chd_test['lrg_miss_imp'] = chd_test.FAGECOMB_IMP | chd_test.FRACEHISP_IMP | chd_test.ILOP_R_IMP | chd_test.ILP_R_IMP | chd_test.FEDUC_IMP
chd_test.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd_test = LabelEncoding(chd_test)

chd_test = add_random_column_to_df(chd_test)

X_test.columns

#test train split
target = 'CA_CCHD'
X_train = chd.drop(target, axis=1)
y_train = chd[target]

X_test = chd_test.drop(target, axis=1)
y_test = chd_test[target]

#RANDOM FOREST INITIAL FIT
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(class_weight={0:1,1:1700})
randomForest.set_params(random_state=0)
randomForest.fit(X_train, y_train) 
print("The training error is: %.5f" % (1 - randomForest.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - randomForest.score(X_test, y_test)))

# set the parameter grid
score_method = 'precision'

import sklearn.model_selection as ms
grid_para_forest = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 16),
    'n_estimators': range(10, 70, 5)
}

# GRID SEARCH
grid_search_forest = ms.GridSearchCV(randomForest, grid_para_forest, scoring=score_method, cv=10, n_jobs=-1,)
%time grid_search_forest.fit(X_train, y_train)

#Best Params and Score
print(grid_search_forest.best_params_)
print('best score:', grid_search_forest.best_score_)

# get the training/test errors
print("The training error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_test, y_test)))

#list of feature importance
feature_importance = list(zip(X_train.columns, randomForest.feature_importances_))
# dtype = [('feature', 'S10'), ('importance', 'float')]
# feature_importance = np.array(feature_importance, dtype=dtype)
# feature_sort = np.sort(feature_importance, order='importance')[::-1]

#Sorting feature importance
# sorted_features = pd.DataFrame(sorted(feature_importance, key=lambda x: x[1], reverse=True))
sorted_features = pd.DataFrame((list(t) for t in feature_importance),columns = ['feature','importance']
                              ).sort_values('importance',ascending = False).reset_index().drop(columns='index')
print(sorted_features)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
%time cm = confusion_matrix(y_test, grid_search_forest.best_estimator_.predict(X_test))

print(score_method, ': ', cm[1,1]/sum(cm[:,1])*100)
print(cm)

### Diabetes Model
variables = {'nominal_categorical':['MBSTATE_REC','MRACEHISP','MAR_P','DMAR','MEDUC','FRACEHISP',\
                                    'FEDUC','WIC','RF_PDIAB','RF_GDIAB','RF_PHYPE','RF_GHYPE',\
                                    'RF_EHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR',\
                                  'IP_GON','IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'PAY', 'SEX'],\
           'ordinal_categorical':['PRECARE', 'DOB_MM'],\
           'continuous':['MAGER', 'FAGECOMB','PRIORTERM','PRIORLIVE','PRIORDEAD','LBO_REC','TBO_REC',\
                         'ILLB_R','ILOP_R','ILP_R','PREVIS','CIG_0','CIG_1','CIG_2','CIG_3','M_Ht_In','BMI',\
                         'WTGAIN','RF_CESARN','OEGest_Comb'],\
            'target':['CA_CCHD']}

#create missingness types:
missing_dict = {'cont9': ['LBO_REC', 'TBO_REC'],\
                'cont99': ['FAGECOMB', 'PRIORTERM','PRIORLIVE', 'PRIORDEAD', 'PRECARE', 'PREVIS',\
                         'CIG_0', 'CIG_1', 'CIG_2', 'CIG_3', 'M_Ht_In', 'WTGAIN', 'RF_CESARN', 'OEGest_Comb'],\
                'cont999':['ILLB_R', 'ILP_R', 'ILOP_R'],\
                'cont99.9': ['BMI'],\
                'cat3': ['MBSTATE_REC'],\
                'cat8': ['MRACEHISP'],\
                'cat9': ['MEDUC', 'FEDUC', 'PAY', 'FRACEHISP', 'DMAR'],\
                'catU': ['WIC','RF_PDIAB','RF_GDIAB','RF_PHYPE',\
                        'RF_GHYPE','RF_EHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR','IP_GON',\
                        'IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'MAR_P']}
missing_vals = [9,99,999,99.9,3,8,9,'U']


#pull selected variables from 2016-2018 databases in SQL and append to a single dataframe
query18 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2018"

query17 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2017"

query16 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2016"


queries = [query18, query17, query16]
            
cchd = pd.DataFrame()
test_cchd = pd.DataFrame()

for query in queries:
    temp = create_table_from_SQL('root','cdc',sql_pw, query)
    train, test = split_sets(temp, 0, test_prop=0.1)
    train = downsample_df(train, 'CA_CCHD')
    cchd = cchd.append(train)  
    test_cchd = test_cchd.append(test)




chd=cchd.copy()

chd_test = test_cchd.copy()



chd = mlp_all_of_the_above(chd,chd,'CA_CCHD')

chd['lrg_miss_imp'] = chd.FAGECOMB_IMP | chd.FRACEHISP_IMP | chd.ILOP_R_IMP | chd.ILP_R_IMP | chd.FEDUC_IMP
chd.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd = LabelEncoding(chd)

chd = add_random_column_to_df(chd)

target = 'CA_CCHD'

chd_test = chd_test.loc[(chd_test[target]=='Y')|(chd_test[target]=='N'),:]

chd_test = mlp_all_of_the_above(chd_test,chd,'CA_CCHD')

chd_test['lrg_miss_imp'] = chd_test.FAGECOMB_IMP | chd_test.FRACEHISP_IMP | chd_test.ILOP_R_IMP | chd_test.ILP_R_IMP | chd_test.FEDUC_IMP
chd_test.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd_test = LabelEncoding(chd_test)

chd_test = add_random_column_to_df(chd_test)



#PDIAB
chd = chd[chd['RF_PDIAB'] == 1]
chd_test = chd_test[chd_test['RF_PDIAB'] == 1]





#test train split
target = 'CA_CCHD'
X_train = chd.drop(target, axis=1)
y_train = chd[target]

X_test = chd_test.drop(target, axis=1)
y_test = chd_test[target]

#RANDOM FOREST INITIAL FIT
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(class_weight={0:1,1:1700})
randomForest.set_params(random_state=0)
randomForest.fit(X_train, y_train) 
print("The training error is: %.5f" % (1 - randomForest.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - randomForest.score(X_test, y_test)))

# set the parameter grid
score_method = 'precision'

import sklearn.model_selection as ms
grid_para_forest = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 16),
    'n_estimators': range(10, 70, 5)
}

# GRID SEARCH
grid_search_forest = ms.GridSearchCV(randomForest, grid_para_forest, scoring=score_method, cv=10, n_jobs=-1,)
%time grid_search_forest.fit(X_train, y_train)

#Best Params and Score
print(grid_search_forest.best_params_)
print('best score:', grid_search_forest.best_score_)

# get the training/test errors
print("The training error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_test, y_test)))

#list of feature importance
feature_importance = list(zip(X_train.columns, randomForest.feature_importances_))
# dtype = [('feature', 'S10'), ('importance', 'float')]
# feature_importance = np.array(feature_importance, dtype=dtype)
# feature_sort = np.sort(feature_importance, order='importance')[::-1]

#Sorting feature importance
# sorted_features = pd.DataFrame(sorted(feature_importance, key=lambda x: x[1], reverse=True))
sorted_features = pd.DataFrame((list(t) for t in feature_importance),columns = ['feature','importance']
                              ).sort_values('importance',ascending = False).reset_index().drop(columns='index')
print(sorted_features)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
%time cm = confusion_matrix(y_test, grid_search_forest.best_estimator_.predict(X_test))

print(score_method, ': ', cm[1,1]/sum(cm[:,1])*100)
print(cm)

###Hypertension model

#pull selected variables from 2016-2018 databases in SQL and append to a single dataframe
query18 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2018"

query17 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2017"

query16 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc.us2016"


queries = [query18, query17, query16]
            
cchd = pd.DataFrame()
test_cchd = pd.DataFrame()

for query in queries:
    temp = create_table_from_SQL('root','cdc',sql_pw, query)
    train, test = split_sets(temp, 0, test_prop=0.1)
    train = downsample_df(train, 'CA_CCHD')
    cchd = cchd.append(train)  
    test_cchd = test_cchd.append(test)



chd=cchd.copy()

chd_test = test_cchd.copy()



chd = mlp_all_of_the_above(chd,chd,'CA_CCHD')

chd['lrg_miss_imp'] = chd.FAGECOMB_IMP | chd.FRACEHISP_IMP | chd.ILOP_R_IMP | chd.ILP_R_IMP | chd.FEDUC_IMP
chd.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd = LabelEncoding(chd)

chd = add_random_column_to_df(chd)

target = 'CA_CCHD'

chd_test = chd_test.loc[(chd_test[target]=='Y')|(chd_test[target]=='N'),:]

chd_test = mlp_all_of_the_above(chd_test,chd,'CA_CCHD')

chd_test['lrg_miss_imp'] = chd_test.FAGECOMB_IMP | chd_test.FRACEHISP_IMP | chd_test.ILOP_R_IMP | chd_test.ILP_R_IMP | chd_test.FEDUC_IMP
chd_test.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd_test = LabelEncoding(chd_test)

chd_test = add_random_column_to_df(chd_test)



#PHYPE
chd = chd[chd['RF_PHYPE'] == 1]
chd_test = chd_test[chd_test['RF_PHYPE'] == 1]





#test train split
target = 'CA_CCHD'
X_train = chd.drop(target, axis=1)
y_train = chd[target]

X_test = chd_test.drop(target, axis=1)
y_test = chd_test[target]

#RANDOM FOREST INITIAL FIT
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(class_weight={0:1,1:1700})
randomForest.set_params(random_state=0)
randomForest.fit(X_train, y_train) 
print("The training error is: %.5f" % (1 - randomForest.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - randomForest.score(X_test, y_test)))

# set the parameter grid
score_method = 'precision'

import sklearn.model_selection as ms
grid_para_forest = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 16),
    'n_estimators': range(10, 70, 5)
}

# GRID SEARCH
grid_search_forest = ms.GridSearchCV(randomForest, grid_para_forest, scoring=score_method, cv=10, n_jobs=-1,)
%time grid_search_forest.fit(X_train, y_train)

#Best Params and Score
print(grid_search_forest.best_params_)
print('best score:', grid_search_forest.best_score_)

# get the training/test errors
print("The training error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_train, y_train)))
print("The test     error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_test, y_test)))

#list of feature importance
feature_importance = list(zip(X_train.columns, randomForest.feature_importances_))
# dtype = [('feature', 'S10'), ('importance', 'float')]
# feature_importance = np.array(feature_importance, dtype=dtype)
# feature_sort = np.sort(feature_importance, order='importance')[::-1]

#Sorting feature importance
# sorted_features = pd.DataFrame(sorted(feature_importance, key=lambda x: x[1], reverse=True))
sorted_features = pd.DataFrame((list(t) for t in feature_importance),columns = ['feature','importance']
                              ).sort_values('importance',ascending = False).reset_index().drop(columns='index')
print(sorted_features)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
%time cm = confusion_matrix(y_test, grid_search_forest.best_estimator_.predict(X_test))

print(sum(y_test == 1)/len(y_test))
print(score_method, ': ', cm[1,1]/sum(cm[:,1])*100)
print(cm)

print(sum(y_test == 1)/len(y_test)*100)

