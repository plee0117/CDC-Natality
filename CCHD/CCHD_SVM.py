import Functions.py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
pd.set_option('display.max_columns', 100)
from sklearn import linear_model
from sklearn import metrics
from statsmodels.discrete.discrete_model import LogitResults
from statsmodels.discrete.discrete_model import Logit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from statsmodels.tools.tools import add_constant
import re

#Data processing

### dataset for pre-pregnancy diabetes population, all CCHD cases with non-CCHD observed downsampled at 10:1 ratio
query18 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
            FROM cdc_project.cdc_2018_full\
            WHERE RF_PDIAB = 'Y'"

query17 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
            FROM cdc_project.cdc_2017_full\
            WHERE RF_PDIAB = 'Y'"

query16 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
            FROM cdc_project.cdc_2016_full\
            WHERE RF_PDIAB = 'Y'"

queries = [query18, query17, query16]
            
cchd = pd.DataFrame()
test_cchd = pd.DataFrame()

for query in queries:
    temp = create_table_from_SQL('root','cdc_project','*****', query)
    train, test = split_sets(temp, 0, test_prop=0.1)
    train = downsample_df_r(train, 'CA_CCHD', 10)
    cchd = cchd.append(train)  
    test_cchd = test_cchd.append(test)

cchd.to_csv('Datasets/baseline_train_pdiab.csv')
test_cchd.to_csv('Datasets/baseline_test_pdiab.csv')

test_cchd = pd.read_csv('Datasets/baseline_test_pdiab.csv')
cchd = pd.read_csv('Datasets/baseline_train_pdiab.csv')
cchd.drop('Unnamed: 0',axis=1, inplace = True)
test_cchd.drop('Unnamed: 0',axis=1, inplace = True)

#new var set, new missing set, RF_PDIAB gone
variables = {'nominal_categorical':['MBSTATE_REC','MRACEHISP','MAR_P','DMAR','MEDUC','FRACEHISP',\
                                    'FEDUC','WIC','RF_GDIAB','RF_PHYPE','RF_GHYPE',\
                                    'RF_EHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR',\
                                  'IP_GON','IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'PAY', 'SEX'],\
           'ordinal_categorical':['PRECARE', 'DOB_MM'],\
           'continuous':['MAGER', 'FAGECOMB','PRIORTERM','PRIORLIVE','PRIORDEAD','LBO_REC','TBO_REC',\
                         'ILLB_R','ILOP_R','ILP_R','PREVIS','CIG_0','CIG_1','CIG_2','CIG_3','M_Ht_In','BMI',\
                         'WTGAIN','RF_CESARN','OEGest_Comb'],\
            'target':['CA_CCHD']}

#reload in processed dataframe
#load in dataframe
cchd_pdiab = pd.read_csv('Datasets/cchd_pdab_allimp.csv')
cchd_pdiab_test = pd.read_csv('Datasets/cchd_pdab_allimp_test.csv')

#redefine variable dictionary

#create train set
cchd_pdiab2 = cchd_pdiab.copy()
cchd_pdiab2 = dummify_columns(cchd_pdiab2, variables['nominal_categorical_ndummified'])
Xpd,ypd = xy_split(cchd_pdiab2, 'CA_CCHD')
Xpd = standardize_columns(Xpd, list(Xpd.columns))

#create validation set
cchd_pdiab_test2 = cchd_pdiab_test.copy()
cchd_pdiab_test2 = dummify_columns(cchd_pdiab_test2, variables['nominal_categorical_ndummified'])
Xpd_test,ypd_test = xy_split(cchd_pdiab_test2, 'CA_CCHD')
Xpd_test = standardize_columns(Xpd_test, list(Xpd_test.columns))

#grid search SVM for accuracy
svm_model = svm.SVC(class_weight='balanced')
grid_para_svm = [
    {'C': [0.001, 0.1, 1, 10, 100, 1000],
     'kernel': ['poly'],
     'degree': [1, 2, 3]},
    {'C': [1, 10, 100, 1000],
     'gamma': [0.001, 0.0001],
     'kernel': ['rbf']}
]
grid_search_svm = GridSearchCV(svm_model, grid_para_svm, scoring='accuracy', cv=3, return_train_score=True,  n_jobs=-1)
grid_search_svm.fit(Xpd, ypd)
print(grid_search_svm.best_params_)
print(grid_search_svm.best_estimator_.score(Xpd,ypd))

#results on validation set - the results are terrible
print(grid_search_svm.best_estimator_.score(Xpd_test,ypd_test))
print(precision(ypd_test, grid_search_svm.best_estimator_.predict(Xpd_test)))
confusion_matrix(ypd_test, grid_search_svm.best_estimator_.predict(Xpd_test))

#grid search SVM for precision
svm_modelp = svm.SVC(class_weight='balanced')
grid_para_svmp = [
    {'C': [1, 10, 100, 1000],
     'kernel': ['poly'],
     'degree': [1, 2, 3]},
    {'C': [1, 10, 100, 1000],
     'gamma': [0.001, 0.0001],
     'kernel': ['rbf']}
]
grid_search_svmp = GridSearchCV(svm_modelp, grid_para_svmp, scoring='precision', cv=3, return_train_score=True,  n_jobs=-1)
grid_search_svmp.fit(Xpd, ypd)
print(grid_search_svmp.best_params_)
print(grid_search_svmp.best_estimator_.score(Xpd,ypd))

print(grid_search_svmp.best_estimator_.score(Xpd_test,ypd_test))
print(precision(ypd_test, grid_search_svmp.best_estimator_.predict(Xpd_test)))

confusion_matrix(ypd_test, grid_search_svmp.best_estimator_.predict(Xpd_test))



