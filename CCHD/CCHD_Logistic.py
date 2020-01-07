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

# begin dictionary of columns to analyze for CCHD - includes pre-pregnancy and gestational features
# features re: delivery and labor are not useful for this use case
variables = {'nominal_categorical':['MBSTATE_REC','MRACEHISP','MAR_P','DMAR','MEDUC','FRACEHISP',\
                                    'FEDUC','WIC','RF_PDIAB','RF_GDIAB','RF_PHYPE','RF_GHYPE',\
                                    'RF_EHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR',\
                                  'IP_GON','IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'PAY', 'SEX'],\
           'ordinal_categorical':['PRECARE', 'DOB_MM'],\
           'continuous':['MAGER', 'FAGECOMB','PRIORTERM','PRIORLIVE','PRIORDEAD','LBO_REC','TBO_REC',\
                         'ILLB_R','ILOP_R','ILP_R','PREVIS','CIG_0','CIG_1','CIG_2','CIG_3','M_Ht_In','BMI',\
                         'WTGAIN','RF_CESARN','OEGest_Comb'],\
            'target':['CA_CCHD']}

vrs = variables['nominal_categorical']+variables['ordinal_categorical']+variables['continuous']+variables['target']
",".join(vrs)

#pull selected variables from 2016-2018 databases in SQL and append to a single dataframe
query18 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc_project.cdc_2018_full"

query17 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc_project.cdc_2017_full"

query16 = "SELECT MBSTATE_REC,MRACEHISP,MAR_P,DMAR,MEDUC,FRACEHISP,FEDUC,WIC,RF_PDIAB,RF_GDIAB,RF_PHYPE,\
                RF_GHYPE,RF_EHYPE,RF_PPTERM,RF_INFTR,RF_FEDRG,RF_ARTEC,RF_CESAR,IP_GON,IP_SYPH,IP_CHLAM,\
                IP_HEPB,IP_HEPC,PAY,SEX,PRECARE,DOB_MM,MAGER,FAGECOMB,PRIORTERM,PRIORLIVE,PRIORDEAD,\
                LBO_REC,TBO_REC,ILLB_R,ILOP_R,ILP_R,PREVIS,CIG_0,CIG_1,CIG_2,CIG_3,M_Ht_In,BMI,WTGAIN,\
                RF_CESARN,OEGest_Comb,CA_CCHD\
         FROM cdc_project.cdc_2016_full"

queries = [query18, query17, query16]
            
cchd = pd.DataFrame()
test_cchd = pd.DataFrame()

for query in queries:
    temp = create_table_from_SQL('root','cdc_project','******', query)
    train, test = split_sets(temp, 0, test_prop=0.1)
    train = downsample_df(train, 'CA_CCHD')
    cchd = cchd.append(train)  
    test_cchd = test_cchd.append(test)

#write out baseline datasets
cchd.to_csv('Datasets/baseline_train.csv')
test_cchd.to_csv('Datasets/baseline_test.csv')

#develop baseline dataset with variables cleaned and imputed apporpriately
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

#read in baseline
test_cchd = pd.read_csv('Datasets/baseline_test.csv')
cchd = pd.read_csv('Datasets/baseline_train.csv')

#clean and process
cchd_all_imp = mlp_all_of_the_above(cchd,cchd,'CA_CCHD')
cchd_all_imp_test = mlp_all_of_the_above(test_cchd,cchd,'CA_CCHD')

#write out processed versions
cchd_all_imp.to_csv('Datasets/cchd_all_imputed_colfixed.csv')
cchd_all_imp_test.to_csv('Datasets/cchd_allimp_test.csv')

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

missing_dictd = {'cont9': ['LBO_REC', 'TBO_REC'],\
                'cont99': ['FAGECOMB', 'PRIORTERM','PRIORLIVE', 'PRIORDEAD', 'PRECARE', 'PREVIS',\
                         'CIG_0', 'CIG_1', 'CIG_2', 'CIG_3', 'M_Ht_In', 'WTGAIN', 'RF_CESARN', 'OEGest_Comb'],\
                'cont999':['ILLB_R', 'ILP_R', 'ILOP_R'],\
                'cont99.9': ['BMI'],\
                'cat3': ['MBSTATE_REC'],\
                'cat8': ['MRACEHISP'],\
                'cat9': ['MEDUC', 'FEDUC', 'PAY', 'FRACEHISP', 'DMAR'],\
                'catU': ['WIC','RF_GDIAB','RF_PHYPE',\
                        'RF_GHYPE','RF_EHYPE','RF_PPTERM','RF_INFTR','RF_FEDRG','RF_ARTEC','RF_CESAR','IP_GON',\
                        'IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'MAR_P']}
missing_valsd = [9,99,999,99.9,3,8,9,'U']

cchd_dp = mlp_all_of_the_above_r(cchd,cchd,'CA_CCHD', missing_valsd, missing_dictd)
test_cchd_dp = mlp_all_of_the_above_r(test_cchd,cchd,'CA_CCHD', missing_valsd, missing_dictd)
cchd_dp.to_csv('Datasets/cchd_pdab_allimp.csv', index=False)
test_cchd_dp.to_csv('Datasets/cchd_pdab_allimp_test.csv', index=False)

###models on baseline datasets

#redefine variable dictionary
variables = {'nominal_categorical_ndummified':['MBSTATE_REC','MRACEHISP','MAR_P','DMAR','MEDUC','FRACEHISP',\
                                    'FEDUC','WIC','RF_PDIAB','RF_GDIAB','RF_PHYPE','RF_GHYPE',\
                                    'RF_EHYPE','RF_PPTERM','RF_FEDRG','RF_ARTEC','DOB_MM',\
                                  'IP_GON','IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'PAY', 'SEX'],\
             'nominal_categorical_dummified': ['lrg_miss_imp'],\
           'continuous':['PRECARE','MAGER', 'FAGECOMB','PRIORTERM','PRIORLIVE','PRIORDEAD',\
                         'ILLB_R','ILOP_R','PREVIS','CIG_0','CIG_1','M_Ht_In','BMI',\
                         'WTGAIN','RF_CESARN','OEGest_Comb'],\
            'target':['CA_CCHD']}

#load in and prepare train and validation dataframes
cchd = pd.read_csv('Datasets/cchd_all_imputed_colfixed.csv')
cchd = cchd.drop('Unnamed: 0', axis=1)
cchd_test = pd.read_csv('Datasets/cchd_allimp_test.csv')
cchd_test = cchd_test.drop('Unnamed: 0', axis=1)
cchd_test = dummify_columns(cchd_test, variables['nominal_categorical_ndummified'])
X_test,y_test = xy_split(cchd_test, 'CA_CCHD')
X_test_standardized = standardize_columns(X_test,list(X_test.columns))
X_test_standardized_constant = add_constant(X_test_standardized)
X_test_constant = add_constant(X_test)

### Basic non-regularized model for feature significance - statsmodels

#run non-regularized model with all features
#output all siginificant features ordered by absolute coefficient value
f_logit_b, coefs_b = basic_significance(cchd, variables['nominal_categorical_ndummified'], 'CA_CCHD')
coefs_b
print('precision: %s' %(precision(y_test, np.round(f_logit_b.predict(X_test_constant)))))

### find best model via forward selection - statsmodels
forward_models = forward_selection(cchd, 'CA_CCHD', variables['nominal_categorical_ndummified'], criteria='bic')

#view coefficients and accuracy on best forward selected model
best = best_forward_set(forward_models)
cchd2 = cchd[best+['CA_CCHD']]
f_logit_bf, coefs_bf= basic_significance(cchd2,['RF_PDIAB', 'RF_GDIAB', 'MRACEHISP', 'RF_GHYPE'], 'CA_CCHD')
coefs_bf['Odds Ratio'] = np.exp(coefs_bf['coefs'])
coefs_bf

#prepare validation dataset with only 'best' feature set from forward selection
cchd_test = pd.read_csv('Datasets/cchd_allimp_test.csv')
cchd_test = cchd_test.drop('Unnamed: 0', axis=1)
cchd_test_best = cchd_test[best+['CA_CCHD']]
cchd_test_best = dummify_columns(cchd_test_best, ['RF_PDIAB', 'RF_GDIAB', 'MRACEHISP', 'RF_GHYPE'])
X_test_best,y_test_garbage = xy_split(cchd_test_best, 'CA_CCHD')
X_test_best_standardized = standardize_columns(X_test_best,list(X_test_best.columns))
X_test_best_standardized_constant = add_constant(X_test_best_standardized)
X_test_best_constant = add_constant(X_test_best)

#view precision on validation set
print('precision: %s' %(precision(y_test, np.round(f_logit_bf.predict(X_test_best_constant)))))

#grid search model for prediction with sklearn - all predictors
params = {'C':np.logspace(-4,4, 20)}
acc, params, coefs, b_estimator = grid_search_logit(cchd, variables['nominal_categorical_ndummified'], 'CA_CCHD', params, standardize = 'N')
print('test accracy: %s' %(acc))
print('params: %s' %(params))
print('validation accuracy: %s' %(b_estimator.score(X_test, y_test)))
print('precision: %s' %(precision(y_test, b_estimator.predict(X_test))))
coefs

#grid search model for prediction with sklearn - all predictors (standardized)
params = {'C':np.logspace(-4,4, 20)}
accs, paramss, coefss, b_estimators = grid_search_logit(cchd, variables['nominal_categorical_ndummified'], 'CA_CCHD', params, standardize = 'Y')
print('test accracy: %s' %(accs))
print('params: %s' %(paramss))
print('validation accuracy: %s' %(b_estimators.score(X_test_standardized, y_test)))
print('precision: %s' %(precision(y_test, b_estimators.predict(X_test_standardized))))
coefs

### grid search model with 'best' feature set

#accuracy is slightly better with forward selected set of variables
#aic used as goal here is prediction oriented
#much less regularization is perfmormed in optimal mode with forward selected set of variables
forward_aic = forward_selection(cchd, 'CA_CCHD', variables['nominal_categorical_ndummified'], criteria='aic')
best_aic = best_forward_set(forward_aic)
best_aic
best_aic[-1]='CIG_0'
cchd3 = cchd[best_aic+['CA_CCHD']]
dummy = [var for var in best_aic if var in variables['nominal_categorical_ndummified']]
params = {'C':np.logspace(-4,4, 20)}
acc2, params2, coefs2 = grid_search_logit(cchd3,dummy, 'CA_CCHD', params, standardized = 'N')
print('test accracy: %s' %(acc2))
print('params: %s' %(params2))
coefs2

#same model using standardized features
forward_aic = forward_selection(cchd, 'CA_CCHD', variables['nominal_categorical_ndummified'], criteria='aic')
best_aic = best_forward_set(forward_aic)
best_aic
best_aic[-1]='CIG_0'
best_aic[10]='CIG_1'
cchd3 = cchd[best_aic+['CA_CCHD']]
dummy = [var for var in best_aic if var in variables['nominal_categorical_ndummified']]
params = {'C':np.logspace(-4,4, 20)}
acc2, params2, coefs2 = grid_search_logit(cchd3,dummy, 'CA_CCHD', params, standardize='Y')
print('test accracy: %s' %(acc2))
print('params: %s' %(params2))
coefs2

### Modeling for population with pre-pregnancy diabetes only (10:1 downsampling)

#variable set with PF_DIAB removed
variables = {'nominal_categorical_ndummified':['MBSTATE_REC','MRACEHISP','MAR_P','DMAR','MEDUC','FRACEHISP',\
                                    'FEDUC','WIC','RF_GDIAB','RF_PHYPE','RF_GHYPE',\
                                    'RF_EHYPE','RF_PPTERM','RF_FEDRG','RF_ARTEC','DOB_MM',\
                                  'IP_GON','IP_SYPH','IP_CHLAM','IP_HEPB','IP_HEPC', 'PAY', 'SEX'],\
             'nominal_categorical_dummified': ['lrg_miss_imp'],\
           'continuous':['PRECARE','MAGER', 'FAGECOMB','PRIORTERM','PRIORLIVE','PRIORDEAD',\
                         'ILLB_R','ILOP_R','PREVIS','CIG_0','CIG_1','M_Ht_In','BMI',\
                         'WTGAIN','RF_CESARN','OEGest_Comb'],\
            'target':['CA_CCHD']}

cchd_pdiab = pd.read_csv('Datasets/cchd_pdab_allimp.csv')
cchd_pdiab_test = pd.read_csv('Datasets/cchd_pdab_allimp_test.csv')

#create train set
cchd_pdiab2 = cchd_pdiab.copy()
cchd_pdiab2 = dummify_columns(cchd_pdiab2, variables['nominal_categorical_ndummified'])
Xpd,ypd = xy_split(cchd_pdiab2, 'CA_CCHD')

#create validation set
cchd_pdiab_test2 = cchd_pdiab_test.copy()
cchd_pdiab_test2 = dummify_columns(cchd_pdiab_test2, variables['nominal_categorical_ndummified'])
Xpd_test,ypd_test = xy_split(cchd_pdiab_test2, 'CA_CCHD')

#grid search model with all features and class weight = balanced
params = {'C':np.logspace(-4,4, 20)}
acc, params, coefs, b_estimator = grid_search_logitw(cchd_pdiab, variables['nominal_categorical_ndummified'], 'CA_CCHD', params, standardize = 'N')
print('test accracy: %s' %(acc))
print('params: %s' %(params))
print('validation accuracy: %s' %(b_estimator.score(Xpd_test, ypd_test)))
print('precision: %s' %(precision(ypd_test, b_estimator.predict(Xpd_test))))
coefs

confusion_matrix(ypd_test, b_estimator.predict(Xpd_test))
