import Functions.py

###Baseline Model

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

target = 'CA_CCHD'

chd = mlp_all_of_the_above(chd,chd,'CA_CCHD')

chd_test = chd_test.loc[(chd_test[target]=='Y')|(chd_test[target]=='N'),:]
chd_test = mlp_all_of_the_above(chd_test,chd,'CA_CCHD')

chd_test.MAR_P = chd_test.MAR_P.replace('U','Y')
chd_test.MAR_P = chd_test.MAR_P.astype('str').astype('category')



chd['lrg_miss_imp'] = chd.FAGECOMB_IMP | chd.FRACEHISP_IMP | chd.ILOP_R_IMP | chd.ILP_R_IMP | chd.FEDUC_IMP
chd.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd = standardize_columns(chd,list(set(chd.select_dtypes(exclude = ['category','object']).columns)))

target = 'CA_CCHD'
dummified_r = dummify_columns(chd,list(set(chd.select_dtypes(include ='category').columns)))
y_train = dummify_columns(dummified_r[[target]],[target])
X_train = dummified_r.loc[:,set(dummified_r.columns) - set([target])]





chd_test['lrg_miss_imp'] = chd_test.FAGECOMB_IMP | chd_test.FRACEHISP_IMP | chd_test.ILOP_R_IMP | chd_test.ILP_R_IMP | chd_test.FEDUC_IMP
chd_test.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd_test = standardize_columns(chd_test,list(chd_test.select_dtypes(exclude = ['category','object']).columns))

target = 'CA_CCHD'
dummified_e = dummify_columns(chd_test,list(set(chd_test.select_dtypes(include ='category').columns)))
y_test = dummify_columns(dummified_e[[target]],[target])
X_test = dummified_e.loc[:,set(dummified_e.columns) - set([target])]



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
y_train_fit = pd.DataFrame([-1 if i == 0 else 1 for i in y_train[list(y_train.columns)[0]]])

lofcv = LocalOutlierFactor(contamination = 0.5, novelty = True)
lof_grid_param = {'n_neighbors': range(20,101,10),}
gsearch = GridSearchCV(lofcv,lof_grid_param,scoring='precision',n_jobs=2,cv=5)
%time gsearch.fit(X_train, y_train_fit)

print(gsearch.best_params_)
print(gsearch.best_score_)
from sklearn.metrics import confusion_matrix
y_test_fit = pd.DataFrame([-1 if i == 0 else 1 for i in y_test[list(y_test.columns)[0]]])
%time cm = confusion_matrix(y_test_fit,gsearch.predict(X_test))
print(cm[1,1]/sum(cm[:,1])*100)
print(cm)

### Diabetes model

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

target = 'CA_CCHD'

chd = mlp_all_of_the_above(chd,chd,'CA_CCHD')

chd_test = chd_test.loc[(chd_test[target]=='Y')|(chd_test[target]=='N'),:]
chd_test = mlp_all_of_the_above(chd_test,chd,'CA_CCHD')

chd_test.MAR_P = chd_test.MAR_P.replace('U','Y')
chd_test.MAR_P = chd_test.MAR_P.astype('str').astype('category')

#PHYPE
chd = chd[chd['RF_PDIAB'] == 'Y']
chd_test = chd_test[chd_test['RF_PDIAB'] == 'Y']

chd['lrg_miss_imp'] = chd.FAGECOMB_IMP | chd.FRACEHISP_IMP | chd.ILOP_R_IMP | chd.ILP_R_IMP | chd.FEDUC_IMP
chd.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd = standardize_columns(chd,list(set(chd.select_dtypes(exclude = ['category','object']).columns)))

target = 'CA_CCHD'
dummified_r = dummify_columns(chd,list(set(chd.select_dtypes(include ='category').columns)))
y_train = dummify_columns(dummified_r[[target]],[target])
X_train = dummified_r.loc[:,set(dummified_r.columns) - set([target])]





chd_test['lrg_miss_imp'] = chd_test.FAGECOMB_IMP | chd_test.FRACEHISP_IMP | chd_test.ILOP_R_IMP | chd_test.ILP_R_IMP | chd_test.FEDUC_IMP
chd_test.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd_test = standardize_columns(chd_test,list(chd_test.select_dtypes(exclude = ['category','object']).columns))

target = 'CA_CCHD'
dummified_e = dummify_columns(chd_test,list(set(chd_test.select_dtypes(include ='category').columns)))
y_test = dummify_columns(dummified_e[[target]],[target])
X_test = dummified_e.loc[:,set(dummified_e.columns) - set([target])]



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
y_train_fit = pd.DataFrame([-1 if i == 0 else 1 for i in y_train[list(y_train.columns)[0]]])

lofcv = LocalOutlierFactor(contamination = 0.5, novelty = True)
lof_grid_param = {'n_neighbors': range(20,101,10),}
gsearch = GridSearchCV(lofcv,lof_grid_param,scoring='precision',n_jobs=2,cv=5)
%time gsearch.fit(X_train, y_train_fit)

print(gsearch.best_params_)
print(gsearch.best_score_)
from sklearn.metrics import confusion_matrix
y_test_fit = pd.DataFrame([-1 if i == 0 else 1 for i in y_test[list(y_test.columns)[0]]])
%time cm = confusion_matrix(y_test_fit,gsearch.predict(X_test))
print(cm[1,1]/sum(cm[:,1])*100)
print(cm)

### Hypertension model

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

target = 'CA_CCHD'

chd = mlp_all_of_the_above(chd,chd,'CA_CCHD')

chd_test = chd_test.loc[(chd_test[target]=='Y')|(chd_test[target]=='N'),:]
chd_test = mlp_all_of_the_above(chd_test,chd,'CA_CCHD')

chd_test.MAR_P = chd_test.MAR_P.replace('U','Y')
chd_test.MAR_P = chd_test.MAR_P.astype('str').astype('category')

#PHYPE
chd = chd[chd['RF_PHYPE'] == 'Y']
chd_test = chd_test[chd_test['RF_PHYPE'] == 'Y']

chd['lrg_miss_imp'] = chd.FAGECOMB_IMP | chd.FRACEHISP_IMP | chd.ILOP_R_IMP | chd.ILP_R_IMP | chd.FEDUC_IMP
chd.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd = standardize_columns(chd,list(set(chd.select_dtypes(exclude = ['category','object']).columns)))

target = 'CA_CCHD'
dummified_r = dummify_columns(chd,list(set(chd.select_dtypes(include ='category').columns)))
y_train = dummify_columns(dummified_r[[target]],[target])
X_train = dummified_r.loc[:,set(dummified_r.columns) - set([target])]



chd_test['lrg_miss_imp'] = chd_test.FAGECOMB_IMP | chd_test.FRACEHISP_IMP | chd_test.ILOP_R_IMP | chd_test.ILP_R_IMP | chd_test.FEDUC_IMP
chd_test.drop(columns = ['FAGECOMB_IMP','FRACEHISP_IMP','ILOP_R_IMP','ILP_R_IMP','FEDUC_IMP'],inplace=True)

chd_test = standardize_columns(chd_test,list(chd_test.select_dtypes(exclude = ['category','object']).columns))

target = 'CA_CCHD'
dummified_e = dummify_columns(chd_test,list(set(chd_test.select_dtypes(include ='category').columns)))
y_test = dummify_columns(dummified_e[[target]],[target])
X_test = dummified_e.loc[:,set(dummified_e.columns) - set([target])]



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
y_train_fit = pd.DataFrame([-1 if i == 0 else 1 for i in y_train[list(y_train.columns)[0]]])

lofcv = LocalOutlierFactor(contamination = 0.5, novelty = True)
lof_grid_param = {'n_neighbors': range(20,101,10),}
gsearch = GridSearchCV(lofcv,lof_grid_param,scoring='precision',n_jobs=2,cv=5)
%time gsearch.fit(X_train, y_train_fit)

print(gsearch.best_params_)
print(gsearch.best_score_)
from sklearn.metrics import confusion_matrix
y_test_fit = pd.DataFrame([-1 if i == 0 else 1 for i in y_test[list(y_test.columns)[0]]])
%time cm = confusion_matrix(y_test_fit,gsearch.predict(X_test))
print(cm[1,1]/sum(cm[:,1])*100)
print(cm)