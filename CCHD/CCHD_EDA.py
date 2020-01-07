import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import mysql.connector
import seaborn as sns
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats.contingency_tables import Table
from scipy.stats import chi2_contingency
pd.set_option('display.max_columns', 100)
from sklearn import linear_model
from sklearn import metrics
from statsmodels.discrete.discrete_model import LogitResults
from statsmodels.discrete.discrete_model import Logit
import Functions.py

#pull in table with all abnormalities for patients admitted to the NICU
query = "SELECT CA_ANEN, CA_MNSB, CA_CCHD, CA_CDH, CA_OMPH,\
        CA_GAST, CA_LIMB, CA_CLEFT, CA_DOWN, CA_DISOR, CA_HYPO, AB_SEIZ\
        FROM cdc_2018_full\
        WHERE AB_NICU = 'Y'"

nicu_defect = create_table_from_SQL('root','cdc_project','******', query)

#plot total number of patients admitted to the NICU for each reported abnormality
#correct to detect 'X' for down and cleft
nicu_defect.apply(lambda x: sum(x == 'Y'), axis=0).sort_values(ascending=False).plot.bar()
plt.title('Total Number of Patients Admitted to the NICU')

query = "SELECT CA_ANEN, CA_MNSB, CA_CCHD, CA_CDH, CA_OMPH,\
        CA_GAST, CA_LIMB, CA_CLEFT, CA_DOWN, CA_DISOR, CA_HYPO, AB_SEIZ, AB_NICU\
        FROM cdc_2018_full"

#pull in table with all observations for abnormalities and NICU admission
defect_nicu_props = create_table_from_SQL('root','cdc_project','******', query)

#find overall rate at which patients are admitted to the NICU
overall_prop=len(defect_nicu_props.AB_NICU[defect_nicu_props.AB_NICU=='Y'])/len(defect_nicu_props.AB_NICU)

#create dataframe with admitted proportion per abnormality
admits = pd.DataFrame(defect_nicu_props[defect_nicu_props.AB_NICU=='Y'].apply(lambda x: sum(x=='Y'), axis=0))
total = defect_nicu_props.apply(lambda x: sum(x=='Y'), axis=0)
admits = pd.concat([admits,total], axis=1)
admits.columns = ['Admits', 'Total']
admits['props'] = admits['Admits']/admits['Total']
admits = admits.iloc[0:-1,:]
admits = admits.fillna(value=0)
admits.sort_values(by='props', ascending = False, inplace = True)

#plot the dataframe
plt.bar(x=admits.index, height = admits.props)
plt.xticks(rotation=45)
plt.axhline(y=overall_prop, color='blue', label='overall NICU admit')
plt.legend(("Proportion admitted overall", "Proportion admitted defect"))
plt.title('Proportion of Patients with Defect Admitted to the NICU')

#create 2x2 plot showing proportion and total cases
#plot indicates that analysis should be focused on CA_CCHD as highest urgency abnormality
#literature review confirms unmet need and mostly unknown etiology
plt.scatter(x=admits.Admits, y=admits.props)
for i, txt in enumerate(admits.index):
    plt.annotate(txt, (admits.Admits[i], admits.props[i]))
plt.axhline(y=0.5, color='blue')
plt.axvline(x=admits['Admits']['CA_CCHD']/2, color='blue')
plt.title('Proportion and Total Admits Two by Two')

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

#create string of variables for SQL query
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

cchd.CA_CCHD = np.where(cchd.CA_CCHD=='Y',1,0)

#convert continuous variables to float
#update to split out ordinal categorical separately
for x in variables['continuous']:
    cchd[x]=cchd[x].astype('float')

#change true nulls to fit missingness definitions already in the dataset
cchd.isnull().sum()
cchd.MAR_P = cchd.MAR_P.fillna(value='U')
cchd.DMAR = cchd.DMAR.fillna(value=9)

#combine FRACEHISP unknowns columns
cchd.FRACEHISP = cchd.FRACEHISP.replace(8,9)

#assign 'X' to 'N' for RF_FEDRG RF_ARTEC and 'Y' for MAR_P since paternity assumed for married
for x in ['RF_FEDRG', 'RF_ARTEC']:
    cchd[x].replace('X','N', inplace=True)
cchd.MAR_P.replace('X','Y', inplace=True)

#view ordinal categorical variables by plotting proportion of CCHD in each category
#based on the plots - DOB_MM should be categorical and PRECARE should be continuous - handled later on
plt.subplot(1,2,1)
cchd.groupby('DOB_MM')['CA_CCHD'].mean().plot.bar()
plt.subplot(1,2,2)
cchd.groupby('PRECARE')['CA_CCHD'].mean().plot.bar()

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

#create table of missingness proportions
missing_props = pd.DataFrame()
for i in range(0,len(missing_vals)):
    temp = cchd.groupby('CA_CCHD')[missing_dict[list(missing_dict.keys())[i]]].apply\
    (lambda x: np.sum(x==missing_vals[i])/(cchd.shape[0]/2))
    missing_props = pd.concat([missing_props, temp], axis=1)   
missing_props

#visualize missingness proportions
#missingness is low, though some is >5% and some differ across cchd groupsb
N = missing_props.iloc[0]
Y = missing_props.iloc[1]
plt.subplot(2,1,1)
plt.bar(x=missing_props.columns, height = N, color='blue')
plt.xticks(rotation=90)
plt.ylim((0,0.2))
plt.legend(['No CCHD'])
plt.subplot(2,1,2)
plt.bar(x=missing_props.columns, height = Y)
plt.xticks(rotation=90)
plt.subplots_adjust(hspace=1)
plt.ylim((0,0.2))
plt.legend(['CCHD'])

#create lists of variables with high missingness vs. low missingness
large_miss = list(missing_props.columns[missing_props.apply(lambda x: sum(x)>0.1, axis=0)])
small_miss = list(missing_props.columns[missing_props.apply(lambda x: sum(x)<0.1, axis=0)])

#sort low missingness categorical variables into types
small_cats = {'cat3': [], 'cat8': [], 'cat9': [], 'catU': []}

for var in small_miss:
    if var in missing_dict['cat3']:
        small_cats['cat3'].append(var) 
    elif var in missing_dict['cat8']:
        small_cats['cat8'].append(var)
    elif var in missing_dict['cat9']:
        small_cats['cat9'].append(var)
    elif var in missing_dict['catU']:
        small_cats['catU'].append(var)           

#mode imputation of categoricals with low missingness
small_vals = [3,8,9,'U']

for i in range(0, len(small_vals)):
    temp_lis = small_cats[list(small_cats.keys())[i]]
    for x in temp_lis:
        major_cat = cchd[x].value_counts().sort_values(ascending=False).index[0]
        cchd[x]=cchd[x].replace(small_vals[i],major_cat) 

#sort low missingness continuous variables
small_conts = {'cont9': [], 'cont99': [], 'cont999': [], 'cont99.9': []}

for var in small_miss:
    if var in missing_dict['cont9']:
        small_conts['cont9'].append(var) 
    elif var in missing_dict['cont99']:
        small_conts['cont99'].append(var)
    elif var in missing_dict['cont999']:
        small_conts['cont999'].append(var)
    elif var in missing_dict['cont99.9']:
        small_conts['cont99.9'].append(var)   

#median imputation of categoricals with low missingness
#statistical significance of relationship with target imnproves on variable by variable basis after 
#median imputation
csmall_vals = [9,99,999,99.9]

for i in range(0, len(csmall_vals)):
    temp_lis = small_conts[list(small_conts.keys())[i]]
    for x in temp_lis:
        cchd[x]=cchd[x].replace(csmall_vals[i],cchd[x].median()) 

#view variables wiht >5% missingness
large_miss

#assign 888 to mean for ILLB_R and ILP_R
for x in ['ILLB_R', 'ILP_R', 'ILOP_R']:
        ser =cchd.loc[cchd[x]==888,'MAGER']*12
        cchd.loc[cchd[x]==888, x] = ser

#Impute FAGECOMB missing vals and store whether column was imputed
def binarize99(x):
    if x==299:
        return 1
    else:
        return 0

cchd['FAGECOMB_IMP'] = cchd.FAGECOMB.apply(lambda x: binarize99(x))
cchd.FAGECOMB.replace(99, cchd.FAGECOMB.median(),inplace = True)

#Impute ILOP_R and ILP_R missing vals and store whether column was imputed
def binarize999(x):
    if x==999:
        return 1
    else:
        return 0
for x in ['ILOP_R', 'ILP_R']:
    cchd[x+'_IMP'] = cchd[x].apply(lambda x: binarize999(x))
for x in ['ILOP_R', 'ILP_R']:
    cchd[x].replace(999,cchd[x].median(), inplace=True)

#Impute FRACEHISP and FEDUC missing vals and store whether column was imputed
def binarize9(x):
    if x==9:
        return 1
    else:
        return 0
for x in ['FRACEHISP', 'FEDUC']:
    cchd[x+'_IMP'] = cchd[x].apply(lambda x: binarize9(x))
for x in ['FRACEHISP', 'FEDUC']:
    cchd[x].replace(9,cchd[x].mode()[0], inplace=True)

#convert nominal categorical variables to category
for x in variables['nominal_categorical']:
    cchd[x]=cchd[x].astype('category')

#convert ordinal categorical variables
cchd.DOB_MM = cchd.DOB_MM.astype('category')
cchd.PRECARE = cchd.PRECARE.astype('float')

#fix MAR_P type
cchd.MAR_P = cchd.MAR_P.astype('category')

#matching the literature, pre-pregnancy cigarette intake looks significant, among other factors
#nothing obvious to drop at first pass, as most show significant differences
continuous_eda(cchd, variables['continuous'], 'CA_CCHD')

#pairwise collinearity investigation
sns.pairplot(cchd[variables['continuous']])

#DMAR, IP_GON, IP_SYPH, IP_CHLAM, IP_HEPB are not significant, but could be due to low sample size
#HEPC not significant but it's borderline
#leave in and model will sort 
categorical_eda(cchd, variables['nominal_categorical'], 'CA_CCHD')

vif_scores_cat = VIF(cchd, variables['nominal_categorical']+['DOB_MM']+['FAGECOMB_IMP',\
                    'FRACEHISP_IMP', 'ILOP_R_IMP', 'ILP_R_IMP', 'FEDUC_IMP'], 'CA_CCHD')

#plot categorical VIFs
from math import log
def plot_log_vifs(vif_list):
    vifs = pd.Series([x[0] for x in vif_list])
    cats = pd.Series([x[1] for x in vif_list])
    vifs.index=cats
    np.log(vifs).sort_values(ascending=False).plot.bar()
    plt.axhline(y=log(5), color='blue')
    plt.title('Log VIF for Categorical Variables')

plot_log_vifs(vif_scores_cat)

#attempt to deal with IMP columns collinearity
cchd2 = cchd.copy()
cchd2['lrg_miss_imp'] = cchd.FAGECOMB_IMP | cchd.FRACEHISP_IMP | cchd.ILOP_R_IMP | cchd.ILP_R_IMP | cchd.FEDUC_IMP
cchd2 = cchd2.drop(['FAGECOMB_IMP','FRACEHISP_IMP', 'ILOP_R_IMP', 'ILP_R_IMP', 'FEDUC_IMP'], axis=1)
vif_scores_cat2 = VIF(cchd2, variables['nominal_categorical']+['DOB_MM', 'lrg_miss_imp'], 'CA_CCHD')

#combining imp columns significantly improves vif
plot_log_vifs(vif_scores_cat2)

#combine imp columns, drop RF_CESAR, drop LBO REC and TBO rec, drop INFRT, CIG_2 and CIG_3
cchd3 = cchd2.drop(['RF_CESAR', 'LBO_REC', 'TBO_REC', 'RF_INFTR', 'CIG_2', 'CIG_3'],axis=1)
new_vars = [x for x in variables['nominal_categorical'] if x not in ['RF_CESAR', 'LBO_REC', 'TBO_REC', 'RF_INFTR', 'CIG_2', 'CIG_3']]
vif_scores_cat3 = VIF(cchd3, new_vars+['DOB_MM', 'lrg_miss_imp'], 'CA_CCHD')
plot_log_vifs(vif_scores_cat3)

continuous_VIF(cchd,'CA_CCHD')

#Remove candidates with high multicollinearity
takeout = ['LBO_REC','TBO_REC', 'RF_CESAR', 'RF_INFTR', 'CIG_2', 'CIG_3', 'ILP_R']
continuous_VIF(cchd,'CA_CCHD',takeout)

