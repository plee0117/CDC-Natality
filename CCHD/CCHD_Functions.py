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
from sklearn import linear_model
from sklearn import metrics
from statsmodels.discrete.discrete_model import LogitResults
from statsmodels.discrete.discrete_model import Logit

#Processing functions - CCHD:

#function to run simple SQL query from python
def create_table_from_SQL(user, database, password, query):
    '''
    - A function that returns a pandas dataframe from a SQL query in python
    ---------------
    - user: user for your local SQL connection in string format
    - database: schema name where your database is stored in string format
    - password: password to access your local SQL connection in string format
    - query: SQL query in string format; enclose with double quotes and use single quotes
    to designate VARCHAR values within queries; use schema_name.table_name after FROM statement
    '''
    import mysql.connector
    cnx = mysql.connector.connect(user=user, database=database, password=password)
    cursor = cnx.cursor()
    query = query
    cursor.execute(query)
    df = pd.DataFrame(cursor.fetchall())
    df.columns = cursor.column_names
    return df

#function to downsample data:
def downsample_df (df, variable, ratio=1):

    '''
    Remove undefined information on defect presence admissions (defect == 'U'),
    create a binary target vector, and create a "balanced" dataframe
    with all defect cases and matching numbers of randomly selected non-defect cases.
    --------------------
    df: full dataframe
    variable: variable or defect of interest in string format
    ratio: takes an integer, the factor by which to scale the sample where variable = 'Y'
    '''

    # remove unknown class from df
    df_no_unknown = df[df[variable].isin(['Y', 'N'])]

    # Create binary target vector, NICU = yes classified as class 0
    df_y_n = pd.DataFrame(np.where((df_no_unknown[variable] == 'Y'), 0, 1))

    # Get indicies of each class' observations
    index_class0 = np.where(df_y_n == 0)[0]
    index_class1 = np.where(df_y_n == 1)[0]

    # Get numbers of observations in class 0
    n_class0 = len(index_class0)*ratio

    # Randomly sample the same number of observations from class 1 as in class 0, without replacement
    np.random.seed(0)
    index_class1_downsampled = np.random.choice(index_class1, size=n_class0, replace=False)

    # Create dataframes for NICU and downsampled non-NICU
    df_defect = df_no_unknown.iloc[index_class0]
    df_adj_NONdefect = df_no_unknown.iloc[index_class1_downsampled]

    # Append into 1 dataframe
    df_downsampled = df_defect.append(df_adj_NONdefect)

    return df_downsampled

# function to split out holdout test set:
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

def mlp_convert_cont_floats(df):
    #convert continuous variables to float
    #update to split out ordinal categorical separately
    for x in variables['continuous']:
        df[x]=df[x].astype('float')
    return df

def mlp_convert_nom_cat(df):
    #convert nominal categorical variables to category
    for x in variables['nominal_categorical']:
        df[x]=df[x].astype('category')
    return df

def mlp_convert_ord_cat(df):
    #convert ordinal categorical variables
    df.DOB_MM = df.DOB_MM.astype('category')
    df.PRECARE = df.PRECARE.astype('float')
    return df

def mlp_fill_MAR_blanks(df):
    #change true nulls to fit missingness definitions already in the dataset
    df.MAR_P = df.MAR_P.fillna(value='U')
    df.DMAR = df.DMAR.fillna(value=9)
    #df.DMAR.replace('',9, inplace=True) # need to take care of 1 vs '1'   
#     df.DMAR.replace('1',1, inplace=True)
#     df.DMAR.replace('2',2, inplace=True)
    return df

def mlp_reassign_FRACE(df):
    #combine FRACEHISP unknowns columns
    df.FRACEHISP = df.FRACEHISP.replace(8,9)
    return df

def mlp_reassign_X_NA(df):
    #assign 'X' to 'N' for RF_FEDRG RF_ARTEC and 'Y' for MAR_P since paternity assumed for married
    for x in ['RF_FEDRG', 'RF_ARTEC']:
        df[x].replace('X','N', inplace=True)
    df.MAR_P.replace('X','Y', inplace=True)
    return df

def mlp_reassign_ILs(df):
    #assign 888 to mean for ILLB_R and ILP_R
    for x in ['ILLB_R', 'ILP_R', 'ILOP_R']:
        ser = df.loc[df[x]==888,'MAGER']*12
        df.loc[df[x]==888, x] = ser
    return df

def measure_missing(df, target):
    #create table of missingness proportions
    missing_props = pd.DataFrame()
    for i in range(0,len(missing_vals)):
        temp = df.groupby(target)[missing_dict[list(missing_dict.keys())[i]]].apply\
        (lambda x: np.sum(x==missing_vals[i])/(df.shape[0]/2))
        missing_props = pd.concat([missing_props, temp], axis=1)   

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
    return small_conts, small_cats, large_miss

def mlp_impute_s_cat(df,df_w,small_cats):
    #mode imputation of categoricals with low missingness
    small_vals = [3,8,9,'U']
    for i in range(0, len(small_vals)):
        temp_lis = small_cats[list(small_cats.keys())[i]]
        for x in temp_lis:
            major_cat = df_w[x].value_counts().sort_values(ascending=False).index[0]
            df[x]=df[x].replace(small_vals[i],major_cat)
    return df

def mlp_fix_MAR_P(df):
    df.MAR_P.replace('U','Y', inplace=True)
    return df

def mlp_fix_DMAR(df):
    df.DMAR.replace(9,1, inplace=True)
    df.DMAR.replace('9',1, inplace=True)
    return df

def mlp_impute_s_num(df,df_w,small_conts):
    #median imputation of categoricals with low missingness
    #statistical significance of relationship with target imnproves on variable by variable basis after 
    #median imputation
    csmall_vals = [9,99,999,99.9]
    for i in range(0, len(csmall_vals)):
        temp_lis = small_conts[list(small_conts.keys())[i]]
        for x in temp_lis:
            df[x]=df[x].replace(csmall_vals[i],df_w[x].median())
    return df


def binarize9(x):
    if x==9:
        return 1
    else:
        return 0
    
def binarize99(x):
    if x==99:
        return 1
    else:
        return 0

def binarize999(x):
    if x==999:
        return 1
    else:
        return 0
    
def mlp_impute_FAGECOMB(df,df_w):
    #Impute FAGECOMB missing vals and store whether column was imputed    
    df['FAGECOMB_IMP'] = df.FAGECOMB.apply(lambda x: binarize99(x))
    df.FAGECOMB.replace(99, df_w.FAGECOMB.median(),inplace = True)
    return df

def mlp_impute_ILPs(df,df_w):
    #Impute ILOP_R and ILP_R missing vals and store whether column was imputed
    for x in ['ILOP_R', 'ILP_R']:
        df[x+'_IMP'] = df[x].apply(lambda x: binarize999(x))
    for x in ['ILOP_R', 'ILP_R']:
        df[x].replace(999,df_w[x].median(), inplace=True)
    return df

def mlp_impute_FRACE_ED(df,df_w):
    #Impute FRACEHISP and FEDUC missing vals and store whether column was imputed
    for x in ['FRACEHISP', 'FEDUC']:
        df[x+'_IMP'] = df[x].apply(lambda x: binarize9(x))
    for x in ['FRACEHISP', 'FEDUC']:
        df[x].replace(9,df_w[x].mode()[0], inplace=True)
    return df

def mlp_impute_combine(df):
    #Combine imputed flag columns into one
    df['lrg_miss_imp'] = df.FAGECOMB_IMP|df.ILOP_R_IMP|df.ILP_R_IMP|df.FRACEHISP_IMP|df.FEDUC_IMP
    df = df.drop(['FAGECOMB_IMP', 'ILOP_R_IMP', 'ILP_R_IMP','FRACEHISP_IMP','FEDUC_IMP'],axis=1)
    return df

def mlp_drop_highcoll(df):
    df = df.drop(['RF_CESAR', 'LBO_REC', 'TBO_REC', 'RF_INFTR', 'CIG_2', 'CIG_3', 'ILP_R'],axis=1)
    return df

def binarize_target(df,target):
    df[target]=np.where(df[target]=='Y',1,0)
    return df

def mlp_all_of_the_above(df,df_w,target):
    df = mlp_fill_MAR_blanks(df)
    df = mlp_reassign_FRACE(df)
    df = mlp_reassign_X_NA(df)
    df = mlp_reassign_ILs(df)
    df = mlp_convert_cont_floats(df)
    small_conts, small_cats, large_miss = measure_missing(df,target)
    df = mlp_impute_s_cat(df,df_w,small_cats)
    df = mlp_fix_MAR_P(df)
    df = mlp_fix_DMAR(df)
    df = mlp_impute_s_num(df,df_w,small_conts)
    df = mlp_impute_FAGECOMB(df,df_w)
    df = mlp_impute_ILPs(df,df_w)
    df = mlp_impute_FRACE_ED(df,df_w)
    df = binarize_target(df, target)
    df = mlp_convert_nom_cat(df)
    df = mlp_convert_ord_cat(df)
    df = mlp_drop_highcoll(df)
    df = mlp_impute_combine(df)
    return df

def calc_prop(df, val, var):
    return (len(df[df[var]==val][var])/df.shape[0],var)

#version of measure missing function for imbalanced dataset
def measure_missing_r(df, missing_vals, missing_dict):
    #create table of missingness proportions
    all_props = []
    for i in range(0,len(missing_vals)):
        lis = missing_dict[list(missing_dict.keys())[i]]
        props = list(map(lambda x: calc_prop(df=df, val=missing_vals[i], var=x), lis))
        all_props+=props

    #create lists of variables with high missingness vs. low missingness
    large_miss = list(filter(lambda x: x[0]>0.05, all_props))
    small_miss = list(filter(lambda x: x[0]<0.05, all_props))
    
    #sort low missingness categorical variables into types
    small_cats = {'cat3': [], 'cat8': [], 'cat9': [], 'catU': []}

    for var in small_miss:
        if var[1] in missing_dict['cat3']:
            small_cats['cat3'].append(var[1]) 
        elif var[1] in missing_dict['cat8']:
            small_cats['cat8'].append(var[1])
        elif var[1] in missing_dict['cat9']:
            small_cats['cat9'].append(var[1])
        elif var[1] in missing_dict['catU']:
            small_cats['catU'].append(var[1])    


    #sort low missingness continuous variables
    small_conts = {'cont9': [], 'cont99': [], 'cont999': [], 'cont99.9': []}

    for var in small_miss:
        if var[1] in missing_dict['cont9']:
            small_conts['cont9'].append(var[1]) 
        elif var[1] in missing_dict['cont99']:
            small_conts['cont99'].append(var[1])
        elif var[1] in missing_dict['cont999']:
            small_conts['cont999'].append(var[1])
        elif var[1] in missing_dict['cont99.9']:
            small_conts['cont99.9'].append(var[1])
    
    large_miss = list(map(lambda x: x[1], large_miss))
    return small_conts, small_cats, large_miss

#version of mlp_all_of_the_above for imbalanced dataset
def mlp_all_of_the_above_r(df,df_w,target, missing_vals, missing_dict):
    df = mlp_fill_MAR_blanks(df)
    df = mlp_reassign_FRACE(df)
    df = mlp_reassign_X_NA(df)
    df = mlp_reassign_ILs(df)
    df = mlp_convert_cont_floats(df)
    small_conts, small_cats, large_miss = measure_missing_r(df, missing_vals, missing_dict)
    df = mlp_impute_s_cat(df,df_w,small_cats)
    df = mlp_fix_MAR_P(df)
    df = mlp_fix_DMAR(df)
    df = mlp_impute_s_num(df,df_w,small_conts)
    df = mlp_impute_FAGECOMB(df,df_w)
    df = mlp_impute_ILPs(df,df_w)
    df = mlp_impute_FRACE_ED(df,df_w)
    df = binarize_target(df, target)
    df = mlp_convert_nom_cat(df)
    df = mlp_convert_ord_cat(df)
    df = mlp_drop_highcoll(df)
    df = mlp_impute_combine(df)
    return df

def add_miss_cols(df, missing_cols, df2):
    for var in missing_cols:
        df[var] = np.zeros(shape=(df.shape[0],1))
    df=df[df2.columns]
    return df

# EDA Functions - CCHD

def continuous_eda(df, var_list, target_var):
    '''
    - A function that analyzes the relationship of a continuous variable to an abnormality target variable
    - Returns overlapping histograms for Y/N target variable groups and a bar graph with means for
    Y/N groups side-by-side. Two-way t-test results are returned underneath the graph.
    ---------------
    - df: the dataframe containing variables of interest
    - var_list: a list of continuous variables as strings
    - target_var: the target abnormality
    '''
    for x in var_list:
        a = df[df[target_var]==1][x]
        b = df[df[target_var]==0][x]
        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(a, color='red',alpha=0.3)
        plt.hist(b, color='blue', alpha=0.3)
        plt.figtext(0,0,stats.ttest_ind(a,b))
        plt.title('dist '+x)
        plt.subplot(1,2,2)
        plt.bar('Y', np.mean(a), alpha=0.3)
        plt.bar('N', np.mean(b), alpha=0.3)
        plt.title('mean '+x)

def categorical_eda(df, var_list, target_var):
    '''
    - A function that analyzes the relationship of a categorical variable to an abnormality target variable
    - Returns mosaic plot with chi-square test results underneath
    ---------------
    - df: the dataframe containing variables of interest
    - var_list: a list of categorical variables as strings
    - target_var: the target abnormality
    '''
    for x in var_list:
        m = pd.crosstab(df[x], df[target_var])
        plt.figure()
        mosaic(m.stack(), gap=0.05)
        plt.title('mosaic'+x)
        plt.figtext(0,0,stats.chi2_contingency(m)[0:2])

def VIF(dataframe, var_list, target):
    scores = []
    for var in var_list:
        #prepare dfs
        y = dataframe[var]
        X = dataframe.drop([var,target], axis=1)
        if var in missing_dict['catU']:
            y = np.array([1 if x=='Y' else 0 for x in y])
        if var == 'SEX':
            y = np.array([1 if x=='F' else 0 for x in cchd.SEX])
        new_cats = [x for x in var_list if x not in [var, 'FAGECOMB_IMP','FRACEHISP_IMP', 'ILOP_R_IMP', 'ILP_R_IMP', 'FEDUC_IMP', 'lrg_miss_imp']]
        X = dummify_columns(X,new_cats)
        
        #run logistic model
        logit = linear_model.LogisticRegression()
        logit.set_params(C=1e4, max_iter = 2000)
        logit.fit(X, y)
        scores+=[[(1/(1-metrics.r2_score(y, logit.predict(X)))), var]]
    return scores

def continuous_VIF(df, target,exclude_=[]):
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    #Identify continuous variables
    cont_vars = list(set(df.select_dtypes(exclude ='category').columns) - set([target]) - set(exclude_))
    
    #Drop features 
    df_no_bs =df.drop(columns = exclude_)
    
    #Label encode and prep for multiple linear regression
    dummified = dummify_columns(df_no_bs,list(set(df.select_dtypes(include ='category').columns)-set(exclude_)))
    target_col = dummified[[target]]
    features = dummified.loc[:,set(dummified.columns) - set(target)]
    
    vif_list=[]
    
    for i in cont_vars:
        lm = LinearRegression()
        lm.fit(features.loc[:,features.columns != i],features[i])
        vif_list += [[i, 1/(1-lm.score(features.loc[:,features.columns != i],features[i]))]]
    vif_cont = pd.DataFrame(vif_list, columns=['feature','vif']).sort_values('vif', ascending = False)\
    .reset_index().drop(columns='index')
    
    #Display VIF > 1.5
    cut_off=max(vif_cont[vif_cont['vif']>1.5].index)
    plt.xticks(rotation = 90)
    plt.bar(vif_cont.loc[:cut_off,'feature'],vif_cont.loc[:cut_off,'vif'])
    
    return vif_cont.loc[:cut_off,['feature','vif']]

#Modeling Function - CCHD:

def dummify_columns(dataframe,var_list):
    '''
    dummifies a columns, merges with the dataframe, and drops the non-dummified column
    ------------
    dataframe: full dataframe
    variable: list of column names as string
    '''
    for vr in var_list:
        dummified_feature = pd.get_dummies(dataframe[vr], prefix=vr,drop_first=True)
        dataframe = pd.concat([dataframe,dummified_feature],axis=1,sort='False')
    dataframe = dataframe.drop(var_list, axis=1)
    return dataframe

def xy_split(dataframe,target):
    '''
    splits a dataframe into a target array and estimator dataframe
    '''
    y=dataframe[target]
    X=dataframe.drop(target, axis=1)
    return X,y

def standardize_columns(df,var_list):
    '''
    standardize a columns, merges with the dataframe, and drops the non-standardized column
    ------------
    dataframe: full dataframe
    variable: column name as string
    '''
    from sklearn.preprocessing import StandardScaler
    scaleit = StandardScaler()
    for vr in var_list:
        scaled_feature = pd.DataFrame(scaleit.fit_transform(df[[vr]]), index = df.index, columns=[vr+'__S'])
        df = pd.concat([df,scaled_feature],axis=1,sort=False)
    df.drop(columns = var_list, inplace = True)
    return df

def add_random_column_to_df (dataframe):
    import random
    mylist = []
    for i in range(0,dataframe.shape[0]):
        x = random.randint(1,1000)
        mylist.append(x)
    dataframe['RANDOM'] = mylist

    return dataframe

#LabelEncoding Function. Thanks Ira!
def LabelEncoding(dataframe):
    '''
    Function that takes a dataframe and transforms it with label encoding on all the categorical features.
    '''
    
    import pandas as pd
    
    #create a list using object types since dataframe.dtypes.value_counts() only shows objects and int64
    objlist = list(dataframe.select_dtypes(include=['object','category']).columns)
    
    #change type then transform column using cat codes
    for col in objlist:
        dataframe[col] = dataframe[col].astype('category')
        dataframe[col] = dataframe[col].cat.codes
    
    return dataframe


def basic_significance(dataframe, list_to_dummify, target):
    '''
    fits a non-regularized logistic model to target using dataframe predictors
    prints model accuracy and outputs significant coefficients order by absolute magnitude
    ----------
    list_to_dummify: a list of columns in string format that require dummification before modeling
    '''
    #process the dataframe
    df = dataframe.copy()
    df = dummify_columns(df, list_to_dummify)
    X,y = xy_split(df, target)
    X = add_constant(X)
    #fit the model
    logit = Logit(y,X)
    fitted_logit = Logit.fit(logit)
    #store accuracy
    c_mat = confusion_matrix(y, np.round(Logit.predict(logit, fitted_logit.params)))
    accuracy = sum(c_mat.diagonal())/np.sum(c_mat)
    print('model train accuracy: %s' %(accuracy))
    #store significant coefs
    coefs = pd.DataFrame(fitted_logit.pvalues[fitted_logit.pvalues<0.05])
    coefs['coefs'] = fitted_logit.params.filter(items=coefs.index)
    coefs.columns = ['p-values', 'coefs']
    coefs['abs_coefs'] = np.abs(coefs.coefs)
    coefs = coefs.sort_values(by='abs_coefs', ascending = False)
    coefs = coefs.drop('abs_coefs', axis =1)
    return fitted_logit, coefs

def forward_selection(dataframe, target, list_to_dummify, criteria='bic'):
    '''
    runs forward selection process to select best predictor set based on bic or aic
    returns a dictionary with the variable set and aic/bic at each step
    ----------
    criteria: default value bic, otherwise aic is used
    list_to_dummify: a list of columns in string format that require dummification before modeling
    '''
    #create target array, intercept only dataframe, and list of variables to select from
    X = pd.DataFrame()
    y = dataframe[target]
    X['const'] = np.ones(cchd.shape[0])
    var_list = list(dataframe.columns)
    var_list.remove(target)
    
    #create empty dictionary to store output of each step
    models = {'model_vars': [], 'scoring_crit':[]}
    
    #define while loop that will run until all variables have been selected
    while len(var_list) > 0: 
        
        #define empty list to store aic/bic values temporarily for step attempt
        crit_vals = []
        
        #try adding variables one by one find lowest vif model for current step
        for var in var_list:
            #create temporary df with all previously selected variables + the new variable being tried
            tempX=pd.concat([X,dataframe[var]],axis=1)
            #dummify the variable if necessary
            if var in list_to_dummify:
                tempX = dummify_columns(tempX, [var])
            #fit the logistic model
            logit = Logit(y,tempX)
            fitted_logit = Logit.fit(logit)
            #store aic or bic in a list for each variable attempted
            if criteria == 'bic':
                crit_vals += [fitted_logit.bic]
            else:
                crit_vals += [fitted_logit.aic]
        
        #find the index of the lowest bic model and store the name of the variable which produced it
        min_crit_idx = crit_vals.index(min(crit_vals))
        best_var = var_list[min_crit_idx]
        
        #add the best variable to the df
        X = pd.concat([X, dataframe[best_var]], axis=1)
        
        #store the variables and aic/bic for the best model at the current step
        models['model_vars']+=[list(X.columns)]
        models['scoring_crit']+=[min(crit_vals)]
        
        #dummify the added variable if necessary
        if best_var in list_to_dummify:
            X = dummify_columns(X, [best_var])
        
        #remove the added variable from the variable list and track progress
        var_list.remove(best_var)
        print('adding var: %s' %(best_var))
        
    return models

def best_forward_set(forward_models):
    '''
    returns cleaned columns as they appear in the original dataset which are used in best forward selection model
    ----------
    forward_models: dictionary output by the forward_selection function
    '''
    model_idx = forward_models['scoring_crit'].index(min(forward_models['scoring_crit']))
    best_cols = forward_models['model_vars'][model_idx]
    best_cols_clean = []
    for i in range(0,len(best_cols)):
        best_cols_clean+=[re.search('\D+', best_cols[i])[0]]
    final_cols=[]
    for i in range(0,len(best_cols_clean)):
        final_cols+=[re.sub("_$", "", best_cols_clean[i])]
    final_cols2=[]
    for i in range(0,len(final_cols)):
        final_cols2+=[re.sub("_(?:Y|M|N|)$", "", final_cols[i])]
    final_cols_clean = set(final_cols2)
    final_cols_clean.remove('const')
    return list(final_cols_clean)

def grid_search_logit(dataframe, columns_to_dummify, target, grid_params, standardize = 'Y'):
    '''
    fit regularized logistic model with grid search to select optimal reg paramter
    returns score, parameters, and coefficients from best model
    ----------
    columns_to_dummify: a list of columns in string format that require dummification before modeling
    grid_params: parameters to grid search across
    '''
    df = dataframe.copy()
    df = dummify_columns(df, columns_to_dummify)
    X,y = xy_split(df, target)
    if standardize == 'Y':
        X = standardize_columns(X,list(X.columns))
    logit = linear_model.LogisticRegression()
    logit.set_params(solver='liblinear')
    log_grid = GridSearchCV(estimator = logit, param_grid=grid_params, scoring='accuracy', cv=5, return_train_score=True)
    log_grid.fit(X,y)
    coefs = pd.Series([item for sublist in log_grid.best_estimator_.fit(X,y).coef_ for item in sublist], index=X.columns)
    order = abs(coefs).sort_values(ascending=False)
    return log_grid.best_score_, log_grid.best_params_, coefs[order.index], log_grid.best_estimator_

def grid_search_logitw(dataframe, columns_to_dummify, target, grid_params, standardize = 'Y'):
    '''
    fit regularized logistic model with grid search to select optimal reg paramter
    returns score, parameters, and coefficients from best model
    ----------
    columns_to_dummify: a list of columns in string format that require dummification before modeling
    grid_params: parameters to grid search across
    '''
    df = dataframe.copy()
    df = dummify_columns(df, columns_to_dummify)
    X,y = xy_split(df, target)
    if standardize == 'Y':
        X = standardize_columns(X,list(X.columns))
    logit = linear_model.LogisticRegression()
    logit.set_params(solver='liblinear', class_weight='balanced')
    log_grid = GridSearchCV(estimator = logit, param_grid=grid_params, scoring='precision', cv=5, return_train_score=True)
    log_grid.fit(X,y)
    coefs = pd.Series([item for sublist in log_grid.best_estimator_.fit(X,y).coef_ for item in sublist], index=X.columns)
    order = abs(coefs).sort_values(ascending=False)
    return log_grid.best_score_, log_grid.best_params_, coefs[order.index], log_grid.best_estimator_

def precision(y, X_pred):
    c = confusion_matrix(y, X_pred)
    return c[1][1]/(c[0][1]+c[1][1])



