# FUNCTIONS FOR NICU admissions project

def add_random_column_to_df (dataframe):

    '''
    this function adds a randomly generated column to the dataframe
    does not have any relationship to the response variable
    where does this end up in feature importance?
    ----------------------
    dataframe: full dataframe
    '''

    import random
    mylist = []
    for i in range(0, dataframe.shape[0]):
        x = random.randint(1,1000)
        mylist.append(x)
    dataframe['RANDOM'] = mylist

    return dataframe



def downsample_df (df):

    '''
    Remove undefined information on NICU admissions (AB_NICU == 'U'),
    create a binary target vector and a "balanced" dataframe with all NICU
    admissions and matching numbers of randomly selected non-NICU admissions.
    -----------------------
    df: full dataframe
    (variable or defect of interest could be added (here AB_NICU))
    '''

    import pandas as pd
    import numpy as np

    # remove unknown class from df
    df_no_unknown = df[df['AB_NICU'].isin(['Y', 'N'])]

    # Create binary target vector, NICU = yes classified as class 1
    df_y_n = np.where((df_no_unknown['AB_NICU'] == 'Y'), 1, 0)

    # Get indicies of each class' observations
    index_class0 = np.where(df_y_n == 0)[0]
    index_class1 = np.where(df_y_n == 1)[0]

    # Get numbers of observations in class 0
    n_class0 = len(index_class0)

    # Randomly sample the same number of observations from class 0 as in class 1, without replacement
    np.random.seed(0)
    index_class0_downsampled = np.random.choice(index_class0, size=n_class1, replace=False)

    # Create dataframes for NICU and downsampled non-NICU
    df_NICU = df_no_unknown.iloc[index_class1]
    df_adj_NONICU = df_no_unknown.iloc[index_class0_downsampled]

    # Append into 1 dataframe
    df_downsampled = df_NICU.append(df_adj_NONICU)

    return df_downsampled



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



def downsample_after_label_df (df):

    '''
    Create a "balanced" dataframe with all NICU admissions and matching numbers of
    randomly selected non-NICU admissions following label encoding.
    -----------------------
    df: full dataframe
    (variable: variable or defect of interest (here AB_NICU))
    '''

    import pandas as pd
    import numpy as np

    # Get indicies of each class' observations
    index_class0 = np.where(df['AB_NICU'] == 0)[0]
    index_class1 = np.where(df['AB_NICU'] == 1)[0]

    # Get numbers of observations in class 0
    n_class1 = len(index_class1)

    # Randomly sample the same number of observations from class 1 as in class 0, without replacement
    np.random.seed(0)
    index_class0_downsampled = np.random.choice(index_class0, size=n_class1, replace=False)

    # Create dataframes for NICU and downsampled non-NICU
    df_defect = df.iloc[index_class1]
    df_adj_NONdefect = df.iloc[index_class0_downsampled]

    # Append into 1 dataframe
    df_downsampled = df_defect.append(df_adj_NONdefect)

    return df_downsampled



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
