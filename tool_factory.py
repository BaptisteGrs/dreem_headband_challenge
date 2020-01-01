import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import time
import warnings


def setup_environment():
    print('Setup environment...', end='')
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)

    # set jupyter notebook pandas dataframe output parameters
    pd.set_option('display.max_rows'    , 200)
    pd.set_option('display.max_columns' , 200)
    pd.set_option('display.max_colwidth', 1000)

    # set jupyter notebook matplotlib figure output default parameters
    matplotlib.rcParams['figure.dpi']     = 200
    matplotlib.rcParams['figure.figsize'] = (4, 4)

    # set the matplotlib figures style
    sns.set_style('whitegrid')

    print(' done!')


def get_table(col):
        """
        returns a tbale with the count of distinct categorical values
        """
        count =  col.value_counts()
        freq = count/len(col)
        freq_str = freq.apply(lambda x : "{:.2f} %".format(x*100))
        #freq_str = "{:.2f}".format(freq*100)
        table=pd.DataFrame({'count': count, 'freq': freq_str})
        return table
    


def plot_categorical_distribution(data, col, n_x, rot=90):
    """
    Plots the distribution of most frequent objects in a pandas Series
    
    data : dataframe
    col : str, column of the dataframe we want to observe
    n_x : print the first n_x objects
    """
    table = get_table(data[col]).head(n_x)
    hist = table.plot.bar(figsize=(20, 4), width=0.8, legend=False, linewidth=0)
    nb = min(data[col].nunique(), n_x) #to avoid n+1/n situations
    plt.title('{} distribution, top {}/{}'.format(col, nb, data[col].nunique()), fontsize=18)
    plt.xlabel('{} name'.format(col))
    plt.ylabel('count')
    plt.xticks(rotation=rot)
    
    plt.gca().xaxis.grid(False) # remove xaxis grid line
    

def plot_continous_distribution(data, col, n_x,rot=0):
    """
    Plots the distribution of most frequent objects in a pandas Series for continuous variables
    
    data : dataframe
    col : str, column of the dataframe we want to observe
    n_x : group into n_x bins
    """
    #ignore FutureWarning due to error "FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use arr[tuple(seq)] instead of arr[seq]. In the future this will be interpreted as an array index, arr[np.array(seq)], which will result either in an error or a different result.
    #return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval"
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    #plot the histogram + density of the continuous variable
    fig, ax = plt.subplots(1, 3,figsize=(20, 10))
    
    sns.distplot(data[col].dropna(), hist=True, bins=n_x, ax=ax[0], kde=True)
    ax[0].set_title('{} distribution, in {} bins, removed {} NaN objects'.format(col, n_x, data[col].isna().sum()), fontsize=18)
    ax[0].set_xlabel('{} value'.format(col), fontsize=14)
    ax[0].set_ylabel('density', fontsize=14)
    ax[0].xaxis.grid(False) # remove xaxis grid line
    ax[1].boxplot(data[col].dropna())
    ax[2].violinplot(data[col].dropna())

def report(results, n_top=3):
    """
    Report the Grid Search results
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
