import pandas as pd

def load_dataset(path):
    '''
    Loads data set into a Pandas dataframe from data set path.
    
    :param path: Data set path
    '''

    df = pd.read_csv(path)
    return df