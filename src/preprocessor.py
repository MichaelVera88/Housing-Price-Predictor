from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split_data(dataframe, y_column):
    '''
    Splits the Pandas dataframe into independent and dependent variables (x, y).
    
    :param dataframe: Pandas Dataframe
    :param y_column: Target Column of Dataframe
    '''

    y = dataframe[y_column]
    x = dataframe.drop(columns=[y_column])

    return train_test_split(x, y, test_size=0.2, random_state=42)

def preprocess(num_columns):
    '''
    Pipeline for data that uses standard scaler and column transformer.
    
    :param num_features: Number features
    '''
    num_pipline = Pipeline(steps=[("scaler", StandardScaler())])
    process = ColumnTransformer(transformers=[("num", num_pipline, num_columns)])


    return process
