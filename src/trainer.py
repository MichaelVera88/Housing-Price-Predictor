import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def train_lr_model(preprocessor, x_train_dataset, y_train_dataset):
    '''
    Uses Linear Regression to train model from given data
    
    :param preprocessor: Preprocessor for Data
    :param x_train_dataset: x Dataset
    :param y_train_dataset: y Dataset
    '''

    lr_model = LinearRegression()
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", lr_model)])
    model_pipeline.fit(x_train_dataset, y_train_dataset)

    return model_pipeline

def save_model(model):
    '''
    Saves trained model to model.pkl file
    
    :param model: Trained Model
    '''
    path = "data/models/model.pkl"

    joblib.dump(model, path)
