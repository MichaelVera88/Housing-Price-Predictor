import numpy as np
import sklearn.metrics as metric

def evaluate_model(model, x_test_dataset, y_test_dataset):
    '''
    Evaluates given model based on MAE, RMSE, and R2 score metrics
    
    :param model: Trained Model
    :param x_test_dataset: x Dataset
    :param y_test_dataset: y Dataset
    '''

    predictions = model.predict(x_test_dataset)

    mae = metric.mean_absolute_error(y_test_dataset, predictions)
    mse = metric.mean_squared_error(y_test_dataset, predictions)
    rmse = np.sqrt(mse)
    r2 = metric.r2_score(y_test_dataset, predictions)

    evaluations = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return evaluations