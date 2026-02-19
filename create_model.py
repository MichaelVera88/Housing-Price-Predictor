from src.loader import load_dataset
from src.preprocessor import split_data, preprocess
from src.trainer import train_lr_model, train_rf_model, save_model
from src.evaluator import evaluate_model

data_path = "data/raw/house_prices_practice.csv"

def create_model():
    '''
    Create Model Function:
        - Loads dataset into a Panda dataframe
        - Preprocesses columns of dataframe into preprocessor pipeline
        - Trains model with processed dataframe and training datasets
        - Evaluates trained model with test datasets
        - Saves the trained model
    '''
    
    df = load_dataset(data_path)

    lr_model_save = "data/models/linear_regression_model.pkl"
    rf_model_save = "data/models/random_forest_regressor_model.pkl"

    num_columns = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "FullBath",
        "BedroomAbvGr",
        "LotArea"
    ]

    x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset = split_data(df, "SalePrice")
    processed = preprocess(num_columns)

    trained_lr_model = train_lr_model(processed, x_train_dataset, y_train_dataset)
    lr_evaluations = evaluate_model(trained_lr_model, x_test_dataset, y_test_dataset)

    trained_rf_model = train_rf_model(processed, x_train_dataset, y_train_dataset)
    rf_evaluations = evaluate_model(trained_rf_model, x_test_dataset, y_test_dataset)

    save_model(trained_lr_model, lr_model_save)
    save_model(trained_rf_model, rf_model_save)


    print("Linear Regression Results:")
    for key, value in lr_evaluations.items():
        print(f"{key}: {value:.2f}")
    
    print("Random Forest Regressor Results:")
    for key, value in rf_evaluations.items():
        print(f"{key}: {value:.2f}")

create_model()

