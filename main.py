import joblib
import pandas as pd

def main():
    '''
    Main function gets user input of house attributes and uses model to estimate price
    '''

    lr_model_path = "data/models/model.pkl"
    lr_model = joblib.load(lr_model_path)

    OverallQual = int(input("Overall Quality (1 - 10): "))
    GrLivArea = int(input("Living Area (sq ft): "))
    GarageCars = int(input("Garage Capacity: "))
    TotalBsmtSF = int(input("Basement Area (sq ft): "))
    YearBuilt = int(input("Year Built: "))
    FullBath = int(input("Bathrooms: "))
    BedroomAbvGr = int(input("Bedrooms: "))
    LotArea = int(input("Lot Area: "))

    house_input = {
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "TotalBsmtSF": TotalBsmtSF,
        "YearBuilt": YearBuilt,
        "FullBath": FullBath,
        "BedroomAbvGr": BedroomAbvGr,
        "LotArea": LotArea
    }

    house_input_df = pd.DataFrame([house_input])
    estimation = lr_model.predict(house_input_df)


    print(f"House Price: ${estimation[0]:,.2f}")

if __name__ == "__main__":
    main()
