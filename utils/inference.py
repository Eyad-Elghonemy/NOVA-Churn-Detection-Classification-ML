from .CustomerData import CustomerData
import pandas as pd

def predict_new(data: CustomerData, preprocessor, model):
        
        # To Dataframe        
        df = pd.DataFrame([data.model_dump()])


        
        # Transform
        X_processed = preprocessor.transform(df)
        
        # Predict
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)
        
        
        return  {
            "Churn_Prediction" : bool(y_pred[0]),
            "Churn_Probability" : float(y_prob[0][1])
        }
    