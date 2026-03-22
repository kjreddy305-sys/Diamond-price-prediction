import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]  
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)   

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)
def model_metrics(true,predicted):
    try :
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, mse,rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)

def print_evaluated_results(X_train,y_train,X_test,y_test,model):
    try:
        y_train_pred=model.predict(X_train)
        y_test_pred=model.predict(X_test)

        model_train_mae ,model_train_mse,model_train_rmse, model_train_r2 = model_metrics(y_train, y_train_pred)
        model_test_mae ,model_test_mse ,model_test_rmse, model_test_r2 = model_metrics(y_test, y_test_pred)

        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- Mean squared error: {:.4f}".format(model_train_mse))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- Mean squared error: {:.4f}".format(model_test_mse))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        

    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    