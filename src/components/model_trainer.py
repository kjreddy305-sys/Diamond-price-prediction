import sys
import os
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print("\n=================================================\n")

            logging.info(f'Model Report : {model_report}')

            # Get best model score
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for catboost')

            cbr=CatBoostRegressor(verbose=False)

            param_dist = {'depth'          : [4,5,6,7,8,9, 10],
                          'learning_rate' : [0.01,0.02,0.03,0.04],
                          'iterations'    : [300,400,500,600]}
            rscv=RandomizedSearchCV(cbr,param_dist,scoring='r2',n_jobs=-1,cv=5)
            rscv.fit(X_train,y_train)
            print(f"best catbbost parameter: {rscv.best_params_}")
            print(f'best catboost score :{rscv.best_score_}')
            print('\n====================================================================================\n')

            best_cbr = rscv.best_estimator_

            logging.info('Hyperparameter tuning complete for Catboost')

            logging.info('Hyperparameter tuning started for KNN')

            # Initialize knn
            knn = KNeighborsRegressor()

            # parameters
            k_range = list(range(2, 31))
            param_grid = dict(n_neighbors=k_range)

            # Fitting the cvmodel
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
            grid.fit(X_train, y_train)

            # Print the tuned parameters and score
            print(f'Best KNN Parameters : {grid.best_params_}')
            print(f'Best KNN Score : {grid.best_score_}')
            print('\n====================================================================================\n')

            best_knn = grid.best_estimator_

            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')
            er = VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
            er.fit(X_train, y_train)
            print('Final Model Evaluation :\n')
            print_evaluated_results(X_train,y_train,X_test,y_test,er)
            logging.info('Voting Regressor Training Completed')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = er
            )
            logging.info('Model pickle file saved')
            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            y_test_pred = er.predict(X_test)

            mae,mse, rmse, r2 = model_metrics(y_test, y_test_pred)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info(f'Test MSE Score : {mse}')
            logging.info('Final Model Training Completed')
            
            return mae, rmse, r2 ,mse
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

            