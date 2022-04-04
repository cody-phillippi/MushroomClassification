from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score
import pandas as pd

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score. We are only using tree
                based alogrithms based on the nature of this data
                Written By: Cody Phillippi
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rf = RandomForestClassifier()
        self.dt = DecisionTreeClassifier()
        self.xgb = GradientBoostingClassifier()

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Cody Phillippi
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 8, 1), "max_features": ['auto', 'sqrt', 'log2'],
                               "min_samples_split":range(2, 8, 1), "min_samples_leaf":range(2, 8, 1),
                               }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rf, param_grid=self.param_grid, cv=5, verbose=3)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']

            # creating a new model with the best parameters
            self.rf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features, min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf)
            # training the mew model
            self.rf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.rf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_decision_tree(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_decision_tree
                                Description: get the parameters for Decision Tree Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Cody Phillippi
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_decision_tree method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 8, 1), "max_features": ['auto', 'sqrt', 'log2'],
                               "min_samples_split": range(2, 8, 1), "min_samples_leaf": range(2, 8, 1),
                               }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.dt, param_grid=self.param_grid, cv=5, verbose=3)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']

            # creating a new model with the best parameters
            self.dt = DecisionTreeClassifier(criterion=self.criterion,
                                             max_depth=self.max_depth, max_features=self.max_features,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf)
            # training the mew model
            self.dt.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Decision Tree best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_decision_tree method of the Model_Finder class')

            return self.dt
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_decision_tree method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Decision Tree Parameter tuning  failed. Exited the get_best_params_for_decision_tree method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgb(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_xgb
                                Description: get the parameters for Gradient Boosting Classifier Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Cody Phillippi
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgb method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"loss": ['deviance', 'exponential'],
                               "learning_rate": [.01, .05, .1, 1, 2, 4, 6], "n_estimators": [10, 50, 100, 130],
                               "max_features": ['auto', 'sqrt', 'log2'], "criterion":['friedman_mse', 'squared_error', 'mse', 'mae'],
                               "min_samples_split": range(2, 8, 1), "min_samples_leaf": range(2, 8, 1),
                               }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.xgb, param_grid=self.param_grid, cv=5, verbose=3)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.loss = self.grid.best_params_['loss']
            self.learning_rate = self.grid.best_params_['learning_rate']

            # creating a new model with the best parameters
            self.xgb = GradientBoostingClassifier(criterion=self.criterion,
                                             max_depth=self.max_depth, max_features=self.max_features,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGB best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgb method of the Model_Finder class')

            return self.dt
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgb method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGB Parameter tuning  failed. Exited the get_best_params_for_decision_tree method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Cody Phillippi
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for dt
        try:
            self.dt= self.get_best_params_for_decision_tree(train_x,train_y)
            self.prediction_dt = self.dt.predict(test_x) # Predictions using the Decision Tree Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.dt_score = accuracy_score(test_y, self.prediction_dt)
                self.logger_object.log(self.file_object, 'Accuracy for DT:' + str(self.knn_score))  # Log AUC
            else:
                self.dt_score = roc_auc_score(test_y, self.prediction_dt) # AUC for DT
                self.logger_object.log(self.file_object, 'AUC for DT:' + str(self.dt_score)) # Log AUC

            # create best model for Random Forest
            self.rf=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_rf=self.rf.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.rf_score = accuracy_score((test_y),self.prediction_rf)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.rf_score))
            else:
                self.rf_score = roc_auc_score((test_y), self.prediction_rf) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.rf_score))

            # create best model for XGB
            self.xgb = self.get_best_params_for_xgb(train_x, train_y)
            self.prediction_xgb = self.xgb.predict(test_x)  # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgb_score = accuracy_score((test_y), self.prediction_xgb)
                self.logger_object.log(self.file_object, 'Accuracy for XGB:' + str(self.xgb_score))
            else:
                self.xgb_score = roc_auc_score((test_y), self.prediction_xgb)  # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for XGB:' + str(self.xgb_score))

            #comparing the three models
            if(self.rf_score <  self.dt_score):
                return 'DT',self.dt
            elif(self.dt_score < self.xgb_score):
                return 'XGB', self.xgb
            else:
                return 'RandomForest',self.rf

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()