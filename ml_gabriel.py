#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
    Script created to consolidate useful functions used in the training of machine learning classification models.
    
"""

"""
--------------------------------------------
---------- IMPORT LIBRARIES ----------------
--------------------------------------------
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, cross_val_predict,                                     learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve,     accuracy_score, precision_score, recall_score, f1_score


"""
--------------------------------------------
------- 1. CLASSIFICATION MODELS -----------
--------------------------------------------
"""


class BinaryBaselineClassifier():

    def __init__(self, model, set_prep, features):
        self.model = model
        self.X_train = set_prep['X_train_prep']
        self.y_train = set_prep['y_train']
        self.X_test = set_prep['X_test_prep']
        self.y_test = set_prep['y_test']
        self.features = features
        self.model_name = model.__class__.__name__

    def random_search(self, scoring, param_grid = None, cv = 5):
        """
        Steps:
            1. automatic definition of search parameters if the model is a Decision Tree
            2. application of RandomizedSearchCV with the defined parameters

        Arguments:
            scoring - metric to be optimized during the search [string]
            param_grid - dictionary with the parameters to be used in the search [dict]
            tree - flag to indicate whether the baseline model is a decision tree [bool]

        Return:
            best_estimator_ - best model found in the search
        """

        #Validating baseline as Decision Tree (grid defined automatically)
        """if tree:
            param_grid = {
                'criterion': ['entropy', 'gini'],
                'max_depth': [3, 4, 5, 8, 10],
                'max_features': np.arange(1, self.X_train.shape[1]),
                'class_weight': ['balanced', None]
            }"""

        #Applying random hyperparameter search
        rnd_search = RandomizedSearchCV(self.model, param_grid, scoring = scoring, cv = cv, verbose = 1,
                                        random_state = 42, n_jobs = -1)
        rnd_search.fit(self.X_train, self.y_train)

        return rnd_search.best_estimator_

    def fit(self, rnd_search = False, scoring = None, param_grid = None):
        """
        Steps:
            1. model training and result attribution as a class attribute

        Arguments:
            rnd_search - flag indicating the application of RandomizedSearchCV [bool]
            scoring - metric to be optimized during the search [string]
            param_grid - dictionary with the parameters to be used in the search [dict]
            tree - flag to indicate whether the baseline model is a decision tree [bool]

        Return:
            None
        """

        #Training model according to the selected argument
        if rnd_search:
            print(f'Training model {self.model_name} with RandomSearchCV.')
            self.trained_model = self.random_search(param_grid = param_grid, scoring = scoring)
            print(f'Training completed successfully. Configuration settings: \n\n{self.trained_model}')
        else:
            print(f'Training model {self.model_name}.')
            self.trained_model = self.model.fit(self.X_train, self.y_train)
            print(f'Training completed successfully. Configuration setting: \n\n{self.trained_model}')

    def evaluate_performance(self, approach, cv = 5, test = False):
        """
        Steps:
            1. measurement of the main metrics for the model

        Arguments:
            cv - number of k-folds during the application of cross validation [int]

        Return:
            df_performance - DataFrame containing the model's performance against the metrics [pandas.DataFrame]
        """

        #Initiate time measurement
        t0 = time.time()

        if test:
            #Returning predictions with test data
            y_pred = self.trained_model.predict(self.X_test)
            y_proba = self.trained_model.predict_proba(self.X_test)[:, 1]

            #Returning metrics to the test data
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba)
        else:
            #Assessing key model metrics through cross-validation
            accuracy = cross_val_score(self.trained_model, self.X_train, self.y_train, cv = cv,
                                       scoring = 'accuracy').mean()
            precision = cross_val_score(self.trained_model, self.X_train, self.y_train, cv = cv,
                                        scoring = 'precision').mean()
            recall = cross_val_score(self.trained_model, self.X_train, self.y_train, cv = cv,
                                     scoring = 'recall').mean()
            f1 = cross_val_score(self.trained_model, self.X_train, self.y_train, cv = cv,
                                 scoring = 'f1').mean()

            #AUC score
            try:
                y_scores = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv = cv,
                                             method = 'decision_function')
            except:
                #Tree-based models do not have the 'decision_function' method, but 'predict_proba'
                y_probas = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv = cv,
                                             method = 'predict_proba')
                y_scores = y_probas[:, 1]
            #Calculate AUC
            auc = roc_auc_score(self.y_train, y_scores)

        #Finalizing time measurement
        t1 = time.time()
        delta_time = t1 - t0

        #Saving data into dataframe
        performance = {}
        performance['approach'] = approach
        performance['acc'] = round(accuracy, 4)
        performance['precision'] = round(precision, 4)
        performance['recall'] = round(recall, 4)
        performance['f1'] = round(f1, 4)
        performance['auc'] = round(auc, 4)
        performance['total_time'] = round(delta_time, 3)

        df_performance = pd.DataFrame(performance, index = performance.keys()).reset_index(drop = True).loc[:0, :]
        df_performance.index = [self.model_name]

        return df_performance

    def plot_confusion_matrix(self, classes, cv = 5, cmap = plt.cm.Blues, title = 'Confusion Matrix', normalize = False):
        """
        Steps:
            1. confusion matrix calculation using cross-validation predictions
            2. plotting configuration and construction
            3. formatting the plot labels

        Arguments:
            classes - name of the classes involved in the model [list]
            cv - number of folds applied in the cross validation [int - default: 5]
            cmap - colorimetric matrix mapping [plt.colormap - default: plt.cm.Blues]
            title - confusion matrix title [string - default: 'Confusion Matrix']
            normalize - indicator for normalization of matrix data [bool - default: False]

        Return
        """

        #Making predictions and returning confusion matrix
        y_pred = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv = cv)
        conf_mx = confusion_matrix(self.y_train, y_pred)

        #Plot matrix
        sns.set(style = 'white', palette = 'muted', color_codes = True)
        plt.imshow(conf_mx, interpolation = 'nearest', cmap = cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        #Customize axes
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        #Customize entries
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment = 'center',
                     color = 'white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title, size = 14)

    def plot_roc_curve(self, cv = 5):
        """
        Steps:
            1. model scores return using cross-validation prediction
            2. Finding false positive and true negative rates
            3. calculation of the AUC metric and plotting of the ROC curve

        Arguments:
            cv - number of k-folds used in cross-validation [int - default: 5]

        Return:
            None
        """

        #Calculating scores using cross-validation prediction
        try:
            y_scores = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv = cv,
                                         method = 'decision_function')
        except:
            #Tree-based algorithms do not have the "decision_function" method but 'predict_proba'
            y_probas = cross_val_predict(self.trained_model, self.X_train, self.y_train, cv = cv,
                                         method = 'predict_proba')
            y_scores = y_probas[:, 1]

        #Calculating false positive and true positive rates
        fpr, tpr, thresholds = roc_curve(self.y_train, y_scores)
        auc = roc_auc_score(self.y_train, y_scores)

        #Plot ROC Curve
        plt.plot(fpr, tpr, linewidth = 2, label = f'{self.model_name} auc = {auc: .3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.02, 1.02, -0.02, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

    def feature_importance_analysis(self):
        """
        Steps:
            1. return of importance of features
            2. construction of a DataFrame with the most important features for the model

        Arguments:
            None

        Return:
            feat_imp - DataFrame with feature importances [pandas.DataFrame]
        """

        #Returning feature importance model
        importances = self.trained_model.feature_importances_
        feat_imp = pd.DataFrame({})
        feat_imp['feature'] = self.features
        feat_imp['importance'] = importances
        feat_imp = feat_imp.sort_values(by = 'importance', ascending = False)
        feat_imp.reset_index(drop = True, inplace = True)

        return feat_imp

    def plot_learning_curve(self, ylim = None, cv = 5, n_jobs = 1, train_sizes = np.linspace(.1, 1.0, 10),
                            figsize = (12, 6)):
        """
        Steps:
            1. calculation of training scores and validation according to the amount of data m
            2. calculation of statistical parameters (mean and standard deviation) of the scores
            3. training and validation learning curve plot

        Arguments:
            y_lim - definition of y-axis limits [list - default: None]
            cv - k folds in the cross validation application [int - default: 5]
            n_jobs - number of jobs during the execution of the learning_curve function [int - default: 1]
            train_sizes - sizes considered for dataset slices [np.array - default: linspace (.1, 1, 10)]
            figsize - graphical plot dimensions [tuple - default: (12, 6)]

        Return:
            None
        """

        #Returning training and validation score parameters
        train_sizes, train_scores, val_scores = learning_curve(self.trained_model, self.X_train, self.y_train,
                                                               cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)

        #Calculating means and standard deviations (training and validation)
        train_scores_mean = np.mean(train_scores, axis = 1)
        train_scores_std = np.std(train_scores, axis = 1)
        val_scores_mean = np.mean(val_scores, axis = 1)
        val_scores_std = np.std(val_scores, axis = 1)

        #Plotting learning curve graph
        fig, ax = plt.subplots(figsize = figsize)

        #Training data results
        ax.plot(train_sizes, train_scores_mean, 'o-', color = 'navy', label = 'Training Score')
        ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                        alpha = 0.1, color = 'blue')

        #Cross-validation results
        ax.plot(train_sizes, val_scores_mean, 'o-', color = 'red', label = 'Cross Val Score')
        ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                        alpha = 0.1, color = 'crimson')

        #Customize graph
        ax.set_title(f'Model {self.model_name} - Learning Curve', size = 14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc = 'best')
        plt.show()


"""
--------------------------------------------
------ 2. ANALYSIS CLASSIFICATION MODELS ---
--------------------------------------------
"""


class BinaryClassifiersAnalysis():

    def __init__(self):
        self.classifiers_info = {}

    def fit(self, classifiers, X, y, approach = '', random_search = False, scoring = 'roc_auc',
            cv = 5, verbose = 5, n_jobs = -1):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Iterating over each model in the classifier dictionary
        for model_name, model_info in classifiers.items():
            clf_key = model_name + approach
            print(f'Training model {clf_key}\n')
            self.classifiers_info[clf_key] = {}

            #Validating the application of RandomizedSearchCV
            if random_search:
                rnd_search = RandomizedSearchCV(model_info['model'], model_info['params'], scoring = scoring, cv = cv,
                                                verbose = verbose, random_state = 42, n_jobs = n_jobs)
                rnd_search.fit(X, y)
                self.classifiers_info[clf_key]['estimator'] = rnd_search.best_estimator_
            else:
                self.classifiers_info[clf_key]['estimator'] = model_info['model'].fit(X, y)

    def compute_train_performance(self, model_name, estimator, X, y, cv = 5):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Computing key metrics by cross-validation
        t0 = time.time()
        accuracy = cross_val_score(estimator, X, y, cv = cv, scoring = 'accuracy').mean()
        precision = cross_val_score(estimator, X, y, cv = cv, scoring = 'precision').mean()
        recall = cross_val_score(estimator, X, y, cv = cv, scoring = 'recall').mean()
        f1 = cross_val_score(estimator, X, y, cv = cv, scoring = 'f1').mean()

        #Probabilities for AUC
        try:
            y_scores = cross_val_predict(estimator, X, y, cv = cv, method = 'decision_function')
        except:
            #Tree-based models do not have a 'decision_function' method, but 'predict_proba'
            y_probas = cross_val_predict(estimator, X, y, cv = cv, method = 'predict_proba')
            y_scores = y_probas[:, 1]
        auc = roc_auc_score(y, y_scores)

        #Saving scores in the classifier dictionary
        self.classifiers_info[model_name]['train_scores'] = y_scores

        #Create dataframe with metrics
        t1 = time.time()
        delta_time = t1 - t0
        train_performance = {}
        train_performance['model'] = model_name
        train_performance['approach'] = f'Train {cv} K-folds'
        train_performance['acc'] = round(accuracy, 4)
        train_performance['precision'] = round(precision, 4)
        train_performance['recall'] = round(recall, 4)
        train_performance['f1'] = round(f1, 4)
        train_performance['auc'] = round(auc, 4)
        train_performance['total_time'] = round(delta_time, 3)

        df_train_performance = pd.DataFrame(train_performance,
                                            index = train_performance.keys()).reset_index(drop = True).loc[:0, :]

        return df_train_performance

    def compute_test_performance(self, model_name, estimator, X, y, cv = 5):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Calculating predictions and scores with training data
        t0 = time.time()
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X)
        y_scores = y_proba[:, 1]

        #Returning metrics to the test data
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_scores)

        #Saving probabilities in the stats of trained classifiers
        self.classifiers_info[model_name]['test_scores'] = y_scores

        #Create dataframe for metrics
        t1 = time.time()
        delta_time = t1 - t0
        test_performance = {}
        test_performance['model'] = model_name
        test_performance['approach'] = f'Test'
        test_performance['acc'] = round(accuracy, 4)
        test_performance['precision'] = round(precision, 4)
        test_performance['recall'] = round(recall, 4)
        test_performance['f1'] = round(f1, 4)
        test_performance['auc'] = round(auc, 4)
        test_performance['total_time'] = round(delta_time, 3)

        df_test_performance = pd.DataFrame(test_performance,
                                           index = test_performance.keys()).reset_index(drop = True).loc[:0, :]

        return df_test_performance

    def evaluate_performance(self, X_train, y_train, X_test, y_test, cv=5, approach=''):
       
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Iterating over each classifier already trained
        df_performances = pd.DataFrame({})
        for model_name, model_info in self.classifiers_info.items():

            #Checking if the model has been evaluated before
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['test_performance'])
                continue

            #Indexing variables for calculations
            print(f'Evaluating model {model_name}\n')
            estimator = model_info['estimator']

            #Returning metrics to training data
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train)
            test_performance = self.compute_test_performance(model_name, estimator, X_test, y_test)

            #Saving results to dictionary of model
            self.classifiers_info[model_name]['train_performance'] = train_performance
            self.classifiers_info[model_name]['test_performance'] = test_performance

            #Returning a single DataFrame with the performances obtained
            model_performance = train_performance.append(test_performance)
            df_performances = df_performances.append(model_performance)

            #Saving data sets as attributes for future access
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            model_info['model_data'] = model_data

        return df_performances

    def feature_importance_analysis(self, features, specific_model = None, graph = True, ax = None, top_n = 30,
                                    palette = 'viridis'):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Iterating over each of the classifiers already trained
        feat_imp = pd.DataFrame({})
        for model_name, model_info in self.classifiers_info.items():
            #Create feature importance dataframe
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                continue
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp.sort_values(by = 'importance', ascending = False, inplace = True)
            feat_imp.reset_index(drop = True, inplace = True)

            #Saving set of feature importances in the classifier dictionary
            self.classifiers_info[model_name]['feature_importances'] = feat_imp

        #Returning feature importances for a specific classifier
        if specific_model is not None:
            try:
                model_feature_importance = self.classifiers_info[specific_model]['feature_importances']
                if graph:  # Plot graph
                    sns.barplot(x = 'importance', y = 'feature', data = model_feature_importance.iloc[:top_n, :],
                                ax = ax, palette = palette)
                    format_spines(ax, right_border = False)
                    ax.set_title(f'Top {top_n} {model_name} Most important features', size = 14, color = 'dimgrey')
                return model_feature_importance
            except:
                print(f'Classifier {specific_model} was not trained.')
                print(f'Possible options: {list(self.classifiers_info.keys())}')
                return None

        #Validating inconsistent combination of arguments
        if graph and specific_model is None:
            print('Please choose a specific model to view the feature import chart')
            return None

    def plot_roc_curve(self, figsize = (16, 6)):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Initialize plot for ROC curve
        fig, axs = plt.subplots(ncols = 2, figsize = figsize)

        #Iterating over each of the trained classifiers
        for model_name, model_info in self.classifiers_info.items():
            #Returning sets from the model
            y_train = model_info['model_data']['y_train']
            y_test = model_info['model_data']['y_test']

            #Return scores
            train_scores = model_info['train_scores']
            test_scores = model_info['test_scores']

            #Calculating false positive and true positive rates
            train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_scores)
            test_fpr, test_tpr, test_thresholds = roc_curve(y_test, test_scores)

            #Return AUC for train and test
            train_auc = model_info['train_performance']['auc'].values[0]
            test_auc = model_info['test_performance']['auc'].values[0]

            #Plot train graph
            plt.subplot(1, 2, 1)
            plt.plot(train_fpr, train_tpr, linewidth = 2, label = f'{model_name} auc = {train_auc}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.02, 1.02, -0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Train Data')
            plt.legend()

            #Plot test graph
            plt.subplot(1, 2, 2)
            plt.plot(test_fpr, test_tpr, linewidth = 2, label = f'{model_name} auc = {test_auc}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.02, 1.02, -0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Test Data', size=12)
            plt.legend()

        plt.show()

    def custom_confusion_matrix(self, model_name, y_true, y_pred, classes, cmap, normalize = False):
       
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Return confusion matrix
        conf_mx = confusion_matrix(y_true, y_pred)

        #Plot matrix
        plt.imshow(conf_mx, interpolation = 'nearest', cmap = cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        #Customize axes
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        #Customize entries
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment = 'center',
                     color = 'white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{model_name}\nConfusion Matrix', size=12)

    def plot_confusion_matrix(self, classes, normalize=False, cmap=plt.cm.Blues):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        k = 1
        nrows = len(self.classifiers_info.keys())
        fig = plt.figure(figsize = (10, nrows * 4))
        sns.set(style = 'white', palette = 'muted', color_codes = True)

        # Iterating through each of the classifiers
        for model_name, model_info in self.classifiers_info.items():
            #Returning data on each model
            X_train = model_info['model_data']['X_train']
            y_train = model_info['model_data']['y_train']
            X_test = model_info['model_data']['X_test']
            y_test = model_info['model_data']['y_test']

            #Making predictions and returning confusion matrix
            train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv = 5)
            test_pred = model_info['estimator'].predict(X_test)

            #Plotting matrix (training data)
            plt.subplot(nrows, 2, k)
            self.custom_confusion_matrix(model_name + ' Train', y_train, train_pred, classes = classes, cmap = cmap,
                                         normalize = normalize)
            k += 1

            # Plotting matrix (test data)
            plt.subplot(nrows, 2, k)
            self.custom_confusion_matrix(model_name + ' Test', y_test, test_pred, classes = classes, cmap = plt.cm.Greens,
                                         normalize = normalize)
            k += 1

        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, model_name, ax, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Returning model to be evaluated
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classifier {model_name} was not trained.')
            print(f'Other options: {list(self.classifiers_info.keys())}')
            return None

        #Returning data on each model
        X_train = model['model_data']['X_train']
        y_train = model['model_data']['y_train']
        X_test = model['model_data']['X_test']
        y_test = model['model_data']['y_test']

        #Returning training and validation score parameters
        train_sizes, train_scores, val_scores = learning_curve(model['estimator'], X_train, y_train, cv = cv,
                                                               n_jobs = n_jobs, train_sizes = train_sizes)

        #Calculating means and standard deviations (training and validation)
        train_scores_mean = np.mean(train_scores, axis = 1)
        train_scores_std = np.std(train_scores, axis = 1)
        val_scores_mean = np.mean(val_scores, axis = 1)
        val_scores_std = np.std(val_scores, axis = 1)

        #Train results
        ax.plot(train_sizes, train_scores_mean, 'o-', color = 'navy', label = 'Training Score')
        ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                        alpha = 0.1, color = 'blue')

        #Cross validation results
        ax.plot(train_sizes, val_scores_mean, 'o-', color = 'red', label = 'Cross Val Score')
        ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                        alpha = 0.1, color = 'crimson')

        #Customize graph
        ax.set_title(f'Model {model_name} - Learning Curve', size=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc = 'best')

    def plot_score_distribution(self, model_name, shade = False):
        
        """
        Parameters
        ----------
        classifiers: set of classifiers in dictionary form [dict]
        X: array with the data to be used in the training [np.array]
        y: array with the target vector of the model [np.array]

        Return
        -------
        None
        """

        #Return model to be validated
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classifier {model_name} was not trained.')
            print(f'Other options: {list(self.classifiers_info.keys())}')
            return None

        #Returning sets from the model
        y_train = self.classifiers_info[model_name]['model_data']['y_train']
        y_test = self.classifiers_info[model_name]['model_data']['y_test']

        #Return test and train scores
        train_scores = self.classifiers_info[model_name]['train_scores']
        test_scores = self.classifiers_info[model_name]['test_scores']

        #Plot distribution of scores
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 5))
        sns.kdeplot(train_scores[y_train == 1], ax = axs[0], label = 'y=1', shade = shade, color = 'darkslateblue')
        sns.kdeplot(train_scores[y_train == 0], ax = axs[0], label = 'y=0', shade = shade, color = 'crimson')
        sns.kdeplot(test_scores[y_test == 1], ax = axs[1], label = 'y=1', shade = shade, color = 'darkslateblue')
        sns.kdeplot(test_scores[y_test == 0], ax = axs[1], label = 'y=0', shade = shade, color = 'crimson')

        #Customize plots
        axs[0].set_title('Score Distribution - Training Data', size = 12, color = 'dimgrey')
        axs[1].set_title('Score Distribution - Testing Data', size = 12, color = 'dimgrey')
        plt.suptitle(f'Score Distribution: a Probability Approach for {model_name}\n', size = 14, color = 'black')
        plt.show()

    def plot_score_bins(self, model_name, bin_range):

        #Return model to be validated
        try:
            model = self.classifiers_info[model_name]
        except:
            print(f'Classifier {model_name} was not trained.')
            print(f'Options available: {list(self.classifiers_info.keys())}')
            return None

        #Create array of bins
        bins = np.arange(0, 1.01, bin_range)
        bins_labels = [str(round(list(bins)[i - 1], 2)) + ' a ' + str(round(list(bins)[i], 2)) for i in range(len(bins))
                       if i > 0]

        #Return scores of training set to df
        train_scores = self.classifiers_info[model_name]['train_scores']
        y_train = self.classifiers_info[model_name]['model_data']['y_train']
        df_train_scores = pd.DataFrame({})
        df_train_scores['scores'] = train_scores
        df_train_scores['target'] = y_train
        df_train_scores['track'] = pd.cut(train_scores, bins, labels = bins_labels)

        #Calculating distribution by each track - training
        df_train_rate = pd.crosstab(df_train_scores['track'], df_train_scores['target'])
        df_train_percent = df_train_rate.div(df_train_rate.sum(1).astype(float), axis = 0)

        #Return scores of test set to df
        test_scores = self.classifiers_info[model_name]['test_scores']
        y_test = self.classifiers_info[model_name]['model_data']['y_test']
        df_test_scores = pd.DataFrame({})
        df_test_scores['scores'] = test_scores
        df_test_scores['target'] = y_test
        df_test_scores['track'] = pd.cut(test_scores, bins, labels=bins_labels)

        #Calculation distribution of each track - test
        df_test_rate = pd.crosstab(df_test_scores['track'], df_test_scores['target'])
        df_test_percent = df_test_rate.div(df_test_rate.sum(1).astype(float), axis = 0)

        #Initialize figure
        fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (16, 12))

        #Plot volume graphs of each class by range
        for df_scores, ax in zip([df_train_scores, df_test_scores], [axs[0, 0], axs[0, 1]]):
            sns.countplot(x = 'track', data = df_scores, hue = 'target', ax = ax, palette = ['darkslateblue', 'crimson'])
            AnnotateBars(n_dec = 0, color = 'dimgrey').vertical(ax)
            ax.legend(loc = 'upper right')

        #Plot the percentage of representation of each class by range
        for df_percent, ax in zip([df_train_percent, df_test_percent], [axs[1, 0], axs[1, 1]]):
            df_percent.plot(kind = 'bar', ax = ax, stacked = True, color = ['darkslateblue', 'crimson'], width = 0.6)

            #Customize plot
            for p in ax.patches:
                #Collecting parameters for labeling
                height = p.get_height()
                width = p.get_width()
                x = p.get_x()
                y = p.get_y()

                #Formatting collected parameters and inserting them into the graph
                label_text = f'{round(100 * height, 1)}%'
                label_x = x + width - 0.30
                label_y = y + height / 2
                ax.text(label_x, label_y, label_text, ha = 'center', va = 'center', color = 'white',
                        fontweight = 'bold', size = 10)
            format_spines(ax, right_border = False)

        #Final touch on graph
        axs[0, 0].set_title('Quantity of each Class by Range - Train', size = 12, color = 'dimgrey')
        axs[0, 1].set_title('Quantity of each Class by Range - Test', size = 12, color = 'dimgrey')
        axs[1, 0].set_title('Percentage of each Class by Range - Train', size = 12, color = 'dimgrey')
        axs[1, 1].set_title('Percentage of each Class by Range - Test', size = 12, color = 'dimgrey')
        plt.suptitle(f'Score Distribution by Range - {model_name}\n', size = 14, color = 'black')
        plt.tight_layout()
        plt.show()

