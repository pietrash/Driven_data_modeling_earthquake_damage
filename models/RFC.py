import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from data.data import get_train_data, save_model, load_model


def train_model(X, y, params, print_score=False):
    # Adjust y
    y = y.values.ravel()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = RandomForestClassifier(**params)

    # Train the Model
    model.fit(x_train, y_train)

    # Evaluate the Model
    y_pred = model.predict(x_test)
    score = f1_score(y_test, y_pred, average='micro')
    if print_score:
        print("Score:", score)
        print(classification_report(y_test, y_pred))

    # Save model
    save_model('RFC', model, params, X.columns, score)


def grid_search(X, y, params, print_score=False):
    # Adjust y
    y = y.values.ravel()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=37)

    model = RandomForestClassifier(random_state=37)

    # Perform Grid Search
    grid_search_cv = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='f1_micro', n_jobs=3, verbose=10)
    grid_search_cv.fit(x_train, y_train)

    # Get the best model
    best_model = grid_search_cv.best_estimator_
    best_params = grid_search_cv.best_params_

    # Train the best model
    best_model.fit(x_train, y_train)

    # Evaluate the best model
    y_pred = best_model.predict(x_test)
    score = f1_score(y_test, y_pred)

    if print_score:
        print("Score:", score)
        print(classification_report(y_test, y_pred))

    save_model('RFC_grid', best_model, best_params, X.columns, score)


def bayesian_optimization(X, y, params, init_points, n_iter, print_score=False):
    # Adjust y
    y = y.values.ravel()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=37)

    def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_depth=int(max_depth),
                                       min_samples_split=int(min_samples_split),
                                       min_samples_leaf=int(min_samples_leaf)
                                       )

        return cross_val_score(model, x_train, y_train.values.ravel(), cv=5, scoring="f1_micro").mean()

    optimizer = BayesianOptimization(f=objective, pbounds=params, random_state=37)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max['params']
    best_params_formatted = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'min_samples_split': int(best_params['min_samples_split']),
        'min_samples_leaf': int(best_params['min_samples_leaf'])
    }

    best_model = RandomForestClassifier(**best_params_formatted, random_state=37)
    best_model.fit(x_train, y_train)

    # Evaluate the best model
    y_pred = best_model.predict(x_test)
    score = f1_score(y_test, y_pred)
    if print_score:
        print("Score:", score)
        print(classification_report(y_test, y_pred))

    save_model('RFC_bayes', best_model, best_params, X.columns, score)


def plot_feature_importance(model, features, top_n_features=0):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(25, 25))
    plt.title(f'Top {top_n_features if top_n_features != 0 else ""} Feature Importances')
    plt.barh(range(len(indices[-top_n_features:])), model.feature_importances_[indices][-top_n_features:],
             color='b', align='center')
    plt.yticks(range(len(indices[-top_n_features:])), features[indices][-top_n_features:])
    plt.xlabel('Relative Importance')
    # plt.savefig('importance_random_forest.png')
    plt.show()
