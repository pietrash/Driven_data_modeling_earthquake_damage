import threading

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from data.data import save_model, get_model, get_submission_format, save_submission
import optuna as opt
from optuna_dashboard import run_server


def train_model(X, y, params, print_score=False):
    # Adjust y
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # X_train, y_train = X, y
    # score = 0

    model = RandomForestClassifier(**params)

    # Train the Model
    model.fit(X_train, y_train)

    # # Evaluate the Model
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')
    if print_score:
        print("Score:", score)
        print(classification_report(y_test, y_pred))

    # Save model
    save_model('RFC', model, params, X.columns, score)


def grid_search(X, y, params, print_score=False):
    # Adjust y
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=37)

    model = RandomForestClassifier(random_state=37)

    # Perform Grid Search
    grid_search_cv = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring='f1_micro', n_jobs=6, verbose=10)
    grid_search_cv.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search_cv.best_estimator_
    best_params = grid_search_cv.best_params_

    # Train the best model
    best_model.fit(X_train, y_train)

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')

    if print_score:
        print("Score:", score)
        print(classification_report(y_test, y_pred))

    save_model('RFC_grid', best_model, best_params, X.columns, score)


def bayesian_optimization(X, y, params, init_points, n_iter, print_score=False):
    # Adjust y
    y = y.values.ravel()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=37)

    def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_depth=int(max_depth),
                                       min_samples_split=int(min_samples_split),
                                       min_samples_leaf=int(min_samples_leaf),
                                       max_features=int(max_features)
                                       )

        return cross_val_score(model, x_train, y_train, cv=5, scoring="f1_micro", n_jobs=6).mean()

    optimizer = BayesianOptimization(f=objective, pbounds=params, random_state=37)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max['params']
    best_params_formatted = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'min_samples_split': int(best_params['min_samples_split']),
        'min_samples_leaf': int(best_params['min_samples_leaf']),
        'max_features': int(best_params['max_features'])
    }

    best_model = RandomForestClassifier(**best_params_formatted, random_state=37)
    best_model.fit(x_train, y_train)

    # Evaluate the best model
    y_pred = best_model.predict(x_test)
    score = f1_score(y_test, y_pred, average='micro')
    if print_score:
        print("Score:", score)
        print(classification_report(y_test, y_pred))

    save_model('RFC_bayes', best_model, best_params, X.columns, score)


def optuna(X, y, print_score=False):
    # Adjust y
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    def objective(trial):
        # Number of trees in random forest
        n_estimators = trial.suggest_int(name="n_estimators", low=100, high=400, step=10)

        # Number of features to consider at every split
        max_features = trial.suggest_float(name="max_features", low=0.0, high=0.5)

        # Maximum number of levels in tree
        max_depth = trial.suggest_int(name="max_depth", low=10, high=200, step=10)

        # Minimum number of samples required to split a node
        min_samples_split = trial.suggest_int(name="min_samples_split", low=2, high=20, step=1)

        # Minimum number of samples required at each leaf node
        min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=15, step=1)

        params = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')

        return f1

    storage = opt.storages.InMemoryStorage()
    study = opt.create_study(direction='maximize', storage=storage)
    thread = threading.Thread(target=run_server, args=(storage,))
    thread.start()
    study.optimize(objective, n_trials=100, n_jobs=5)

    # Train the best model
    model = RandomForestClassifier(**study.best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')

    if print_score:
        print("Score:", score)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    save_model('RFC_optuna', model, study.best_params, X.columns, score)


def prediction(X, model_dir):
    model, _, _ = get_model(model_dir)

    # Make prediction
    y_pred = model.predict(X)

    # Save results
    submission = get_submission_format()
    submission['damage_grade'] = y_pred
    save_submission(submission, model_dir)


def plot_feature_importance(model_dir, top_n_features=0):
    model, _, features = get_model(model_dir)
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(25, 25))
    plt.title(f'Top {top_n_features if top_n_features != 0 else ""} Feature Importances')
    plt.barh(range(len(indices[-top_n_features:])), model.feature_importances_[indices][-top_n_features:],
             color='b', align='center')
    plt.yticks(range(len(indices[-top_n_features:])), features[indices][-top_n_features:])
    plt.xlabel('Relative Importance')
    plt.savefig('importance_random_forest.png')
    plt.show()
