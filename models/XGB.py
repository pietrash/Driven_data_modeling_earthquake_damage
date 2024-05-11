import os
import threading

import xgboost as xgb
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, f1_score
from xgboost import plot_importance
from data.data import save_model, get_model, get_submission_format, save_submission
import cupy as cp
import optuna as opt
from optuna_dashboard import run_server


def train_model(X, y, params, print_score=False):
    # Adjust y
    y -= 1

    # Train model
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, y_train = X, y
    score = 0

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Test model
    # y_pred = model.predict(X_test)
    # score = f1_score(y_test, y_pred, average='micro')
    #
    # if print_score:
    #     print("Score:", score)
    #     print("\nClassification Report:")
    #     print(classification_report(y_test, y_pred))

    # Save model
    save_model('XGB', model, params, X.columns, score)


def grid_search(X, y, params, use_gpu=False, print_score=False):
    # Adjust y
    y -= 1

    # Perform grid search
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = xgb.XGBClassifier()

    grid_search_cv = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='f1_micro', n_jobs=5,
                                  verbose=10)
    grid_search_cv.fit(X_train, y_train)

    # Get the best model
    model = grid_search_cv.best_estimator_
    best_params = grid_search_cv.best_params_

    # Train the best Model
    # TODO check if necessary (isn't the model already trained from the grid search?)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')

    if print_score:
        print("Score:", score)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    save_model('XGB_grid_search', model, best_params, X.columns, score)


def bayesian_optimization(X, y, params, init_points, n_iter, print_score):
    # Adjust y
    y -= 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    def objective(n_estimators, learning_rate, max_depth, subsample):
        xgb_model = xgb.XGBClassifier(n_estimators=int(n_estimators),
                                      max_depth=int(max_depth),
                                      learning_rate=float(learning_rate),
                                      subsample=float(subsample))

        return cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="f1_micro").mean()

    optimizer = BayesianOptimization(f=objective, pbounds=params)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max['params']
    best_params_formatted = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'learning_rate': float(best_params['max_depth']),
        'subsample': float(best_params['subsample'])
    }

    model = xgb.XGBClassifier(**best_params_formatted)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')

    if print_score:
        print("Score:", score)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    save_model('XGB_bayes', model, best_params_formatted, X.columns, score)


def optuna(X, y, print_score=False):
    # Adjust y
    y -= 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    def objective(trial):
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'num_boost_round': trial.suggest_int('num_boost_round', 10, 600),
            # 'n_estimators': trial.suggest_int('n_estimators', 10, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'eta': trial.suggest_float('eta', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.05, 1.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 1.5),
            'gamma': trial.suggest_float('gamma', 0.1, 1.5),
            'random_state': 37
        }

        #
        # model = xgb.XGBClassifier(**params)
        # model.fit(X_train, y_train)
        #
        # y_pred = model.predict(X_test)
        # f1 = f1_score(y_test, y_pred, average='micro')

        #
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(params, dtrain, num_boost_round=params['num_boost_round'])
        y_pred = bst.predict(dtest)
        f1 = f1_score(y_test, y_pred, average='micro')

        #
        # model = xgb.XGBClassifier(
        #     random_state=37,
        #     **params
        # )
        #
        # model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=150, verbose=False)
        #
        # y_pred = model.predict(X_test)
        # f1 = f1_score(y_test, y_pred, average='micro')

        return f1

    storage = opt.storages.InMemoryStorage()
    study = opt.create_study(direction='maximize', storage=storage)
    thread = threading.Thread(target=run_server, args=(storage,))
    thread.start()
    study.optimize(objective, n_trials=1000, n_jobs=5)

    # Train the best model
    model = xgb.XGBClassifier(**study.best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')

    if print_score:
        print("Score:", score)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    save_model('XGB_optuna', model, study.best_params, X.columns, score)


def prediction(X, model_dir):
    model, _, _ = get_model(model_dir)

    # Make prediction
    y_pred = model.predict(X)

    # Adjust y
    y_pred += 1

    # Save results
    submission = get_submission_format()
    submission['damage_grade'] = y_pred
    save_submission(submission, model_dir)


def plot_feature_importance(model_dir, top_n_features=None):
    model, _, _ = get_model(model_dir)
    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    plot_importance(model, max_num_features=top_n_features, ax=ax)
    plt.title(f'Top {top_n_features if top_n_features is not None else ""} Importance Feature')
    plt.show()


def get_model_info(model_dir):
    model, params, features = get_model(model_dir)
    print(model)
    print(params)
    print(features)
