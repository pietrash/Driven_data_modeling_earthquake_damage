import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from data.data_preparation import get_train_data, save_model, load_model


def train_model():
    x, y, _ = get_train_data(encoded_x=True, encoded_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Choose a Model
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'max_features': 25,
        'min_samples_split': 16,
        'min_samples_leaf': 2
    }

    model = RandomForestClassifier(**params)

    # Train the Model
    model.fit(x_train, y_train)

    # Evaluate the Model
    y_pred = model.predict(x_test)
    score = f1_score(y_test, y_pred, average='micro')
    print("Score:", score)

    # Print classification report for detailed evaluation
    print(classification_report(y_test, y_pred))

    save_model('RFC', model, params, x.columns, score)


def grid_search():
    x, y, _ = get_train_data(encoded_x=True, encoded_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Define parameters for grid search
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        # 'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [None, 10, 15, 20, 25, 30, 35],
        # 'min_samples_split': [3, 4, 5],
        # 'min_samples_leaf': [1, 2, 4],
        # 'criterion': ['gini', 'entropy'],
    }

    model = RandomForestClassifier(random_state=6)

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1_micro', n_jobs=3, verbose=10)
    grid_search.fit(x_train, y_train.values.ravel())

    # Get the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Train the Best Model
    best_model.fit(x_train, y_train.values.ravel())

    # Evaluate the Best Model
    y_pred = best_model.predict(x_test)
    score = f1_score(y_test, y_pred)
    print("Score:", score)

    # Print classification report for detailed evaluation
    print(classification_report(y_test, y_pred))

    save_model('RFC_grid', best_model, best_params, x.columns, score)


def bayesian_optimization():
    x, y, _ = get_train_data(encoded_x=True, encoded_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_depth=int(max_depth),
                                       min_samples_split=int(min_samples_split),
                                       min_samples_leaf=int(min_samples_leaf)
                                       )

        return cross_val_score(model, x_train, y_train.values.ravel(), cv=3, scoring="f1_micro").mean()

    # Bounds for hyperparameters
    param_bounds = {
        'n_estimators': (400, 600),
        'max_depth': (20, 70),
        'min_samples_split': (5, 50),
        'min_samples_leaf': (1, 30),
    }

    optimizer = BayesianOptimization(f=objective, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=50, n_iter=25)

    best_params = optimizer.max['params']

    best_params_formatted = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'min_samples_split': int(best_params['min_samples_split']),
        'min_samples_leaf': int(best_params['min_samples_leaf'])
    }
    best_model = RandomForestClassifier(**best_params_formatted, random_state=42)
    best_model.fit(x_train, y_train.values.ravel())
    score = best_model.score(x_test, y_test)
    print(f"Score: {score}")

    save_model('RFC_bayes', best_model, best_params, x.columns, score)


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


train_model()
# grid_search()
# bayesian_optimization()

# model, _, features = load_model('RFC_2024-04-25_12-44-50_score_0.7529')
# df = pd.DataFrame(columns=features)
# plot_feature_importance(model, features)
