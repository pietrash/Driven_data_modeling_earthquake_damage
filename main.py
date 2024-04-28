from sklearn.ensemble import RandomForestRegressor

from models import XGB, RFC
from data import data


def xgb_bayes():
    X, y = data.get_train_data()

    xgb_param_bounds = {
        'n_estimators': (100, 800),
        'learning_rate': (0.001, 1),
        'max_depth': (5, 25),
        'subsample': (0.2, 1),
    }

    XGB.bayesian_optimization(
        X=X,
        y=y,
        params=xgb_param_bounds,
        init_points=25,
        n_iter=10,
        print_score=True
    )


def xgb_grid_search():
    X, y = data.get_train_data()

    xgb_param_grid = {
        'n_estimators': [100, 300, 400, 500, 750],
        'learning_rate': [0.01, 0.1],
        'max_depth': [10, 12, 14, 16, 18, 20],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }

    XGB.grid_search(
        X=X,
        y=y,
        params=xgb_param_grid,
        print_score=True
    )


def xgb_prediction():
    X = data.get_test_data()

    XGB.prediction(
        X=X,
        model_dir='XGB_grid_search_2024-04-28_06-58-46_score_0.7588'
    )


def xgb_visualization():
    XGB.plot_feature_importance(
        model_dir='XGB_grid_search_2024-04-28_06-58-46_score_0.7588'
    )


def rfc_train_model():
    X, y = data.get_train_data()

    params = {
        'n_estimators': 100,
        'max_depth': None,
        'max_features': 25,
        'min_samples_split': 16,
        'min_samples_leaf': 2
    }

    RFC.train_model(
        X=X,
        y=y,
        params=params,
        print_score=True
    )


def rfc_grid_search():
    X, y = data.get_train_data()

    rfc_param_grid = {
        'n_estimators': [200, 300, 400, 500],
        # 'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [None, 10, 15, 20, 25, 30, 35],
        # 'min_samples_split': [3, 4, 5],
        # 'min_samples_leaf': [1, 2, 4],
        # 'criterion': ['gini', 'entropy'],
    }

    RFC.grid_search(
        X=X,
        y=y,
        params=rfc_param_grid,
        print_score=True
    )


def rfc_bayes():
    X, y = data.get_train_data()

    rfc_param_bounds = {
        'n_estimators': (400, 600),
        'max_depth': (20, 70),
        'min_samples_split': (5, 50),
        'min_samples_leaf': (1, 30),
    }

    RFC.bayesian_optimization(
        X=X,
        y=y,
        params=rfc_param_bounds,
        init_points=50,
        n_iter=25,
        print_score=True
    )


# xgb_bayes()
xgb_grid_search()
# xgb_prediction()
# xgb_visualization()
