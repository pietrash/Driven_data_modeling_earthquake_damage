from models import XGB, RFC
from data import data


def xgb_train_model():
    X, y = data.get_data(
        data='train'
    )

    xgb_params = {'max_depth': 13, 'subsample': 0.9215269930113558,
                  'num_boost_round': 447,
                  'learning_rate': 0.021252258962912492,
                  'colsample_bytree': 0.42858394263757116,
                  'eta': 0.38341228669593524,
                  'reg_lambda': 1.4355819838765098,
                  'reg_alpha': 0.12648845868715758,
                  'gamma': 0.5069783649787669}

    XGB.train_model(
        X=X,
        y=y,
        params=xgb_params,
        print_score=True
    )


def xgb_bayes():
    X, y = data.get_data(
        data='train'
    )

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
    X, y = data.get_data(
        data='train'
    )
    xgb_param_grid = {
        'n_estimators': [400, 500, 750],
        'learning_rate': [0.01],
        'max_depth': [10, 12, 14, 16],
        'subsample': [0.7, 0.8, 0.9]
    }

    XGB.grid_search(
        X=X,
        y=y,
        params=xgb_param_grid,
        print_score=True
    )


def xgb_prediction():
    X = data.get_data(
        data='test'
    )

    XGB.prediction(
        X=X,
        model_dir='XGB_2024-05-06_11-16-00_score_0'
    )


def xgb_optuna():
    X, y = data.get_data(
        data='train'
    )
    # Params in objective

    XGB.optuna(
        X=X,
        y=y,
        print_score=True
    )


def xgb_visualization():
    XGB.plot_feature_importance(
        model_dir='XGB_optuna_2024-05-06_02-55-28_score_0.7562'
    )


def rfc_train_model():
    X, y = data.get_data(
        data='train'
    )
    params = {
        'n_estimators': 450,
        'max_depth': 22,
        'max_features': 14,
        'min_samples_split': 16,
        'min_samples_leaf': 4,
        'criterion': 'gini'
    }

    RFC.train_model(
        X=X,
        y=y,
        params=params,
        print_score=True
    )


def rfc_grid_search():
    X, y = data.get_data(
        data='train'
    )
    rfc_param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 15, 20, 30],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 3, 5, 10, 25],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', 8]
    }

    RFC.grid_search(
        X=X,
        y=y,
        params=rfc_param_grid,
        print_score=True
    )


def rfc_bayes():
    X, y = data.get_data(
        data='train'
    )
    rfc_param_bounds = {
        'n_estimators': (100, 600),
        'max_depth': (10, 30),
        'min_samples_split': (5, 50),
        'min_samples_leaf': (1, 30),
        'max_features': (1, 30)
    }

    RFC.bayesian_optimization(
        X=X,
        y=y,
        params=rfc_param_bounds,
        init_points=100,
        n_iter=50,
        print_score=True
    )


def rfc_optuna():
    X, y = data.get_data(
        data='train'
    )
    # Params in objective
    RFC.optuna(
        X=X,
        y=y,
        print_score=True
    )


def rfc_prediction():
    X = data.get_data(
        data='test'
    )

    RFC.prediction(
        X=X,
        model_dir='RFC_2024-05-02_21-17-43_score_0'
    )


def rfc_visualization():
    print(RFC.plot_feature_importance('RFC_2024-05-02_21-17-43_score_0'))


# xgb_train_model()
# xgb_bayes()
# xgb_grid_search()
# xgb_prediction()
# xgb_optuna()
# xgb_visualization()

# XGB.get_model_info('XGB_optuna_2024-05-06_02-55-28_score_0.7562')
# XGB.get_model_info('XGB_grid_search_2024-04-30_16-06-47_score_0.7582')

# rfc_grid_search()
# rfc_train_model()
# rfc_bayes()
rfc_optuna()
# rfc_prediction()
# rfc_visualization()
