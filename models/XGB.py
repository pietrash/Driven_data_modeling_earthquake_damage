import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, f1_score
from xgboost import plot_importance
from data.data_preparation import get_train_data, save_model, load_model, get_test_data


def train_model():
    params = {
        'encoded_x': True,
        'encoded_y': False,
        'sampling_method': 'oversampling',
        'n_estimators': 1073,
        'max_depth': 22,
        'learning_rate': 0.02584,
        'subsample': 0.8778
    }

    # Load data
    X, y, _ = get_train_data(
        encoded_x=params['encoded_x'],
        encoded_y=params['encoded_y'],
        sampling_method=params['sampling_method']
    )

    # Adjust y for XGB
    y -= 1

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample']
    )
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')
    print("Score:", score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    save_model('XGB', model, params, X.columns, score)


def grid_search_():
    params = {
        'encoded_x': True,
        'encoded_y': False,
        'sampling_method': 'oversampling',
    }

    X, y, _ = get_train_data(
        encoded_x=params['encoded_x'],
        encoded_y=params['encoded_y'],
        sampling_method=params['sampling_method']
    )

    y -= 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 1000, 2000],
        'learning_rate': [0.01, 0.1],
        'max_depth': [8, 9, 10, 11, 12, 13, 14, 15],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }

    model = xgb.XGBClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1_micro', n_jobs=5, verbose=10)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Train the Best Model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')
    print("Score:", score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    save_model('XGB_grid_search', model, best_params, X.columns, score)


def bayesian_optimization():
    params = {
        'encoded_x': True,
        'encoded_y': False,
        'sampling_method': 'oversampling',
    }

    X, y, _ = get_train_data(
        encoded_x=params['encoded_x'],
        encoded_y=params['encoded_y'],
        sampling_method=params['sampling_method']
    )

    y -= 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_bounds = {
        'n_estimators': (100, 1500),
        'learning_rate': (0.001, 1),
        'max_depth': (5, 25),
        'subsample': (0.1, 1),
    }

    def objective(n_estimators, learning_rate, max_depth, subsample):
        model = xgb.XGBClassifier(n_estimators=int(n_estimators),
                                  max_depth=int(max_depth),
                                  learning_rate=float(learning_rate),
                                  subsample=float(subsample))

        return cross_val_score(model, X_train, y_train, cv=3, scoring="f1_micro").mean()

    optimizer = BayesianOptimization(f=objective, pbounds=param_bounds)
    optimizer.maximize(init_points=50, n_iter=25)

    best_params = optimizer.max['params']

    # Best was
    # |  target   | learni... | max_depth | n_esti... | subsample |
    # | 0.8466    | 0.02584   | 22.17     | 1.073e+03 | 0.8778
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
    print("Score:", score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    save_model('XGB_bayes', model, best_params_formatted, X.columns, score)


def prediction():
    # Load data
    x, ids = get_test_data(
        encoded_x=True,
        # encoded_y=False
    )

    # no norm
    # XGB_2024-04-22_19-14-28_score_0.8176
    # Score: 0.8132355593416756

    # STANDARD SCALER
    # all norm
    # XGB_2024-04-22_19-16-00_score_0.8168
    # Score: 0.8132662576122117

    # norm ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage']
    # XGB_2024-04-22_19-17-51_score_0.817
    # Score: 0.8134274235325267

    # norm ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    # XGB_2024-04-22_19-19-21_score_0.8161
    # Score: 0.812801946270352

    # MIN MAX SCALER
    # all norm
    # XGB_2024-04-22_19-47-16_score_0.8183
    # Score: 0.8142754632560888

    # norm ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage']
    # XGB_2024-04-22_19-48-35_score_0.8151
    # Score: 0.8124258924562837

    # norm ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    # XGB_2024-04-22_19-49-58_score_0.8161
    # Score: 0.8143944190544166

    # Make prediction
    model, _, _ = load_model('XGB_2024-04-24_13-12-49_score_0.8771')
    y_pred = model.predict(x)

    # Adjust prediction values
    y_pred += 1

    # Print score
    # score = f1_score(y, y_pred, average='micro')
    # print("Score:", score)
    # print("\nClassification Report:")
    # print(classification_report(y, y_pred))

    # Save results
    submission = pd.read_csv('data/submission_format.csv')
    submission['damage_grade'] = y_pred
    submission.to_csv('data/submission.csv', index=False)

    # Plot feature importance
    # plot_feature_importance(model)


def plot_feature_importance(model, top_n_features=None):
    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
    plot_importance(model, max_num_features=top_n_features, ax=ax)
    plt.title(f'Top {top_n_features if top_n_features is not None else ""} Importance Feature')
    plt.show()


# train_model()
# prediction()
# grid_search_()
# bayesian_optimization()

model, _, _ = load_model('XGB_2024-04-22_19-17-51_score_0.817')
plot_feature_importance(model)

# modele
# normalizacja
# samplingi
# czysczenie danych
# korelacja
# analiza składowych głównych
# hiperparametryzacja (grid search, bayes)
