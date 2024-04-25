import os
import joblib
import configparser
import pandas as pd
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

config = configparser.ConfigParser()
config.read('config.ini')
DATA_DIR = config['DEFAULT']['DATA_DIR']
SAVED_MODELS_DIR = config['DEFAULT']['SAVED_MODELS_DIR']


def get_train_values():
    return pd.read_csv(f'{DATA_DIR}/train_values.csv')


def get_train_labels():
    return pd.read_csv(f'{DATA_DIR}/train_labels.csv')


def get_test_values():
    return pd.read_csv(f'{DATA_DIR}/test_values.csv')


def get_train_data(encoded_x=False, encoded_y=False, sampling_method=None):
    data_x = preprocess_data(get_train_values())
    data_y = get_train_labels()
    data_y = data_y[data_y['building_id'].isin(data_x['building_id'])]
    ids = data_y[['building_id']]

    data_x.drop('building_id', axis=1, inplace=True)
    data_y.drop('building_id', axis=1, inplace=True)

    if sampling_method == 'oversampling':
        data_x, data_y = RandomOverSampler().fit_resample(data_x, data_y)
    if sampling_method == 'undersampling':
        data_x, data_y = RandomUnderSampler().fit_resample(data_x, data_y)

    if encoded_x:
        data_x = pd.get_dummies(data_x)
    if encoded_y:
        data_y = pd.get_dummies(data_y['damage_grade'])

    return data_x, data_y, ids


def get_test_data(encoded_x=False):
    data_x = preprocess_data(get_test_values())
    ids = data_x['building_id']

    data_x.drop('building_id', axis=1, inplace=True)

    if encoded_x:
        data_x = pd.get_dummies(data_x)
    return data_x, ids


def preprocess_data(data_x, bin_age=False):
    modified_data = data_x.copy()

    # Dropping all has_secondary_use columns but the main one
    # modified_data = modified_data.drop(
    #     columns=['has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental',
    #              'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry',
    #              'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police',
    #              'has_secondary_use_other'])
    #
    # # Adding new columns and deleting olds ones
    # modified_data['has_superstructure_with_mud'] = np.where(
    #     (modified_data['has_superstructure_adobe_mud'] == 0) & (
    #             modified_data['has_superstructure_mud_mortar_stone'] == 0) & (
    #             modified_data['has_superstructure_mud_mortar_brick'] == 0), 0, 1)
    # modified_data['has_superstructure_with_cement'] = np.where(
    #     (modified_data['has_superstructure_cement_mortar_brick'] == 0) & (
    #             modified_data['has_superstructure_cement_mortar_stone'] == 0), 0, 1)
    # modified_data['has_superstructure_rc'] = np.where(
    #     (modified_data['has_superstructure_rc_engineered'] == 0) & (
    #             modified_data['has_superstructure_rc_non_engineered'] == 0), 0, 1)
    # modified_data['plan_configuration_not_d'] = np.where(modified_data['plan_configuration'] == 'd', 0, 1)
    # modified_data = modified_data.drop(
    #     columns=['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
    #              'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
    #              'has_superstructure_cement_mortar_stone', 'has_superstructure_rc_engineered',
    #              'has_superstructure_rc_non_engineered', 'plan_configuration'])

    # modified_data = pd.get_dummies(modified_data)
    # modified_data = pd.merge(modified_data, get_train_labels(), on='building_id')
    # modified_data.drop('building_id', axis=1, inplace=True)
    # modified_data = modified_data[get_high_correlation_columns(modified_data, 'damage_grade', 0.0001)]

    # BIN AGE
    # if bin_age:
    #     # TODO bin age

    # SCALE NUMERIC VALUES
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
                  'count_floors_pre_eq', 'age', 'area_percentage',
                  'height_percentage']

    # scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
    #               'area_percentage', 'height_percentage']

    # scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    scaler.fit(get_train_values()[scaled_col])

    modified_data[scaled_col] = scaler.transform(modified_data[scaled_col])

    # ENCODE GEO ID
    # data_y should contain only damage_grade col
    def get_geo_id_means_std(data_x, data_y):
        X = data_x.copy()
        y = data_y.copy()

        for i in range(1, 4):
            X[f'geo_level_{i}_id_mean'] = pd.concat([X[f'geo_level_{i}_id'], y], axis=1).groupby(
                f'geo_level_{i}_id').transform('mean')

        for i in range(1, 4):
            X[f'geo_level_{i}_id_std'] = pd.concat([X[f'geo_level_{i}_id'], y], axis=1).groupby(
                f'geo_level_{i}_id').transform('std')

        return X

    modified_data = get_geo_id_means_std(modified_data, get_train_labels().drop('building_id', axis=1))

    return modified_data


def load_model(path):
    model = joblib.load(f'{SAVED_MODELS_DIR}/{path}/model.pkl')
    params = joblib.load(f'{SAVED_MODELS_DIR}/{path}/params.pkl')
    features = joblib.load(f'{SAVED_MODELS_DIR}/{path}/features.pkl')

    return model, params, features


def save_model(prefix, model, params, features, score):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = f'{SAVED_MODELS_DIR}/{prefix}_{time}_score_{round(score, 4)}'
    os.makedirs(directory)

    joblib.dump(params, f'{directory}/params.pkl')
    joblib.dump(features, f'{directory}/features.pkl')
    joblib.dump(model, f'{directory}/model.pkl')

    print(f'Model saved to: {directory}')
