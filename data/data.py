import os
import joblib
import configparser
import shutil

import numpy as np
import pandas as pd
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

config = configparser.ConfigParser()
config.read('config.ini')
DATA_DIR = config['DEFAULT']['DATA_DIR']
SAVED_MODELS_DIR = config['DEFAULT']['SAVED_MODELS_DIR']
SUBMISSIONS_DIR = config['DEFAULT']['SUBMISSIONS_DIR']


def get_submission_format():
    return pd.read_csv(f'{SUBMISSIONS_DIR}/submission_format.csv')


def save_submission(submission, model_dir):
    submission.to_csv(f'{SUBMISSIONS_DIR}/submission_{model_dir}.csv', index=False)


def get_train_values():
    return pd.read_csv(f'{DATA_DIR}/train_values.csv')


def get_train_labels():
    return pd.read_csv(f'{DATA_DIR}/train_labels.csv')


def get_test_values():
    return pd.read_csv(f'{DATA_DIR}/test_values.csv')


def get_data(encoded_y=False, data='train'):
    train_data_x = get_train_values()
    train_data_y = get_train_labels()
    test_data_x = get_test_values()

    train_data_x, test_data_x = preprocess_data(train_data_x, test_data_x)

    train_data_x.drop('building_id', axis=1, inplace=True)
    train_data_y.drop('building_id', axis=1, inplace=True)
    test_data_x.drop('building_id', axis=1, inplace=True)

    if encoded_y:
        train_data_y = pd.get_dummies(train_data_y['damage_grade'])

    if data == 'train':
        return train_data_x, train_data_y
    if data == 'test':
        return test_data_x

    return


def preprocess_data(train_data_x, test_data_x):
    train_modified_data = train_data_x.copy()
    test_modified_data = test_data_x.copy()

    # FILL NA GEO ID IN TEST SET WITH MODE OF THE TRAIN SET
    def fill_na_geo_id(train_data, test_data):
        cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

        for col in cols:
            unique_train_ids = set(train_data[col])
            test_data[col] = test_data[col].apply(
                lambda x: x if x in unique_train_ids else train_data[col].mode()[0])

        return train_data, test_data

    train_modified_data, test_modified_data = fill_na_geo_id(train_modified_data, test_modified_data)

    # FREQUENCY ENCODING GEO ID
    def freq_encoding(train_modified_data, test_modified_data):
        cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

        freq_dict = {}
        for col in cols:
            freq_dict[col] = dict(train_modified_data[col].value_counts())
            train_modified_data[col + '_freq'] = train_modified_data[col].apply(lambda x: freq_dict[col][x])
            test_modified_data[col + '_freq'] = test_modified_data[col].apply(lambda x: freq_dict[col][x])
        return train_modified_data, test_modified_data

    train_modified_data, test_modified_data = freq_encoding(train_modified_data, test_modified_data)

    # DROP SECONDARY USE COLUMNS
    def drop_secondary_use(train_data, test_data):
        def modify_data(data):
            # Dropping all has_secondary_use columns but the main one
            data.drop(
                columns=['has_secondary_use_use_police', 'has_secondary_use_gov_office', 'has_secondary_use_rental',
                         'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry',
                         'has_secondary_use_health_post', 'has_secondary_use_gov_office',
                         'has_secondary_use_use_police',
                         'has_secondary_use_other'],
                inplace=True
            )
            return data

        return modify_data(train_data), modify_data(test_data)

    train_modified_data, test_modified_data = drop_secondary_use(train_modified_data, test_modified_data)

    # MERGE HAS SUPERSTRUCTURE COLUMNS (also plan configuration)
    def merge_has_superstructure(train_data, test_data):
        def modify_data(data):
            # Adding new columns and deleting olds ones
            data['has_superstructure_with_mud'] = np.where(
                (data['has_superstructure_adobe_mud'] == 0) & (
                        data['has_superstructure_mud_mortar_stone'] == 0) & (
                        data['has_superstructure_mud_mortar_brick'] == 0), 0, 1)
            data['has_superstructure_with_cement'] = np.where(
                (data['has_superstructure_cement_mortar_brick'] == 0) & (
                        data['has_superstructure_cement_mortar_stone'] == 0), 0, 1)
            data['has_superstructure_rc'] = np.where(
                (data['has_superstructure_rc_engineered'] == 0) & (
                        data['has_superstructure_rc_non_engineered'] == 0), 0, 1)
            data['plan_configuration_not_d'] = np.where(data['plan_configuration'] == 'd', 0, 1)
            data = data.drop(
                columns=['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                         'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
                         'has_superstructure_cement_mortar_stone', 'has_superstructure_rc_engineered',
                         'has_superstructure_rc_non_engineered', 'plan_configuration'])
            return data

        return modify_data(train_data), modify_data(test_data)

    train_modified_data, test_modified_data = merge_has_superstructure(train_modified_data, test_modified_data)

    # ONE HOT ENCODING
    train_modified_data = pd.get_dummies(train_modified_data)
    test_modified_data = pd.get_dummies(test_modified_data)

    # CREATE GEO_LOCATION_MEAN - mean damage_grade for each location
    def get_geo_location_mean(train_data, test_data):
        # Get mean for each geolocation
        geo_id_mean = pd.concat(
            objs=[
                get_train_values()[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']],
                get_train_labels()['damage_grade']
            ],
            axis=1)
        geo_id_mean['geo_location_mean'] = geo_id_mean.groupby(['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'])[
            'damage_grade'].transform('mean')

        def modify_data(data):
            # Add geo location mean to data
            data = pd.merge(
                data,
                geo_id_mean[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'geo_location_mean']],
                on=['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'],
                how='left'
            ).drop_duplicates().reset_index(drop=True)
            return data

        return modify_data(train_data), modify_data(test_data)

    train_modified_data, test_modified_data = get_geo_location_mean(train_modified_data, test_modified_data)

    def get_geo_id_mean(train_data, test_data):
        geo_id_mean = pd.concat(
            objs=[
                get_train_values()[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']],
                get_train_labels()['damage_grade']
            ],
            axis=1)

        for i in range(1, 4):
            # Calculate geo_id_mean
            geo_id_mean[f'geo_level_{i}_id_mean'] = geo_id_mean.groupby(
                f'geo_level_{i}_id')['damage_grade'].transform('mean')

        def modify_data(data):
            # Add geo_id_mean to data
            for i in range(1, 4):
                data = pd.merge(
                    data,
                    geo_id_mean[[f'geo_level_{i}_id', f'geo_level_{i}_id_mean']].drop_duplicates(),
                    on=f'geo_level_{i}_id',
                    how='left'
                )
            return data

        return modify_data(train_data), modify_data(test_data)

    train_modified_data, test_modified_data = get_geo_id_mean(train_modified_data, test_modified_data)

    # SCALE NUMERIC VALUES
    def scale_numeric_cols(train_data, test_data):
        # scaler = StandardScaler()
        scaler = MinMaxScaler()

        scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
                      'count_floors_pre_eq', 'age', 'area_percentage',
                      'height_percentage']

        # scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
        #               'area_percentage', 'height_percentage']

        # scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

        scaler.fit(train_data[scaled_col])

        train_data[scaled_col] = scaler.transform(train_data[scaled_col])
        test_data[scaled_col] = scaler.transform(test_data[scaled_col])

        return train_data, test_data

    # train_modified_data, test_modified_data = scale_numeric_cols(train_modified_data, test_modified_data)

    return train_modified_data, test_modified_data


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

    # Copy data.py file, so it will be known what data preparation was done
    shutil.copyfile(f'data/data.py', f'{directory}/data.py')

    print(f'Model saved to: {directory}')
