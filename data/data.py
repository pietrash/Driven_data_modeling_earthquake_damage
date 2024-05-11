import os
import joblib
import configparser
import shutil

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import data.transformers as t

config = configparser.ConfigParser()
config.read('config.ini')
DATA_DIR = config['DEFAULT']['DATA_DIR']
SAVED_MODELS_DIR = config['DEFAULT']['SAVED_MODELS_DIR']
SUBMISSIONS_DIR = config['DEFAULT']['SUBMISSIONS_DIR']


def get_submission_format():
    return pd.read_csv(f'{SUBMISSIONS_DIR}/submission_format.csv')


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

    train_data_x, test_data_x = pipeline_preprocessing(train_data_x, train_data_y, test_data_x)

    train_data_x.drop('building_id', axis=1, inplace=True)
    train_data_y.drop('building_id', axis=1, inplace=True)
    test_data_x.drop('building_id', axis=1, inplace=True)

    if encoded_y:
        train_data_y = pd.get_dummies(train_data_y['damage_grade'])

    if data == 'train':
        return train_data_x, train_data_y
    if data == 'test':
        return test_data_x


def get_model(path):
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


def save_submission(submission, model_dir):
    submission.to_csv(f'{SUBMISSIONS_DIR}/submission_{model_dir}.csv', index=False)


def pipeline_preprocessing(train_data_x, train_data_y, test_data_x):
    train_modified_data = train_data_x.copy()
    test_modified_data = test_data_x.copy()

    geo_level_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    numerical_columns = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
    categorical_columns = ['foundation_type', 'ground_floor_type', 'land_surface_condition',
                           'legal_ownership_status', 'other_floor_type',
                           'plan_configuration', 'position', 'roof_type']

    preprocessing = ColumnTransformer(
        transformers=[
            # ('fill_unk_geo_id', t.FillUnkWithMode(), geo_level_columns),
            ('freq_encoding', t.FrequencyEncoding(), geo_level_columns),
            ('target_probability', t.TargetProbability(), geo_level_columns),
            ('target_mean_grouped', t.TargetMeanGrouped(), geo_level_columns),
            ('target_mean', t.TargetMean(), geo_level_columns),
            ('one_hot_encoder', OneHotEncoder(sparse_output=False), categorical_columns)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough')
    preprocessing.set_output(transform='pandas')

    fill_unk = t.FillUnkWithMode()
    train_modified_data = fill_unk.fit_transform(train_modified_data)
    test_modified_data = fill_unk.transform(test_modified_data)

    train_data_modified = preprocessing.fit_transform(train_modified_data, train_data_y['damage_grade'])
    test_data_modified = preprocessing.transform(test_modified_data)

    return train_data_modified, test_data_modified
