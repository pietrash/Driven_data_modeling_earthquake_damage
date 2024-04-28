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


def get_train_data(encoded_y=False, sampling_method=None):
    data_x = get_train_values()
    data_y = get_train_labels()

    data_x = preprocess_data(data_x)
    data_y = data_y[data_y['building_id'].isin(data_x['building_id'])]

    data_x.drop('building_id', axis=1, inplace=True)
    data_y.drop('building_id', axis=1, inplace=True)

    if sampling_method == 'oversampling':
        data_x, data_y = RandomOverSampler().fit_resample(data_x, data_y)
    if sampling_method == 'undersampling':
        data_x, data_y = RandomUnderSampler().fit_resample(data_x, data_y)

    if encoded_y:
        data_y = pd.get_dummies(data_y['damage_grade'])

    return data_x, data_y


def get_test_data():
    data_x = preprocess_data(get_test_values())

    data_x.drop('building_id', axis=1, inplace=True)

    return data_x


def preprocess_data(data_x, bin_age=False):
    modified_data = data_x.copy()

    # ENCODE CATEGORICAL COLS
    modified_data = pd.get_dummies(modified_data)

    # ENCODE GEO ID
    def get_geo_id_mean(data):
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

            # Add geo_id_mean to data
            data = pd.merge(
                data,
                geo_id_mean[[f'geo_level_{i}_id', f'geo_level_{i}_id_mean']].drop_duplicates(),
                on=f'geo_level_{i}_id',
                how='left'
            )

        # Fill NA
        data['geo_level_2_id_mean'] = data['geo_level_2_id_mean'].fillna(data['geo_level_1_id_mean'])
        data['geo_level_3_id_mean'] = data['geo_level_3_id_mean'].fillna(data['geo_level_2_id_mean'])

        return data

    modified_data = get_geo_id_mean(modified_data)

    # SCALE NUMERIC VALUES
    def scale_numeric_cols(data):
        # scaler = StandardScaler()
        scaler = MinMaxScaler()

        # scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
        #               'count_floors_pre_eq', 'age', 'area_percentage',
        #               'height_percentage']

        # scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
        #               'area_percentage', 'height_percentage']

        scaled_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

        scaler.fit(get_train_values()[scaled_col])

        data[scaled_col] = scaler.transform(modified_data[scaled_col])

        return data

    modified_data = scale_numeric_cols(modified_data)

    # KEEP ONLY TOP 20 HIGH IMPORTANCE COLS (and building_id for data_y filtering)
    high_importance_cols = [
        'building_id',
        'area_percentage',
        'geo_level_3_id',
        'age',
        'geo_level_2_id',
        'geo_level_3_id_mean',
        'geo_level_2_id_mean',
        'height_percentage',
        'geo_level_1_id',
        'geo_level_1_id_mean',
        'count_families',
        'count_floors_pre_eq',
        'position_s',
        'has_superstructure_timber',
        'roof_type_n',
        'has_superstructure_mud_mortar_stone',
        'other_floor_type_q',
        'has_secondary_use',
        'has_superstructure_cement_mortar_brick',
        'position_t',
        'land_surface_condition_n',
        'foundation_type_r'
    ]

    modified_data = modified_data[high_importance_cols]

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

    # BIN AGE
    # if bin_age:
    #     # TODO bin age

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
