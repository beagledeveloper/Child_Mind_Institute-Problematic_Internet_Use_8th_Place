import os
import copy
import json

from joblib import dump, load

from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.optimize import minimize
import numpy as np

import optuna

import lightgbm as lgb
import catboost as cb
import xgboost as xgb


class Config:
    feature_cols = [
        'Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
        'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
        'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
        'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
        'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
        'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
        'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
        'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
        'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
        'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
        'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
        'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
        'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
        'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
        'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
        'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
        'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
        'PreInt_EduHx-computerinternet_hoursday'
    ]

    cat_cols = [
        'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
        'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season'
    ]

    remove_ts_columns = {
        'X': ['count'],
        'Y': ['count'],
        'Z': ['count'],
        'enmo': ['count'],
        'anglez': ['count'],
        'non-wear_flag': ['count'],
        'light': ['count'],
        # 特徴量エンジニアリング
        'XYZ': ['count'],
        'abs_X': ['count'],
        'abs_Y': ['count'],
        'abs_Z': ['count'],
        'abs_anglez': ['count']
    }

    drop_ts_columns = ['step', 'battery_voltage', 'time_of_day', 'weekday', 'quarter', 'relative_date_PCIAT']

    n_splits = 5

    seed = 42

    k_fold_seeds = [777, 42, 1732]

    bins = [0, 21600, 43200, 64800, 86400]

    select_feature_cols1 = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'BIA-BIA_Activity_Level_num', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday', 'light_max']
    select_feature_cols2 = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'PAQ_A-PAQ_A_Total', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday', 'X_25%_0-21600', 'Y_std', 'Z_mean_0-21600', 'Z_min_43200-64800', 'Z_75%_64800-86400', 'enmo_50%_43200-64800', 'non-wear_flag_mean_0-21600', 'light_std', 'light_max_21600-43200', 'XYZ_50%_21600-43200', 'XYZ_mean_43200-64800', 'XYZ_mean_64800-86400', 'XYZ_std_64800-86400', 'XYZ_50%_64800-86400', 'abs_X_50%_21600-43200', 'abs_X_25%_64800-86400', 'abs_X_75%_64800-86400', 'abs_Y_75%_0-21600', 'abs_Y_min_21600-43200', 'abs_Y_75%_64800-86400', 'abs_Z_75%_21600-43200', 'abs_Z_75%_43200-64800', 'abs_anglez_min_43200-64800', 'BMI_Age', 'Internet_Hours_Age', 'Muscle_to_Fat']
    select_feature_cols3 = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight', 'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'BIA-BIA_Activity_Level_num', 'PAQ_A-PAQ_A_Total', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday']
    
    lgb_dir1 = 'lgb/plain'
    xgb_dir1 = 'xgb/plain'

    xgb_dir2 = 'xgb/feateng_tsgroup'

    lgb_dir3 = 'lgb/plain_no-ts'
    xgb_dir3 = 'xgb/plain_no-ts'
    cat_dir3 = 'cat/plain_no-ts'

    imputer_dir = 'imputer'

    threshold = [0.57721265, 1.01302488, 2.66556245]

def update(df):
    for c in Config.cat_cols: 
        df[c] = df[c].fillna('Missing')
        df[c] = df[c].astype('category')
    return df

def create_mapping(column, df):
    unique_values = df[column].unique()
    return {value: idx for idx, value in enumerate(unique_values)}

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

def ts_feature_engineering(df):
    df['XYZ'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    df['abs_X'] = np.abs(df['X'])
    df['abs_Y'] = np.abs(df['Y'])
    df['abs_Z'] = np.abs(df['Z'])
    df['abs_anglez'] = np.abs(df['anglez'])

    # 時刻を0〜86500にスケーリング
    df['time_of_day'] = (df['time_of_day'] // 1e+9).astype(np.int64)    
    df['time_of_day_bins'] = pd.cut(df['time_of_day'], bins=Config.bins)

    return df

def create_TimeSeries():
    # JSONファイルを開いて読み込む
    with open("./SETTINGS.json", "r", encoding="utf-8") as file:
        test_dir = json.load(file)['TEST_DATA_CLEAN_PATH']  # JSONファイルをPythonの辞書型に変換
    ts_dir = os.path.join(test_dir, 'series_test.parquet')
    # TimeSeriesのIDリストを作成(id=を除去)
    ts_dirs = os.listdir(os.path.join(test_dir, 'series_test.parquet'))
    ts_ids = [s.split('=')[1] for s in ts_dirs]

    ts_records = []
    for ts_id in tqdm(ts_ids):
        # 時系列データの読み込み
        ts_df = pd.read_parquet(os.path.join(ts_dir, f'id={ts_id}/part-0.parquet'))
        # 時系列要約特徴量エンジニアリング
        ts_df = ts_feature_engineering(ts_df)
        # 不要な特徴量を削減
        ts_df.drop(Config.drop_ts_columns, axis=1, inplace=True)
        
        record = {}
        # 特徴量ごとに統計量を算出
        for feature in ts_df.columns.tolist():
            if feature == 'time_of_day_bins':
                continue
            desc = ts_df[feature].describe()
            # 不要な統計量を除去
            desc = desc.drop(Config.remove_ts_columns[feature])
            for stat_name, value in desc.items():
                record[f'{feature}_{stat_name}'] = value

            # 時間帯でグルーピングして、統計量を算出
            for time_range_idx in range(len(Config.bins) - 1):
                desc_time_range = ts_df.groupby('time_of_day_bins', observed=False)[feature].describe().iloc[time_range_idx]
                # 不要な統計量を除去
                desc_time_range = desc_time_range.drop(Config.remove_ts_columns[feature])
                for stat_name, value in desc_time_range.items():
                    record[f'{feature}_{stat_name}_{Config.bins[time_range_idx]}-{Config.bins[time_range_idx+1]}'] = value

        record['id'] = ts_id

        ts_records.append(record)

    ts_df = pd.DataFrame(ts_records)

    return ts_df

def feature_engineering(df):
    df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
    df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
    df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
    df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
    df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
    df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
    df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
    df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
    df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
    df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
    df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
    
    return df

if __name__ == '__main__':
    # JSONファイルを開いて読み込む
    with open("./SETTINGS.json", "r", encoding="utf-8") as file:
        settings = json.load(file)  # JSONファイルをPythonの辞書型に変換

    # 数値特徴量リストを作成
    num_cols = [item for item in Config.feature_cols if item not in Config.cat_cols]

    # 学習データの読み込み
    df = pd.read_csv(os.path.join(settings["TEST_DATA_CLEAN_PATH"], 'test.csv'))
    train_df = pd.read_csv(os.path.join(settings["TRAIN_DATA_CLEAN_PATH"], 'train.csv'))

    # カテゴリ変数のエンコード
    df = update(df)
    train_df = update(train_df)
    for cat_col in Config.cat_cols:
        mapping = create_mapping(cat_col, train_df)
        df[cat_col] = df[cat_col].replace(mapping).astype(int)

    # 時系列データフレームを作成
    ts_df = create_TimeSeries()

    # 時系列データを結合
    df = pd.merge(df, ts_df, how="left", on='id')

    # 時系列特徴量を保存
    ts_features = df[ts_df.drop('id', axis=1).columns.tolist()]

    ids = df['id'].to_numpy()
    preds = np.zeros((len(df)), np.float64)

    # 異なるランダムSeedで層化分別
    for i, k_fold_seed in tqdm(enumerate(Config.k_fold_seeds)):
        # クロスバリデーション
        for k in range(Config.n_splits):
            # モデル読み込み
            imputer = load(os.path.join(settings["MODEL_CHECKPOINT_DIR"], Config.imputer_dir, str(k_fold_seed), str(k), 'model.joblib'))

            lgb_model1 = load(os.path.join(settings["MODEL_CHECKPOINT_DIR"], Config.lgb_dir1, str(k_fold_seed), str(k), 'model.joblib'))
            xgb_model1 = load(os.path.join(settings["MODEL_CHECKPOINT_DIR"], Config.xgb_dir1, str(k_fold_seed), str(k), 'model.joblib'))

            xgb_model2 = load(os.path.join(settings["MODEL_CHECKPOINT_DIR"], Config.xgb_dir2, str(k_fold_seed), str(k), 'model.joblib'))

            lgb_model3 = load(os.path.join(settings["MODEL_CHECKPOINT_DIR"], Config.lgb_dir3, str(k_fold_seed), str(k), 'model.joblib'))
            cat_model3 = load(os.path.join(settings["MODEL_CHECKPOINT_DIR"], Config.cat_dir3, str(k_fold_seed), str(k), 'model.joblib'))
            xgb_model3 = load(os.path.join(settings["MODEL_CHECKPOINT_DIR"], Config.xgb_dir3, str(k_fold_seed), str(k), 'model.joblib'))

            # 入力データ準備
            X = df.copy()
            X.loc[:, num_cols] = imputer.transform(X[num_cols])

            # 特徴量エンジニアリング
            X = feature_engineering(X)

            # 入力特徴量作成
            X1 = X[Config.select_feature_cols1]
            X2 = X[Config.select_feature_cols2]
            X3 = X[Config.select_feature_cols3]

            # 検証
            preds += (lgb_model1.predict(X1, num_iteration=lgb_model1.best_iteration_) * 0.2 + xgb_model1.predict(X1) * 0.2 \
                        + xgb_model2.predict(X2) * 0.2 \
                            + lgb_model3.predict(X3, num_iteration=lgb_model3.best_iteration_) * 0.2 + cat_model3.predict(X3) * 0.1 + xgb_model3.predict(X3) * 0.1)

    preds /= (len(Config.k_fold_seeds) * Config.n_splits)
    oof_tuned = threshold_Rounder(preds, Config.threshold)
    sub_df = pd.DataFrame({
        'id': ids,
        'sii': oof_tuned
    })
    sub_df.to_csv(os.path.join(settings["SUBMISSION_DIR"], 'submission.csv'), index=False)