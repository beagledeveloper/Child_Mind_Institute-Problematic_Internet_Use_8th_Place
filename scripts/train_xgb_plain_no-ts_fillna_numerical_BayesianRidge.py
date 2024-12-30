import os
import copy
import json

from joblib import dump

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
        'X': ['count', 'mean', '50%'],
        'Y': ['count', 'mean', 'std', '50%'],
        'Z': ['count', 'mean', 'min', '50%', '75%'],
        'enmo': ['count', 'min', '25%', '50%'],
        'anglez': ['count', 'mean', 'min', '75%', 'max'],
        'non-wear_flag': ['count', 'min', '75%', 'max'],
        'light': ['count', 'mean', 'min', '25%', '50%', '75%'],
        # 特徴量エンジニアリング
        'XYZ': ['count', '25%', '50%', '75%'],
        'abs_X': ['count', 'std', 'min', '50%', '75%'],
        'abs_Y': ['count', 'std', 'min', '50%', '75%'],
        'abs_Z': ['count', 'std', 'min', '50%', '75%'],
        'abs_anglez': ['count', 'std', 'min', '75%', 'max']
    }

    drop_ts_columns = ['step', 'battery_voltage', 'time_of_day', 'weekday', 'quarter', 'relative_date_PCIAT']

    params_base = {
        'random_state': 1234,
        'learning_rate': 0.05,
        'n_estimators': 10000,
        'tree_method': 'gpu_hist',
        'early_stopping_rounds': 50,
        'verbosity': 0
    }

    nb_runs = 1000

    na_nums_th = 30

    n_splits = 5
    n_trials = 200

    seed = 42

    k_fold_seeds = [777, 42, 1732]

    bins = [0, 21600, 43200, 64800, 86400]

    select_feature_cols = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-Height', 'Physical-Weight', 'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_PU', 'BIA-BIA_Activity_Level_num', 'PAQ_A-PAQ_A_Total', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday']
    

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

    return df

def create_TimeSeries():
    # JSONファイルを開いて読み込む
    with open("./SETTINGS.json", "r", encoding="utf-8") as file:
        train_dir = json.load(file)['TRAIN_DATA_CLEAN_PATH']  # JSONファイルをPythonの辞書型に変換
    ts_dir = os.path.join(train_dir, 'series_train.parquet')
    # TimeSeriesのIDリストを作成(id=を除去)
    ts_dirs = os.listdir(os.path.join(train_dir, 'series_train.parquet'))
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
            desc = ts_df[feature].describe()
            # 不要な統計量を除去
            desc = desc.drop(Config.remove_ts_columns[feature])
            for stat_name, value in desc.items():
                record[f'{feature}_{stat_name}'] = value

        record['id'] = ts_id

        ts_records.append(record)

    ts_df = pd.DataFrame(ts_records)

    return ts_df

def cv_models(Datasets, feature_cols):
    # JSONファイルを開いて読み込む
    with open("./SETTINGS.json", "r", encoding="utf-8") as file:
        output_dir = json.load(file)  # JSONファイルをPythonの辞書型に変換
    output_dir = os.path.join(output_dir['MODEL_CHECKPOINT_DIR'], 'xgb', 'plain_no-ts')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 教師データと予測値を格納するNumpy配列を定義
    y_gt = np.array([])
    y_preds = np.array([])

    # 異なるランダムSeedで層化分別
    for i, k_fold_seed in enumerate(Config.k_fold_seeds):
        # ランダムSeed毎にフォルダを作成
        if not os.path.exists(os.path.join(output_dir, str(k_fold_seed))):
            os.makedirs(os.path.join(output_dir, str(k_fold_seed)))

        def objective(trial):
            params_tuning = {
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample' : trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
        }

            params_tuning.update(Config.params_base)

            val_scores = []

            # クロスバリデーション
            for k in range(Config.n_splits):
                train_idx, val_idx = splits[k]

                # データ取得
                X_tr, X_va, y_tr, y_va = Datasets[i][k]
                X_tr, X_va = X_tr[feature_cols], X_va[feature_cols]

                # モデル生成
                model = xgb.XGBRegressor(**params_tuning)

                # 学習
                model.fit(
                    X_tr.values, y_tr.values,
                    eval_set=[(X_va.values, y_va.values)],   # バリデーションセットを指定
                    verbose=False
                )

                # 検証
                y_preds = model.predict(X_va)

                # スコア値保存
                val_scores.append(mean_squared_error(y_va, y_preds))

            return np.mean(val_scores)
        
        # ハイパラ最適化
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), direction='minimize')
        study.optimize(objective, n_trials=Config.n_trials)

        # 最適なハイパラを取得
        trial = study.best_trial
        params_best = trial.params
        params_best.update(Config.params_base)

        # クロスバリデーション
        for k in range(Config.n_splits):
            # Fold毎にフォルダを作成
            if not os.path.exists(os.path.join(output_dir, str(k_fold_seed), str(k))):
                os.makedirs(os.path.join(output_dir, str(k_fold_seed), str(k)))

            train_idx, val_idx = splits[k]

            # データ取得
            X_tr, X_va, y_tr, y_va = Datasets[i][k]
            X_tr, X_va = X_tr[feature_cols], X_va[feature_cols]

            # モデル生成
            model = xgb.XGBRegressor(**params_best)
            # 学習
            model.fit(
                X_tr.values, y_tr.values,
                eval_set=[(X_va.values, y_va.values)],   # バリデーションセットを指定
                verbose=False
            )

            # モデル保存
            dump(model, os.path.join(os.path.join(output_dir, str(k_fold_seed), str(k), 'model.joblib')))

            # 検証
            y_pred = model.predict(X_va)

            # 予測値を保存
            y_gt = np.concatenate((y_gt, y_va))
            y_preds = np.concatenate((y_preds, y_pred))

    KappaOPtimizer = minimize(evaluate_predictions,
                                x0=[0.5, 1.5, 2.5], args=(y_gt, y_preds), 
                                method='Nelder-Mead') # Nelder-Mead | # Powell
    oof_tuned = threshold_Rounder(y_preds, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y_gt, oof_tuned)

    print(f"Optimized QWK SCORE :: {tKappa:.3f}", flush=True)

if __name__ == '__main__':
    # 数値特徴量リストを作成
    num_cols = [item for item in Config.feature_cols if item not in Config.cat_cols]

    # ログレベルを非表示に設定
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # JSONファイルを開いて読み込む
    with open("./SETTINGS.json", "r", encoding="utf-8") as file:
        train_dir = json.load(file)['TRAIN_DATA_CLEAN_PATH']  # JSONファイルをPythonの辞書型に変換
    
    # 学習データの読み込み
    df = pd.read_csv(os.path.join(train_dir, 'train.csv'))

    # カテゴリ変数のエンコード
    df = update(df)
    for cat_col in Config.cat_cols:
        mapping = create_mapping(cat_col, df)
        df[cat_col] = df[cat_col].replace(mapping).astype(int)

    # 時系列データフレームを作成
    ts_df = create_TimeSeries()

    # 時系列データを結合
    df = pd.merge(df, ts_df, how="left", on='id')

    # id列削除
    df = df.drop('id', axis=1)

    # 欠損値の数を特徴量として作成
    df['missing_count'] = df[num_cols].isnull().sum(axis=1)

    # sii(target列)がNaNの行を削除
    df = df.dropna(subset='sii')
    df = df.reset_index(drop=True)

    # 時系列特徴量を保存
    ts_features = df[ts_df.drop('id', axis=1).columns.tolist()]

    print(Config.select_feature_cols)

    # データセットを作成
    Datasets = [[] for _ in range(len(Config.k_fold_seeds))]

    # 異なるランダムSeedで層化分別
    for i, k_fold_seed in tqdm(enumerate(Config.k_fold_seeds)):
        # 説明変数、目的変数に分割
        X = df[Config.feature_cols]
        y = df['sii']

        # 時系列特徴量を保存
        ts_features = df[ts_df.drop('id', axis=1).columns.tolist()]

        # 層化分別
        skf = StratifiedKFold(n_splits=Config.n_splits, shuffle=True, random_state=k_fold_seed)
        splits = list(skf.split(X, y))

        # クロスバリデーション
        for k in range(Config.n_splits):
            train_idx, val_idx = splits[k]

            # 学習データ取得
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]

            # 検証データ取得
            X_va = X.iloc[val_idx]
            y_va = y.iloc[val_idx]

            # Imputerの学習データ取得
            imputer_train_df = df.iloc[train_idx]

            # 欠損値補完
            # IterativeImputerのインスタンスを作成
            imputer = IterativeImputer(random_state=0, max_iter=100)

            # データをフィットして欠損値を補完
            imputer.fit(imputer_train_df[imputer_train_df['missing_count']<=Config.na_nums_th][num_cols])

            X_tr = np.concatenate((imputer.transform(X_tr[num_cols]), X_tr[Config.cat_cols].to_numpy()), axis=1)
            X_va = np.concatenate((imputer.transform(X_va[num_cols]), X_va[Config.cat_cols].to_numpy()), axis=1)
            X_tr = pd.DataFrame(X_tr, columns=num_cols + Config.cat_cols)
            X_va = pd.DataFrame(X_va, columns=num_cols + Config.cat_cols)

            # 時系列特徴量を学習データと検証データに分割
            ts_features_tr = ts_features.iloc[train_idx].reset_index(drop=True)
            ts_features_va = ts_features.iloc[val_idx].reset_index(drop=True)

            # 時系列要約特徴量と結合
            X_tr = pd.concat([X_tr, ts_features_tr], axis=1).copy()
            X_va = pd.concat([X_va, ts_features_va], axis=1).copy()

            # Datasetsに追加
            Datasets[i].append((X_tr, X_va, y_tr.copy(), y_va.copy()))

    cv_models(Datasets, Config.select_feature_cols)