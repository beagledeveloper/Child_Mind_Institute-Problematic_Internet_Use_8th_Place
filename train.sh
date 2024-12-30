#!/bin/bash

########################## training catboost ##########################
python ./scripts/train_cat_feateng_tsgroup_fillna_numerical_BayesianRidge.py > ./logs/train_cat_feateng_tsgroup_fillna_numerical_BayesianRidge.log
python ./scripts/train_cat_plain_fillna_numerical_BayesianRidge.py >  ./logs/train_cat_plain_fillna_numerical_BayesianRidge.log
python ./scripts/train_cat_plain_no-ts_fillna_numerical_BayesianRidge.py > ./logs/train_cat_plain_no-ts_fillna_numerical_BayesianRidge.log

########################## training lightgbm ##########################
python ./scripts/train_lgb_feateng_tsgroup_fillna_numerical_BayesianRidge.py > ./logs/train_lgb_feateng_tsgroup_fillna_numerical_BayesianRidge.log
python ./scripts/train_lgb_plain_fillna_numerical_BayesianRidge.py >  ./logs/train_lgb_plain_fillna_numerical_BayesianRidge.log
python ./scripts/train_lgb_plain_no-ts_fillna_numerical_BayesianRidge.py > ./logs/train_lgb_plain_no-ts_fillna_numerical_BayesianRidge.log

########################## training xgboost ##########################
python ./scripts/train_xgb_feateng_tsgroup_fillna_numerical_BayesianRidge.py > ./logs/train_xgb_feateng_tsgroup_fillna_numerical_BayesianRidge.log
python ./scripts/train_xgb_plain_fillna_numerical_BayesianRidge.py >  ./logs/train_xgb_plain_fillna_numerical_BayesianRidge.log
python ./scripts/train_xgb_plain_no-ts_fillna_numerical_BayesianRidge.py > ./logs/train_xgb_plain_no-ts_fillna_numerical_BayesianRidge.log

########################## training IterativeImputer(estimator=BayesianRidge) ##########################
python ./scripts/train_imputer_fillna_numerical_BayesianRidge.py > ./logs/train_imputer_fillna_numerical_BayesianRidge.log