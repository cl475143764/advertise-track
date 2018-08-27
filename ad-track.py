import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import numpy as np
from datetime import datetime
start = datetime.now()
VALIDATE =      True
RANDOM_STATE = 50
VALID_SIZE = 0.90
MAX_ROUNDS = 650
EARLY_STOP = 650
OPT_ROUNDS = 650
skiprows = range(0)
nrows = 100000
output_filename = 'submission.csv'
dtypes = {
    'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
}

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv("train_sample.csv", skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=train_cols,parse_dates=['click_time'])
len_train = len(train_df)
gc.collect()

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
middle1_freq_hours_in_test_data = [16, 17, 22]
least_freq_hours_in_test_data = [6, 11, 15]


def prep_data(df):
    new_feature = 'nextClick'
    D = 2 ** 26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df[
        'device'].astype(str)+ "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, t in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - t)
        click_buffer[category] = t
    del (click_buffer)
    QQ = list(reversed(next_clicks))
    df[new_feature] = QQ
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()

    df['in_test_hh'] = (4
                        - 3 * df['hour'].isin(most_freq_hours_in_test_data)
                        - 2 * df['hour'].isin(middle1_freq_hours_in_test_data)
                        - 1 * df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day', 'in_test_hh'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hh'})
    df = df.merge(gp, on=['ip', 'day', 'in_test_hh'], how='left')
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')

    del gp
    gc.collect()

    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_hh'})
    df = df.merge(gp, on=['ip', 'day', 'hour'], how='left')
    df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
    del gp
    gc.collect()

    gp = df[['ip', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'hour'])[['channel']].count().reset_index().rename(
        index=str, columns={'channel': 'nip_hh_os'})
    df = df.merge(gp, on=['ip', 'os', 'hour'], how='left')
    df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
    del gp
    gc.collect()

    gp = df[['ip', 'app', 'hour', 'channel']].groupby(by=['ip', 'app', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_app'})
    df = df.merge(gp, on=['ip', 'app', 'hour'], how='left')
    df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
    del gp
    gc.collect()

    gp = df[['ip', 'device', 'hour', 'channel']].groupby(by=['ip', 'device', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_dev'})
    df = df.merge(gp, on=['ip', 'device', 'hour'], how='left')
    df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
    del gp

    gc.collect()

    df.drop(['ip', 'day'], axis=1, inplace=True)

    gc.collect()
    print('df',df)
    return df

print('==================================开始训练=========================================================')
train_df = prep_data(train_df)
gc.collect()

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 9,
    'max_depth': 5,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'min_split_gain': 0,
    'nthread': 8,
    'verbose': 0,
    'scale_pos_weight': 99.7,
}

target = 'is_attributed'
predictors = ['app', 'device', 'os', 'channel', 'hour', 'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app',
              'nip_hh_dev','nextClick']
categorical = ['app', 'device', 'os', 'channel', 'hour']

if VALIDATE:

    train_df, val_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True)
    dtrain = lgb.Dataset(train_df[predictors].values,
                         label=train_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)
    del train_df
    gc.collect()

    dvalid = lgb.Dataset(val_df[predictors].values,
                         label=val_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)
    del val_df
    gc.collect()

    evals_results = {}

    model = lgb.train(params,
                      dtrain,
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      evals_result=evals_results,
                      num_boost_round=650,
                      early_stopping_rounds=EARLY_STOP,
                      verbose_eval=50,
                      feval= None)
    del dvalid

else:

    gc.collect()
    dtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical
                         )
    del train_df
    gc.collect()

    evals_results = {}

    model = lgb.train(params,
                      dtrain,
                      valid_sets=[dtrain],
                      valid_names=['train'],
                      evals_result=evals_results,
                      num_boost_round=OPT_ROUNDS,
                      verbose_eval=50,
                      feval=None)
    print('AUC' + ":", evals_results['valid']['AUC'][model.best_iteration - 1])
del dtrain
gc.collect()
count=0
print("=========================================载入测试集================================================")
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
test_dfer = pd.read_csv("test.csv", dtype=dtypes, usecols=test_cols,chunksize=100000,parse_dates=['click_time'])
for test_df in test_dfer:
    test_df = prep_data(test_df)
    gc.collect()
    count+=1
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id']
    sub['is_attributed'] = model.predict(test_df[predictors])
    sub.to_csv('%doutput_filename-.csv'%count, index=False,)

print('==' * 35)
print('============================ 结果预览 ============================')
print('==' * 35)
print(datetime.now(), '\n')
print('{:^17} : {:}'.format('train time', datetime.now() - start))
print('{:^17} : {:}'.format('output file', output_filename))
print('{:^17} : {:.5f}'.format('train auc', model.best_score['train']['auc']))
if VALIDATE:
    print('{:^17} : {:.5f}\n'.format('valid auc', model.best_score['valid']['auc']))
    print(
        '{:^17} : {:}\n{:^17} : {}\n{:^17} : {}'.format('VALIDATE', VALIDATE, 'VALID_SIZE', VALID_SIZE, 'RANDOM_STATE',
                                                        RANDOM_STATE))
print(
    '{:^17} : {:}\n{:^17} : {}\n{:^17} : {}\n'.format('MAX_ROUNDS', MAX_ROUNDS, 'EARLY_STOP', EARLY_STOP, 'OPT_ROUNDS',
                                                      model.best_iteration))
print('{:^17} : {:}\n{:^17} : {}\n'.format('skiprows', skiprows, 'nrows', nrows))
print('{:^17} : {:}\n{:^17} : {}\n'.format('variables', predictors, 'categorical', categorical))
print('{:^17} : {:}\n'.format('model params', params))
print('==' * 35)