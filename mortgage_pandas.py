import sys
import time
import numpy as np
import xgboost as xgb
import pandas as pd
import daal4py
from collections import OrderedDict
from pandas.core.dtypes.dtypes import CategoricalDtype

from numba import gdb_init, gdb_breakpoint


dt64_fill = np.dtype('datetime64[ns]').type('1970-01-01').astype('datetime64[ns]')

# to download data for this script,
# visit https://rapidsai.github.io/demos/datasets/mortgage-data
# and update the following paths accordingly
if len(sys.argv) != 4:
    raise ValueError("needed to point path to mortgage folder, "
                     "count quarter to process and ML framework")
else:
    mortgage_path = sys.argv[1]
    count_quarter_processing = int(sys.argv[2])
    ml_fw = sys.argv[3]

    
acq_data_path = mortgage_path + "/acq"
perf_data_path = mortgage_path + "/perf"
col_names_path = mortgage_path + "/names.csv"


import hpat
import numba
import ctypes

fflush = ctypes.CDLL(None).fflush
fflush.argtypes = (ctypes.c_void_p,)
fflush.restype = ctypes.c_int
@numba.jit(nopython=False)
def do_log(msg):
    print(msg)
    fflush(0)
#   f = open('log.txt','a')
 #  f.write(msg + '\n')
  # f.close()

def null_workaround(df):#, **kwargs):
    return df
    '''
    for column, data_type in df.dtypes.items():
        if str(data_type) == "category":
            df[column] = df[column].cat.codes
        if str(data_type) in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            df[column] = df[column].fillna(np.dtype(data_type).type(-1))
    return df
    '''


def run_cpu_workflow(quarter=1, year=2000, perf_file="", limit_rows=False):#, **kwargs):
    names = pd_load_names()
    do_log('FOO')
    acq_gdf = cpu_load_acquisition_csv(acq_data_path + "/Acquisition_"
                                      + str(year) + "Q" + str(quarter) + ".txt", limit_rows)
    acq_gdf = acq_gdf.merge(names, how='left', on=['seller_name'])
    acq_gdf = acq_gdf.drop(['seller_name'], axis=1)
    acq_gdf['seller_name'] = acq_gdf['new']
    acq_gdf = acq_gdf.drop(['new'], axis=1)
    do_log('FOO')

    perf_df_tmp = cpu_load_performance_csv(perf_file, limit_rows)
    do_log('FOO')
    gdf = perf_df_tmp
    everdf = create_ever_features(gdf)
    delinq_merge = create_delinq_features(gdf)
    everdf = join_ever_delinq_features(everdf, delinq_merge)
    do_log('FOO')
    #del(delinq_merge)

    joined_df = create_joined_df(gdf, everdf)
    do_log('FOO')

    testdf = create_12_mon_features(joined_df)
    do_log('FOO')
    joined_df = combine_joined_12_mon(joined_df, testdf)
    final_gdf = joined_df
    return final_gdf
    do_log('FOO')
    #del(testdf)

    perf_df = final_performance_delinquency(gdf, joined_df)
    do_log('FOO')
    #del(gdf, joined_df)

    final_gdf = join_perf_acq_gdfs(perf_df, acq_gdf)
    do_log('FOO')
    #del(perf_df)
    #del(acq_gdf)

    final_gdf = last_mile_cleaning(final_gdf)
    do_log('FOO')
    return final_gdf


def cpu_load_performance_csv(performance_path, limit_rows):#, **kwargs):
    """ Loads performance data

    Returns
    -------
    GPU DataFrame
    """
    
    cols = [
        "loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
        "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
        "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
        "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
        "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
        "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
        "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount", "servicing_activity_indicator"
    ]
    
    dtypes = { 
        "loan_id": np.int64,
        "monthly_reporting_period": str,# "datetime64"),
        "servicer": str, #CategoricalDtype(categories=[]),
        "interest_rate": np.float64,
        "current_actual_upb": np.float64,
        "loan_age": np.float64,
        "remaining_months_to_legal_maturity": np.float64,
        "adj_remaining_months_to_maturity": np.float64,
        "maturity_date": str,# "datetime64"),
        "msa": np.float64,
        "current_loan_delinquency_status": np.int32,
        "mod_flag": CategoricalDtype(['N', 'Y']),
        "zero_balance_code": CategoricalDtype(['01', '02', '06', '09', '03', '15', '16']),
        "zero_balance_effective_date": str,# "datetime64"),
        "last_paid_installment_date": str,# "datetime64"),
        "foreclosed_after": str,# "datetime64"),
        "disposition_date": str,# "datetime64"),
        "foreclosure_costs": np.float64,
        "prop_preservation_and_repair_costs": np.float64,
        "asset_recovery_costs": np.float64,
        "misc_holding_expenses": np.float64,
        "holding_taxes": np.float64,
        "net_sale_proceeds": np.float64,
        "credit_enhancement_proceeds": np.float64,
        "repurchase_make_whole_proceeds": np.float64,
        "other_foreclosure_proceeds": np.float64,
        "non_interest_bearing_upb": np.float64,
        "principal_forgiveness_upb": np.float64,
        "repurchase_make_whole_proceeds_flag": CategoricalDtype(['N', 'Y']),
        "foreclosure_principal_write_off_amount": np.float64,
        "servicing_activity_indicator": CategoricalDtype(['N', 'Y']),
    }
    dates_only = [1, 8, 13, 14, 15, 16]
    if limit_rows:
        nrows = 10
    else:
        nrows = None
        print(performance_path)

    return pd.read_csv(performance_path, dtype=dtypes, parse_dates=dates_only,
                       names=cols, delimiter='|', index_col=True, nrows=nrows)


def cpu_load_acquisition_csv(acq_path, limit_rows):#, **kwargs):
    """ Loads acquisition data

    Returns
    -------
    GPU DataFrame
    """
    
    cols = [
        'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term', 
        'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score', 
        'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
        'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type', 
        'relocation_mortgage_indicator'
    ]
    
    dtypes = {
        "loan_id": np.int64,
        "orig_channel": CategoricalDtype(['B', 'C', 'R']),
        "seller_name": str, 
        "orig_interest_rate": np.float64,
        "orig_upb": np.int64,
        "orig_loan_term": np.int64,
        "orig_date": str,
        "first_pay_date": str,
        "orig_ltv": np.float64,
        "orig_cltv": np.float64,
        "num_borrowers": np.float64,
        "dti": np.float64,
        "borrower_credit_score": np.float64,
        "first_home_buyer": CategoricalDtype(['N', 'U', 'Y']),
        "loan_purpose": CategoricalDtype(['C', 'P', 'R', 'U']),
        "property_type": CategoricalDtype(['CO', 'CP', 'MH', 'PU', 'SF']),
        "num_units": np.int64,
        "occupancy_status": CategoricalDtype(['I', 'P', 'S']),
        "property_state": CategoricalDtype(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO',
                                            'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
                                            'IA', 'ID', 'IL', 'IN', 'KS', 'KY',
                                            'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
                                            'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                                            'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                                            'OK', 'OR', 'PA', 'PR', 'RI', 'SC',
                                            'SD', 'TN', 'TX', 'UT', 'VA', 'VI',
                                            'VT', 'WA', 'WI', 'WV', 'WY']),
        "zip": np.int64,
        "mortgage_insurance_percent": np.float64,
        "product_type": CategoricalDtype(['FRM']),
        "coborrow_credit_score": np.float64,
        "mortgage_insurance_type": np.float64,
        "relocation_mortgage_indicator": CategoricalDtype(['N', 'Y']),
    }
    dates_only = [6, 7]
    if not limit_rows:
        print(acq_path)
    return pd.read_csv(acq_path, dtype=dtypes, parse_dates=dates_only,
                       names=cols, delimiter='|', index_col=False)


def pd_load_names():#**kwargs):
    """ Loads names used for renaming the banks
    
    Returns
    -------
    GPU DataFrame
    """

    cols = [
        'seller_name', 'new'
    ]
    
    dtypes = { 
        "seller_name": str,# CategoricalDtype(categories=[]),
        "new": str,# CategoricalDtype(categories=[]),
    }
    #dtypes1 = {}
    #for col, valtype in dtypes.items():
    #    dtypes1[col] = pd.core.dtypes.common.pandas_dtype(valtype)

    return pd.read_csv(col_names_path, names=cols, dtype=dtypes, delimiter='|')


def create_ever_features(gdf):#, **kwargs):
    everdf = gdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id', as_index=False).max()
    #del(gdf)
    everdf['ever_30'] = (everdf['current_loan_delinquency_status'] >= 1).astype(np.int8)
    everdf['ever_90'] = (everdf['current_loan_delinquency_status'] >= 3).astype(np.int8)
    everdf['ever_180'] = (everdf['current_loan_delinquency_status'] >= 6).astype(np.int8)
    everdf = everdf.drop(['current_loan_delinquency_status'], axis=1)
    return everdf


def create_delinq_features(gdf):#, **kwargs):
    delinq_gdf = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    #del(gdf)
    delinq_30 = delinq_gdf[delinq_gdf['current_loan_delinquency_status'] >= 1][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    delinq_30 = delinq_30.drop(['monthly_reporting_period'], axis=1)
    delinq_90 = delinq_gdf[delinq_gdf['current_loan_delinquency_status'] >= 3][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    delinq_90 = delinq_90.drop(['monthly_reporting_period'], axis=1)
    delinq_180 = delinq_gdf[delinq_gdf['current_loan_delinquency_status'] >= 6][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    delinq_180 = delinq_180.drop(['monthly_reporting_period'], axis=1)
    #del(delinq_gdf)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'])
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(dt64_fill)
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'])
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(dt64_fill)
    #del(delinq_30)
    #del(delinq_90)
    #del(delinq_180)
    return delinq_merge


def join_ever_delinq_features(everdf_tmp, delinq_merge):#, **kwargs):
    everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left')
    #del(everdf_tmp)
    #del(delinq_merge)
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(dt64_fill)
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(dt64_fill)
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(dt64_fill)
    return everdf


def create_joined_df(gdf, everdf):#, **kwargs):
    test = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    #del(gdf)
    test['timestamp'] = test['monthly_reporting_period']
    test = test.drop(['monthly_reporting_period'], axis=1)
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test = test.drop(['current_loan_delinquency_status'], axis=1)
    test['upb_12'] = test['current_actual_upb']
    test = test.drop(['current_actual_upb'], axis=1)
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)
    
    joined_df = test.merge(everdf, how='left', on=['loan_id'])
    #del(everdf)
    #del(test)
    
    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(dt64_fill)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(dt64_fill)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(dt64_fill)
    
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype(np.int32)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype(np.int32)
    
    return joined_df


def _create_month_features(joined_df, y):
    n_months = 12
    tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
    tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
    tmpdf['josh_mody_n'] = np.floor((tmpdf['josh_months'].astype(np.float64) - 24000 - y) / 12)
    #tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
    tmpdf_d = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False)['delinquency_12'].max()
    tmpdf_m = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False)['upb_12'].min()
    tmpdf_d['upb_12'] = tmpdf_m['upb_12']
    tmpdf1 = tmpdf_d
    tmpdf1['delinquency_12'] = (tmpdf1['delinquency_12']>3).astype(np.int32)
    tmpdf1['delinquency_12'] +=(tmpdf1['upb_12']==0).astype(np.int32)
    #tmpdf.drop('max_delinquency_12', axis=1)
    #tmpdf['upb_12'] = tmpdf['min_upb_12']
    #tmpdf.drop('min_upb_12', axis=1)
    tmpdf1['timestamp_year'] = np.floor(((tmpdf1['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).astype(np.int16)
    tmpdf1['timestamp_month'] = np.int8(y)
    return tmpdf.drop(['josh_mody_n'], axis=1)


def create_12_mon_features(joined_df):#, **kwargs):
    concats = [_create_month_features(joined_df, 1),
               _create_month_features(joined_df, 2),
               _create_month_features(joined_df, 3),
               _create_month_features(joined_df, 4),
               _create_month_features(joined_df, 5),
               _create_month_features(joined_df, 6),
               _create_month_features(joined_df, 7),
               _create_month_features(joined_df, 8),
               _create_month_features(joined_df, 9),
               _create_month_features(joined_df, 10),
               _create_month_features(joined_df, 11),
               _create_month_features(joined_df, 12)
    ]
    return pd.concat(concats)
    

def combine_joined_12_mon(joined_df, testdf):#, **kwargs):
    joined_df = joined_df.drop(['delinquency_12'], axis=1)
    joined_df = joined_df.drop(['upb_12'], axis=1)
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype(np.int16)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype(np.int8)
    return joined_df.merge(testdf, how='left',
                           on=['loan_id', 'timestamp_year', 'timestamp_month'])


def final_performance_delinquency(gdf, joined_df):#, **kwargs):
    merged = null_workaround(gdf)
    joined_df = null_workaround(joined_df)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype(np.int8)
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype(np.int16)
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype(np.int8)
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype(np.int16)
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])
    merged = merged.drop(['timestamp_year'], axis=1)
    merged = merged.drop(['timestamp_month'], axis=1)
    return merged


def join_perf_acq_gdfs(perf, acq):#, **kwargs):
    perf = null_workaround(perf)
    acq = null_workaround(acq)
    return perf.merge(acq, how='left', on=['loan_id'])


def last_mile_cleaning(df):#, **kwargs):
    drop_list = [
        'loan_id', 'orig_date', 'first_pay_date', 'seller_name',
        'monthly_reporting_period', 'last_paid_installment_date', 'maturity_date', 'ever_30', 'ever_90', 'ever_180',
        'delinquency_30', 'delinquency_90', 'delinquency_180', 'upb_12',
        'zero_balance_effective_date','foreclosed_after', 'disposition_date','timestamp'
    ]
    df.drop(drop_list, axis=1, inplace=True)
    #for column in drop_list:
    #    df.drop([column], axis=1, inplace=True)
    #for col, dtype in df.dtypes.iteritems():
    #    if str(dtype)=='category':
    #        df[col] = df[col].cat.codes
    #    #df[col] = df[col].astype('float32')
    df['delinquency_12'] = df['delinquency_12'] > 0
    df['delinquency_12'] = df['delinquency_12'].fillna(False).astype(np.int32)
    #for column in df.columns:
    #    df[column] = df[column].fillna(np.dtype(str(df[column].dtype)).type(-1))
    return df


def train_daal(pd_df):
    dxgb_daal_params = {
		'fptype':                       'float',
		'maxIterations':                100,
		'maxTreeDepth':                 8,
		'minSplitLoss':                 0.1,
		'shrinkage':                    0.1,
		'observationsPerTreeFraction':  1,
		'lambda_':                      1,
		'minObservationsInLeafNode':    1,
		'maxBins':                      256,
		'featuresPerNode':              0,
		'minBinSize':                   5,
		'memorySavingMode':             False,
	}


    y = np.ascontiguousarray(pd_df["delinquency_12"], dtype=np.float32).reshape(len(pd_df), 1)
    x = np.ascontiguousarray(pd_df.drop(["delinquency_12"], axis=1), dtype=np.float32)

    train_algo = daal4py.gbt_regression_training(**dxgb_daal_params)
    train_result = train_algo.compute(x, y)
    return train_result


def train_xgb(pd_df):
    dxgb_cpu_params = {
        'nround':            100,
        'max_depth':         8,
        'max_leaves':        2**8,
        'alpha':             0.9,
        'eta':               0.1,
        'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'min_child_weight':  30,
        'tree_method':       'hist',
        #n_gpus':            1,
        # 'distributed_dask':  True,
        'loss':              'ls',
        'objective':         'reg:linear',
        'max_features':      'auto',
        'criterion':         'friedman_mse',
        'grow_policy':       'lossguide',
        'verbose':           True
    }
    y = pd_df['delinquency_12']
    x = pd_df.drop(['delinquency_12'], axis=1)
    dtrain = xgb.DMatrix(x, y)
    model_xgb = xgb.train(dxgb_cpu_params, dtrain,
                          num_boost_round=dxgb_cpu_params['nround'])
    return model_xgb

if 'hpat' in ml_fw:
    import hpat
    run_cpu_workflow = hpat.jit(locals={'final_gdf:return': 'distributed'}, debug=True)(run_cpu_workflow)
    for func_name in ('pd_load_names', 'cpu_load_acquisition_csv',
                      'cpu_load_performance_csv', 'create_ever_features',
                      'create_delinq_features', 'join_ever_delinq_features',
                      'create_joined_df', 'create_12_mon_features',
                      'combine_joined_12_mon', 'final_performance_delinquency',
                      'join_perf_acq_gdfs', 'last_mile_cleaning',
                      '_create_month_features',
                      'null_workaround'):
        globals()[func_name] = hpat.jit(debug=True)(globals()[func_name])
    _train_xgb = train_xgb
    def train_xgb(pd_df):
        for column in ('servicer', 'mod_flag', 'zero_balance_code',
                       'repurchase_make_whole_proceeds_flag',
                       'servicing_activity_indicator', 'orig_channel',
                       'first_home_buyer', 'loan_purpose', 'property_type',
                       'occupancy_status', 'property_state', 'product_type',
                       'relocation_mortgage_indicator'):
            pd_df[column] = pd_df[column].astype('category').cat.codes
        return _train_xgb(pd_df)

ML_FWS = {
    'xgb': train_xgb,
    'daal': train_daal,
    'hpat-xgb': train_xgb,
}

def _run_workflow(quarter, limit_rows):
    perf_format_path = perf_data_path + "/Performance_%sQ%s.txt"
    year = 2000 + quarter // 4
    file = perf_format_path % (str(year), str(quarter % 4))
    return run_cpu_workflow(year=year, quarter=(quarter % 4), perf_file=file, limit_rows=limit_rows)


def main():
    # end_year = 2016 # end_year is inclusive
    # part_count = 16 # the number of data files to train against
    # gpu_time = 0
    try:
        ml_func = ML_FWS[ml_fw]
    except KeyError:
        sys.exit('Unsupported ML framework, known are: %s' % ', '.join(ML_FWS))

    pd_dfs = []

    # warmup
    time_warmup = time.time()
    _run_workflow(1, True)
    print('warmup time: ', time.time() - time_warmup)

    time_ETL = time.time()
    for quarter in range(1, count_quarter_processing + 1):
        pd_dfs.append(_run_workflow(quarter, False))
    time_ETL_end = time.time()
    print("ETL time: ", time_ETL_end - time_ETL)

    ##########################################################################
    pd_df = pd_dfs[0]
    ml_func(pd_df)
    time_ML_train_end = time.time()
    print("Machine learning - train: ", time_ML_train_end - time_ETL_end)



if __name__ == '__main__':
    main()
