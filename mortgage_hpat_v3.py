import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import hpat
import time

from glob import glob
import os

dt64_fill = np.dtype('datetime64[ns]').type('1970-01-01').astype('datetime64[ns]')


import daal4py
import daal4py.hpat

from numba import types

@hpat.jit#(locals={'y': hpat.float32[:]})
def train_daal(pd_df):
    cat_cols = ['servicer', 'mod_flag', 'zero_balance_code',
                   'repurchase_make_whole_proceeds_flag',
                   'servicing_activity_indicator', 'orig_channel',
                   'first_home_buyer', 'loan_purpose', 'property_type',
                   'occupancy_status', 'property_state', 'product_type',
                   'relocation_mortgage_indicator']
    pd_df.drop(cat_cols, axis=1, inplace=True)

    y1 = pd_df["delinquency_12"].astype(np.float32).values.reshape(len(pd_df), 1)
    x1 = pd_df.drop(["delinquency_12"], axis=1).values.astype(np.float32)
#    y = y1#np.array(y1).reshape(len(pd_df), 1)
#    x = np.ascontiguousarray(pd_df.drop(["delinquency_12"], axis=1).astype(np.float32))
#    y = np.ascontiguousarray(pd_df["delinquency_12"].astype(np.float32)).reshape(len(pd_df), 1)
#    x = np.ascontiguousarray(pd_df.drop(["delinquency_12"], axis=1).astype(np.float32))

    train_algo = daal4py.gbt_regression_training(
		fptype=                       'float',
		maxIterations=                100,
		maxTreeDepth=                 8,
		minSplitLoss=                 0.1,
		shrinkage=                    0.1,
		observationsPerTreeFraction=  1,
		lambda_=                      1,
		minObservationsInLeafNode=    1,
		maxBins=                      256,
		featuresPerNode=              0,
		minBinSize=                   5,
		memorySavingMode=             False,
        )
    train_result = train_algo.compute(x1, y1)
    return train_result

@hpat.jit
def do_stuff(year, quarter, perf_file):
    pd_df = morg_func(year, quarter, perf_file)
    train_start = time.time()
    res = train_daal(pd_df)
    print('training time:', (time.time() - train_start))
    return res

@hpat.jit(locals={'final_gdf:return': 'distributed'}, cache=True)
def morg_func(year, quarter, perf_file):
    t1 = time.time()
    # read names file
    names_file = "./names.csv"
    names_df = pd.read_csv(names_file, delimiter='|',
        names=['seller_name', 'new'], dtype={'seller_name':str, 'new':str})

    # read acquisition file
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
        "property_state": CategoricalDtype(
            ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
            'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
            'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
            'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI',
            'VT', 'WA', 'WI', 'WV', 'WY']),
        "zip": np.int64,
        "mortgage_insurance_percent": np.float64,
        "product_type": CategoricalDtype(['FRM']),
        "coborrow_credit_score": np.float64,
        "mortgage_insurance_type": np.float64,
        "relocation_mortgage_indicator": CategoricalDtype(['N', 'Y']),
    }
    acq_file = ('./acq/Acquisition_'
                + str(year) + 'Q' + str(quarter) + '.txt')
    #acq_file = 'mortgage/acq/acq_test.txt'
    acq_df = pd.read_csv(acq_file, names=cols, delimiter='|', dtype=dtypes, parse_dates=[6,7], index_col=False)

    # read performance file
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
        "monthly_reporting_period": str,
        "servicer": str,
        "interest_rate": np.float64,
        "current_actual_upb": np.float64,
        "loan_age": np.float64,
        "remaining_months_to_legal_maturity": np.float64,
        "adj_remaining_months_to_maturity": np.float64,
        "maturity_date": str,
        "msa": np.float64,
        "current_loan_delinquency_status": np.int32,
        "mod_flag": CategoricalDtype(['N', 'Y']),
        "zero_balance_code": CategoricalDtype(['01', '02', '06', '09', '03', '15', '16']),
        "zero_balance_effective_date": str,
        "last_paid_installment_date": str,
        "foreclosed_after": str,
        "disposition_date": str,
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
    pdf = pd.read_csv(perf_file, names=cols, delimiter='|', dtype=dtypes,
                        parse_dates=[1,8,13,14,15,16], index_col=False)
    print("read time", time.time()-t1)
    t1 = time.time()

    acq_df = acq_df.merge(names_df, how='left', on=['seller_name'])
    acq_df.drop(columns=['seller_name'], inplace=True)
    acq_df['seller_name'] = acq_df['new']
    acq_df.drop(columns=['new'], inplace=True)

    # create ever features
    everdf = pdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id', as_index=False).max()
    everdf['ever_30'] = (everdf['current_loan_delinquency_status'] >= 1).astype(np.int8)
    everdf['ever_90'] = (everdf['current_loan_delinquency_status'] >= 3).astype(np.int8)
    everdf['ever_180'] = (everdf['current_loan_delinquency_status'] >= 6).astype(np.int8)
    everdf.drop(columns=['current_loan_delinquency_status'], inplace=True)

    # create delinq features
    delinq_df = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    delinq_30 = delinq_df[delinq_df['current_loan_delinquency_status'] >= 1][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    delinq_30.drop(columns=['monthly_reporting_period'], inplace=True)
    delinq_90 = delinq_df[delinq_df['current_loan_delinquency_status'] >= 3][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    delinq_90.drop(columns=['monthly_reporting_period'], inplace=True)
    delinq_180 = delinq_df[delinq_df['current_loan_delinquency_status'] >= 6][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    delinq_180.drop(columns=['monthly_reporting_period'], inplace=True)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'])
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(dt64_fill)
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'])
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(dt64_fill)

    # join ever delinq features
    everdf = everdf.merge(delinq_merge, on=['loan_id'], how='left')
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(dt64_fill)
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(dt64_fill)
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(dt64_fill)

    # create joined df
    test = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    test['timestamp'] = test['monthly_reporting_period']
    test.drop(columns=['monthly_reporting_period'], inplace=True)
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test.drop(columns=['current_loan_delinquency_status'], inplace=True)
    test['upb_12'] = test['current_actual_upb']
    test.drop(columns=['current_actual_upb'], inplace=True)
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)

    joined_df = test.merge(everdf, how='left', on=['loan_id'])
    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(dt64_fill)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(dt64_fill)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(dt64_fill)

    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype(np.int32)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype(np.int32)

    # create_12_mon_features
    tmpdf_1 = get_tmp_df(joined_df, 1)
    tmpdf_2 = get_tmp_df(joined_df, 2)
    tmpdf_3 = get_tmp_df(joined_df, 3)
    tmpdf_4 = get_tmp_df(joined_df, 4)
    tmpdf_5 = get_tmp_df(joined_df, 5)
    tmpdf_6 = get_tmp_df(joined_df, 6)
    tmpdf_7 = get_tmp_df(joined_df, 7)
    tmpdf_8 = get_tmp_df(joined_df, 8)
    tmpdf_9 = get_tmp_df(joined_df, 9)
    tmpdf_10 = get_tmp_df(joined_df, 10)
    tmpdf_11 = get_tmp_df(joined_df, 11)
    tmpdf_12 = get_tmp_df(joined_df, 12)
    testdf = pd.concat([tmpdf_1, tmpdf_2, tmpdf_3, tmpdf_4, tmpdf_5, tmpdf_6,
        tmpdf_7, tmpdf_8, tmpdf_9, tmpdf_10, tmpdf_11, tmpdf_12])

    # combine_joined_12_mon
    joined_df.drop(columns=['delinquency_12', 'upb_12'], inplace=True)
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype(np.int32)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype(np.int8)
    joined_df = joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])

    # final_performance_delinquency
    #merged = null_workaround(pdf)
    merged = pdf
    #joined_df = null_workaround(joined_df)
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype(np.int8)
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype(np.int32)
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])
    perf_df = merged.drop(columns=['timestamp_year', 'timestamp_month'])

    # #perf_df = null_workaround(perf_df)
    # #acq_df = null_workaround(acq_df)
    final_gdf = perf_df.merge(acq_df, how='left', on=['loan_id'])

    # final_gdf = last_mile_cleaning(final_gdf)
    # for col, dtype in final_gdf.dtypes.iteritems():
    #     if str(dtype)=='category':
    #         final_gdf[col] = final_gdf[col].cat.codes
    drop_list = [
        'loan_id', 'orig_date', 'first_pay_date', 'seller_name',
        'monthly_reporting_period', 'last_paid_installment_date', 'maturity_date', 'ever_30', 'ever_90', 'ever_180',
        'delinquency_30', 'delinquency_90', 'delinquency_180', 'upb_12',
        'zero_balance_effective_date','foreclosed_after', 'disposition_date','timestamp'
    ]
    final_gdf.drop(drop_list, axis=1, inplace=True)
    final_gdf['delinquency_12'] = final_gdf['delinquency_12'] > 0
    final_gdf['delinquency_12'] = final_gdf['delinquency_12'].fillna(False).astype(np.int32)

#    cat_cols = ['servicer', 'mod_flag', 'zero_balance_code',
#                   'repurchase_make_whole_proceeds_flag',
#                   'servicing_activity_indicator', 'orig_channel',
#                   'first_home_buyer', 'loan_purpose', 'property_type',
#                   'occupancy_status', 'property_state', 'product_type',
#                   'relocation_mortgage_indicator']
#    final_gdf.drop(cat_cols, axis=1, inplace=True)
#    for foo in ['mod_flag']:
#        final_gdf[foo] = hpat.hiframes.pd_categorical_ext.cat_array_to_int(final_gdf[foo])
#    final_gdf['mod_flag'] = hpat.hiframes.pd_categorical_ext.cat_array_to_int(final_gdf['mod_flag'])
    #final_gdf['mod_flag'] = final_gdf['mod_flag'].astype(np.int8)#cat.codes
    #final_gdf['servicer'] = final_gdf['servicer'].astype().cat.codes
    
#    for column in ('servicer', 'mod_flag', 'zero_balance_code',
#                   'repurchase_make_whole_proceeds_flag',
#                   'servicing_activity_indicator', 'orig_channel',
#                   'first_home_buyer', 'loan_purpose', 'property_type',
#                   'occupancy_status', 'property_state', 'product_type',
#                   'relocation_mortgage_indicator'):
#        pdc = hpat.hiframes.pd_categorical_ext.cat_array_to_int(final_gdf[column])
#        final_gdf[column] = pdc
        #final_gdf[column] = hpat.hiframes.pd_categorical_ext.cat_array_to_int(final_gdf[column].astype('category'))

    t2 = time.time()
    print("exec time", t2-t1)
    return final_gdf


@hpat.jit(cache=True)
def get_tmp_df(joined_df, y):
    n_months = 12
    tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
    tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
    tmpdf['josh_mody_n'] = np.floor((tmpdf['josh_months'].astype(np.float64) - 24000 - y) / 12)
    #tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
    tmpdf_d = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False)['delinquency_12'].max()
    tmpdf_m = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False)['upb_12'].min()
    tmpdf_d['upb_12'] = tmpdf_m['upb_12']
    tmpdf = tmpdf_d
    tmpdf['delinquency_12'] = (tmpdf['delinquency_12']>3).astype(np.int32)
    tmpdf['delinquency_12'] +=(tmpdf['upb_12']==0).astype(np.int32)
    tmpdf['timestamp_year'] = (((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).astype(np.int32)
    # n = len(tmpdf['timestamp_year'])
    # tmpdf['timestamp_month'] = np.full(n, y, np.int8)
    tmpdf['timestamp_month'] = np.full_like(tmpdf['timestamp_year'].values, y, np.int8)
    tmpdf.drop(columns=['josh_mody_n'], inplace=True)
    return tmpdf




acq_data_path = "./acq"
perf_data_path = "./perf"
col_names_path = "./names.csv"
start_year = 2000
end_year = 2001 # end_year is inclusive
count_limit = 2

#@hpat.jit(locals={'final_gdf:return': 'distributed'})
def main():
    quarter = 1
    year = start_year
    count = 0
    while year <= end_year:
        for file in glob(os.path.join(perf_data_path + "/Performance_" + str(year) + "Q" + str(quarter) + "*")):
            #gpu_dfs.append(process_quarter_gpu(year=year, quarter=quarter, perf_file=file))
            print(year, quarter, file)
            #df = morg_func(year=year, quarter=quarter, perf_file=file)
            train_res = do_stuff(year, quarter, file)
            count += 1
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    print("count", count)


if __name__ == '__main__':
    perf_file = "./perf/Performance_2000Q1.txt"
    #df = morg_func(2000, 1, perf_file)
    df = do_stuff(2000, 1, perf_file)

#main()






