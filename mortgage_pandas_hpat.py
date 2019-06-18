import sys
import os
import time

import mortgage_pandas as mp
import mortgage_hpat_v3 as mh

def clean_categories(pd_df):
    for column in ('servicer', 'mod_flag', 'zero_balance_code',
                   'repurchase_make_whole_proceeds_flag',
                   'servicing_activity_indicator', 'orig_channel',
                   'first_home_buyer', 'loan_purpose', 'property_type',
                   'occupancy_status', 'property_state', 'product_type',
                   'relocation_mortgage_indicator'):
        pd_df[column] = pd_df[column].astype('category').cat.codes
    return pd_df

def main():
    try:
        ml_func = mp.ML_FWS[mp.ml_fw]
    except KeyError:
        sys.exit('Unsupported ML framework, known are: %s' % ', '.join(mp.ML_FWS))

    pd_dfs = []

    time_ETL = time.time()
    os.chdir(mp.mortgage_path)
    for quarter in range(1, mp.count_quarter_processing + 1):
        perf_format_path = "./perf/Performance_%sQ%s.txt"
        year = 2000 + quarter // 4
        file = perf_format_path % (str(year), str(quarter % 4))
        pd_dfs.append(mh.morg_func(year=year, quarter=quarter, perf_file=file))
    time_ETL_end = time.time()
    print("ETL time: ", time_ETL_end - time_ETL)

    ##########################################################################
    pd_df = clean_categories(pd_dfs[0])
    ml_func(pd_df)
    time_ML_train_end = time.time()
    print("Machine learning - train: ", time_ML_train_end - time_ETL_end)

if __name__ == '__main__':
    main()

