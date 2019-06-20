import sys
import os
import time

import hpat
import daal4py
import daal4py.hpat

import mortgage_pandas as mp
import mortgage_hpat_v3 as mh

@hpat.jit(distributed={'pd_df'})
def clean_categories(pd_df):
    for column in ('servicer', 'mod_flag', 'zero_balance_code',
                   'repurchase_make_whole_proceeds_flag',
                   'servicing_activity_indicator', 'orig_channel',
                   'first_home_buyer', 'loan_purpose', 'property_type',
                   'occupancy_status', 'property_state', 'product_type',
                   'relocation_mortgage_indicator'):
        pd_df[column] = pd_df[column].astype('category').cat.codes
    return pd_df


@hpat.jit
def train_daal(pd_df):
    y = np.ascontiguousarray(pd_df["delinquency_12"], dtype=np.float32).reshape(len(pd_df), 1)
    x = np.ascontiguousarray(pd_df.drop(["delinquency_12"], axis=1), dtype=np.float32)

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
    train_result = train_algo.compute(x, y)
    return train_result


def main():
    #try:
    #    ml_func = hpat.jit(distributed={'pd_df'})(mp.ML_FWS[mp.ml_fw])
    #except KeyError:
    #    sys.exit('Unsupported ML framework, known are: %s' % ', '.join(mp.ML_FWS))
    ml_func = train_daal

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
    for p in pd_dfs:
        print('%d items (pid=%d)' % (len(p), os.getpid()))

    ##########################################################################
    pd_df = pd_dfs[0]#clean_categories(pd_dfs[0])
    #print('categories clean: %.3f sec' % (time.time() - time_ETL_end))
    #time_ETL_end = time.time()
    ml_func(pd_df)
    time_ML_train_end = time.time()
    print("Machine learning - train: ", time_ML_train_end - time_ETL_end)

if __name__ == '__main__':
    main()

