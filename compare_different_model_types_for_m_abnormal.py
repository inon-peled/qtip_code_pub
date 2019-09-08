import os
from glob import glob

import pandas as pd


def get_df(are_blocked_lanes_known, stat_name):
    known_str = 'known' if are_blocked_lanes_known else 'unknown'
    def get_one_df(path):
        return pd.read_csv(path)[lambda df: df.piece_index == 0].set_index([
            'demand_level', 'blocked_lane_1', 'blocked_lane_2', 'location_on_link'])\
            [stat_name]\
            .rename(os.path.dirname(path).split(os.sep)[0])
    paths = glob(os.path.join('output_*_5step*', '2_piece_output_lags_2_future_5',
                              '*piecewise_abnormal_%s*.csv' % known_str))
    return pd.concat(list(map(get_one_df, paths)), axis='columns', join='inner')\
        .mean()\
        .rename(stat_name + '_' + known_str)\
        .sort_values()


def to_latex_table():
    return pd.concat([get_df(True, 'rmse'), get_df(False, 'rmse')], axis=1) \
            .to_latex(float_format='%.3f')
    # return pd.concat([get_df(True, 'r2'), get_df(False, 'r2'), get_df(True, 'rmsne'), get_df(False, 'rmsne')], axis=1)\
    #     .to_latex(float_format='%.3f')
    # return df.to_latex(float_format='%.1f')


if __name__ == '__main__':
    print(to_latex_table())
