import hashlib
import os
import re
from io import BytesIO
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd

from cache import cache
from common_definitions import links_ordered

TARGET_LINK = 74
UP_LINK = links_ordered[links_ordered.index(TARGET_LINK) - 1]
DOWN_LINK = links_ordered[links_ordered.index(TARGET_LINK) + 1]
INDEX_COLUMNS = ['simrun', 'link_id', 't']

# TODO: tests


@cache
def get_vectors_from_multiple_att_input_files(num_lags, earliest_lag_minutes_back, att_input_paths):
    return pd.concat(ThreadPool().imap_unordered(
        lambda path: __get_vectors(num_lags, earliest_lag_minutes_back, path).assign(
            simrun=lambda df: (hashlib.sha256(path).hexdigest() + df.simrun.astype(str)),
            demand=re.search(r'demand_(high|medium|low)', path).groups(0)[0],
            basename=os.path.basename(path)),
        att_input_paths))


def __get_speeds_from_att_which_is_already_1min_aggregated(input_file_path):
    with open(input_file_path) as f_in:
        lines_only_headers_and_simulation_data_vectors = ''.join(map(lambda line: line.replace(',', '.'), filter(
            lambda line: re.match(r'^[0-9]|[$]LINKEVALSEGMENTEVALUATION[:]SIMRUN', line),
            f_in)))
        return pd.read_csv(BytesIO(lines_only_headers_and_simulation_data_vectors), sep=';')\
            .rename(columns=lambda col_name: col_name
                    .lower()
                    .replace('(all)', '')
                    .replace('$linkevalsegmentevaluation:', ''))\
            [lambda df: df.timeint.str.match(r'\d+[-]\d+')] \
            .assign(link_id=lambda df: df.linkevalsegment.apply(lambda val: val.split('-')[0]).astype(int),
                    t=lambda df: df.timeint.apply(lambda val: int(val.split('-')[0])))\
            [lambda df: df.link_id.isin([TARGET_LINK, UP_LINK, DOWN_LINK])]\
            .set_index(INDEX_COLUMNS)\
            [['speed']]


def __get_df_before(df, minutes_before):
    return df\
        .reset_index()\
        .assign(t=lambda df: df.t + 60 * minutes_before)\
        .set_index(INDEX_COLUMNS)\
        .rename(columns={'speed': 'speed_before_%dmin' % minutes_before})


@cache
def __get_vectors(num_lags, earliest_lag_minutes_back, simulation_att_path):
    speeds_1min_aggregated = __get_speeds_from_att_which_is_already_1min_aggregated(simulation_att_path)
    df_with_past = speeds_1min_aggregated.join(map(lambda i: __get_df_before(speeds_1min_aggregated, i + earliest_lag_minutes_back), range(num_lags)),
                                           how='inner')\
        .reset_index(level='link_id')
    df_target_link = df_with_past[lambda df: df.link_id == TARGET_LINK].drop('link_id', axis=1)
    df_uplink = df_with_past[lambda df: df.link_id == UP_LINK]\
        .drop(['speed', 'link_id'], axis=1) \
        .rename(columns=lambda column_name: 'uplink_' + column_name)
    df_downlink = df_with_past[lambda df: df.link_id == DOWN_LINK] \
        .drop(['speed', 'link_id'], axis=1) \
        .rename(columns=lambda column_name: 'downlink_' + column_name)
    df_vectors = df_target_link\
        .join([df_uplink, df_downlink], how='inner')\
        .reset_index()
    return df_vectors.dropna()


# def combine_normal_atts(att_input_paths):
#     return pd.concat(__get_df_from_att_which_is_already_1min_aggregated(path, 1).reset_index().assign(
#         simrun=lambda df: (hashlib.sha256(path).hexdigest() + df.simrun.astype(str)),
#         demand=re.search(r'demand_(high|medium|low)', path).groups(0)[0])
#                      for path in att_input_paths)


# def combine_incident_atts(att_input_paths):
#     return pd.concat(__get_df_from_att_which_is_already_1min_aggregated(path, 1).reset_index().assign(
#         simrun=lambda df: (hashlib.sha256(path).hexdigest() + df.simrun.astype(str)),
#         demand=re.search(r'demand_(high|medium|low)', path).groups(0)[0],
#         block_location=re.search(r'location_(start|middle|end)', path).groups(0)[0],
#         duration_minutes=re.search(r'durationmin_(\d+)', path).groups(0)[0],
#         lane_block_1=re.search(r'laneblock1_(TopLane|MiddleLane|BottomLane|None)', path).groups(0)[0],
#         lane_block_2=re.search(r'laneblock2_(TopLane|MiddleLane|BottomLane|None)', path).groups(0)[0])
#                      .assign(num_lanes_blocked=lambda df: (df.lane_block_1 != 'None').astype(int) + (df.lane_block_2 != 'None').astype(int))
#                      for path in att_input_paths)


# from datetime import date, datetime, time, timedelta
#
# SIMULATION_START_TIME = time(hour=6, minute=45, second=0)
#
# def __convert_to_time_of_day(sec):
#     return (datetime.combine(date.today(), SIMULATION_START_TIME) + timedelta(seconds=int(sec)))
