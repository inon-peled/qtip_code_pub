import glob
import os

import pandas as pd


def compare_normal_abnormal(input_output_dir, known_lanes):
    index_col = ['demand_level', 'blocked_lane_1', 'blocked_lane_2', 'location_on_link', 'piece_index']
    normal_df = pd.read_csv(
        os.path.join(input_output_dir, 'stats_normal_on_abnormal_%s.csv' % ('known_lanes' if known_lanes else 'unknown_lanes')),
        index_col=index_col)
    abnormal_df = pd.read_csv(
        os.path.join(input_output_dir, 'stats_piecewise_abnormal_%s.csv' % ('known_lanes' if known_lanes else 'unknown_lanes')),
        index_col=index_col)
    joined_df = normal_df.join(abnormal_df, lsuffix='_normal', rsuffix='_abnormal')
    comparison_df = pd.DataFrame({
        'improvement_mae_absolute': joined_df.mae_normal - joined_df.mae_abnormal,
        'improvement_mae_relative': (joined_df.mae_normal - joined_df.mae_abnormal) / joined_df.mae_normal,
        'improvement_rmse_absolute': joined_df.rmse_normal - joined_df.rmse_abnormal,
        'improvement_rmse_relative': (joined_df.rmse_normal - joined_df.rmse_abnormal) / joined_df.rmse_normal
    })
    comparison_df.to_csv(
        os.path.join(input_output_dir, 'comparison_normal_abnormal_%s.csv' % ('known_lanes' if known_lanes else 'unknown_lanes')))
    return comparison_df


def compare_normal_abnormal_just_rmse_known_and_unknown_lanes(input_output_dir):
    comparison_df_known = compare_normal_abnormal(input_output_dir, True)
    comparison_df_unknown = compare_normal_abnormal(input_output_dir, False)

    def rename_lanes(lane):
        return {'B': 'Bottom', 'N': 'None', 'M': 'Center', 'T': 'Top'}[lane[0]]

    df_known_and_unknown_only_rsme = pd.merge(
        suffixes=('_unknown', '_known'),
        left=comparison_df_unknown[['improvement_rmse_relative']],
        right=comparison_df_known[['improvement_rmse_relative']],
        how='inner', left_index=True, right_index=True, )\
        .rename(columns={'improvement_rmse_relative_known': 'improvement_known', 'improvement_rmse_relative_unknown': 'improvement_unknown'})\
        .reset_index()\
        [['demand_level', 'location_on_link', 'blocked_lane_1', 'blocked_lane_2', 'improvement_known', 'improvement_unknown', 'piece_index']]\
        .assign(blocked_lane_1=lambda df: df.blocked_lane_1.apply(rename_lanes),
                blocked_lane_2=lambda df: df.blocked_lane_2.apply(rename_lanes))
    for piece_index in (0,):
        # print '#'* 40, input_output_dir, 'PIECE %d' % piece_index, '#' * 40
        values_for_excel = []
        df_piece = df_known_and_unknown_only_rsme[lambda df: df.piece_index == piece_index].drop('piece_index', axis=1)
        for location in ['start', 'middle', 'end']:
            for lane1 in ['Bottom', 'Center', 'Top']:
                for lane2 in ['Bottom', 'Center', 'Top', 'None']:
                    for demand in ['high', 'medium', 'low']:
                        values = df_piece[lambda df: (df.location_on_link == location) &
                                                     (df.blocked_lane_1 == lane1) &
                                                     (df.blocked_lane_2 == lane2) &
                                                     (df.demand_level == demand)]
                        if not values.empty:
                            values_for_excel.extend([values.improvement_unknown.values[0], values.improvement_known.values[0]])
        output_file_name = os.path.join(input_output_dir, 'comparison_normal_abnormal_piece_%d.csv' % piece_index)
        with open(output_file_name, 'w') as output_f:
            for i in range(0, len(values_for_excel), 6):
                output_f.write(','.join(str(v) for v in values_for_excel[i:i + 6]) + '\n')
        print('Wrote comparison values for Excel into %s' % output_file_name)


if __name__ == '__main__':
    for input_output_dir in glob.glob(os.path.join('output_LR_5step', '*_piece_output_*')):
        compare_normal_abnormal_just_rmse_known_and_unknown_lanes(input_output_dir)
