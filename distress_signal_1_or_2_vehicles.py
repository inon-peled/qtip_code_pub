import logging

# from DeepNNRegressor import DeepNNRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import glob
import os
import re
from collections import defaultdict
from random import Random

from common_definitions import DEMAND_NAMES, LINK_LOCATION_NAMES, get_combinations, INCIDENT_START_SECOND, \
    INCIDENT_END_SECOND
from regressor import Regressor

TOP_DIR_FOR_ALL_SCENARIOS = 'sim_5sec'
INCIDENTS_SUBDIR = os.path.join(TOP_DIR_FOR_ALL_SCENARIOS, 'incident_scenarios')
ARBITRARY_SEED = 777
NUM_SIMULATIONS_IN_TRAIN_SET_IF_BLOCKED_LANES_ARE_UNKNOWN = 100


def get_fresh_randomizer(seed):
    return Random(seed)


class DistressSignal1Or2Vehicles(object):
    def __init__(self, logger, regressor, incidents_directory, seed, num_lags, earliest_lag_minutes_back):
        self.logger = logger
        self.earliest_lag_minutes_back = earliest_lag_minutes_back
        self.num_lags = num_lags
        self.regressor = regressor
        self.incidents_directory = incidents_directory
        self.seed = seed

    def __get_train_and_test_sets_lanes_known_in_train(
            self, demand_level, blocked_lane_1, blocked_lane_2, location_on_link, num_test_per_exact_parameters):
        exactly_matching_directories = glob.glob(os.path.join(
            self.incidents_directory,
            INCIDENTS_SUBDIR,
            'incident_link74_location_%s_durationmin_30_demand_%s_laneblock1_%s_laneblock2_%s_simrun_*' %
            (location_on_link, demand_level, blocked_lane_1, blocked_lane_2)))
        distress_signal_directories = glob.glob(os.path.join(
            self.incidents_directory,
            INCIDENTS_SUBDIR,
            'incident_link74_location_%s_durationmin_30_demand_*_laneblock1_%s_laneblock2_%s_simrun_*' %
            (location_on_link, blocked_lane_1, blocked_lane_2)))
        test_set = set(get_fresh_randomizer(self.seed).sample(exactly_matching_directories, num_test_per_exact_parameters))
        train_set = get_fresh_randomizer(self.seed).sample(set(distress_signal_directories) - test_set,
                                                           NUM_SIMULATIONS_IN_TRAIN_SET_IF_BLOCKED_LANES_ARE_UNKNOWN)
        return test_set, train_set

    def __get_train_and_test_sets_lanes_unknown_in_train(
            self, demand_level, blocked_lane_1, blocked_lane_2, location_on_link, num_test_per_exact_parameters,
            num_train_per_distress_signal_parameters):
        exactly_matching_directories = glob.glob(os.path.join(
            self.incidents_directory,
            INCIDENTS_SUBDIR,
            'incident_link74_location_%s_durationmin_30_demand_%s_laneblock1_%s_laneblock2_%s_simrun_*' %
            (location_on_link, demand_level, blocked_lane_1, blocked_lane_2)))
        distress_signal_directories = glob.glob(os.path.join(
            self.incidents_directory,
            INCIDENTS_SUBDIR,
            'incident_link74_location_%s_durationmin_30_demand_*_laneblock1_*_laneblock2_%s_simrun_*' %
            (location_on_link, '*Lane*' if blocked_lane_2 not in (None, 'None') else 'None')))
        test_set = set(get_fresh_randomizer(self.seed).sample(exactly_matching_directories, num_test_per_exact_parameters))
        d = defaultdict(list)
        for t in set(distress_signal_directories) - test_set:
            d[re.search(
                r'incident_link74_location_([^_]+)_durationmin_30_demand_([^_]+)_laneblock1_([^_]+)_laneblock2_([^_]+)',
                t).groups()].append(t)
        train_set = reduce(list.__add__,
                           map(lambda lst: get_fresh_randomizer(self.seed).sample(lst, num_train_per_distress_signal_parameters),
                               d.values()))
        train_set_sampled = get_fresh_randomizer(self.seed).sample(train_set,
                                                                   NUM_SIMULATIONS_IN_TRAIN_SET_IF_BLOCKED_LANES_ARE_UNKNOWN)
        return test_set, train_set_sampled

    def stats_one_ground_truth_scenario(self, random_sample_of_normal_atts, demand_level, blocked_lane_1, blocked_lane_2, location_on_link,
                                        num_test_per_exact_parameters, num_train_per_distress_signal_parameters,
                                        stats_abnormal_fh, stats_normal_fh, lanes_known_in_train):
        self.logger.info('stats_one_ground_truth_scenario with %s, %s, %s, %s, %s' % (lanes_known_in_train, demand_level, blocked_lane_1, blocked_lane_2, location_on_link))

        def write_stats_abnormal(test_set):
            self.logger.info('write_stats_abnormal')
            stats_abnormal = self.regressor.train_and_test_piecewise_m_abnormal(
                append_results_att(train_set), append_results_att(test_set),
                'piecewise_' + file_name_suffix,
                title='%s %d-Piece $M_{abnormal}$, %s Blocked Lanes\nDemand=%s, Location=%s, BlockedLane1=%s, BlockedLane2=%s' % (
                    self.regressor.regression_cls.__name__,
                    len(self.regressor.t_ranges_for_piecewise_modeling),
                    ('Known' if lanes_known_in_train else 'Unknown'), demand_level.capitalize(),
                    location_names(location_on_link),
                    link_names(blocked_lane_1), link_names(blocked_lane_2)))
            for st_abnormal in stats_abnormal:
                stats_abnormal_fh.write(','.join(str(e) for e in [
                    demand_level, blocked_lane_1, blocked_lane_2, location_on_link] + st_abnormal) + '\n')

        def write_stats_normal_on_abnormal(test_set):
            self.logger.info('write_stats_normal_on_abnormal')
            all_normal_atts = glob.glob(os.path.join(
                self.incidents_directory,
                TOP_DIR_FOR_ALL_SCENARIOS,
                'normal_scenario_demand_*',
                '*',
                '*Link Segment Results_001.att'))
            sample_normal_atts = all_normal_atts if random_sample_of_normal_atts is None else \
                get_fresh_randomizer(self.seed).sample(all_normal_atts, int(len(all_normal_atts) * random_sample_of_normal_atts))
            abnormal_atts = [os.path.join(d, 'results.att') for d in test_set]
            stats_normal_on_abnormal = self.regressor.model_degradation_on_time_pieces(
                '_' + file_name_suffix, sample_normal_atts, abnormal_atts)
            for st_normal in stats_normal_on_abnormal:
                stats_normal_fh.write(','.join(
                    str(e) for e in ((demand_level, blocked_lane_1, blocked_lane_2, location_on_link) + st_normal)) + '\n')

        if lanes_known_in_train:
            test_set, train_set = self.__get_train_and_test_sets_lanes_known_in_train(
                demand_level, blocked_lane_1, blocked_lane_2, location_on_link, num_test_per_exact_parameters)
        else:
            test_set, train_set = self.__get_train_and_test_sets_lanes_unknown_in_train(
                demand_level, blocked_lane_1, blocked_lane_2, location_on_link, num_test_per_exact_parameters,
                num_train_per_distress_signal_parameters)
        print demand_level, blocked_lane_1, blocked_lane_2, location_on_link + ':', \
            len(test_set), 'samples in test set,', len(train_set), 'samples in train set'

        def append_results_att(directories_list):
            return map(lambda s: os.path.join(s, 'results.att'), directories_list)

        file_name_suffix = '%s_lanes_gt_%s_%s_%s_%s' % (
            ('known' if lanes_known_in_train else 'unknown'), demand_level, blocked_lane_1, blocked_lane_2, location_on_link)

        def location_names(block_location_on_link):
            return {'start': 'Start', 'end': 'End', 'middle': 'Center'}[block_location_on_link]

        def link_names(blocked_lane):
            return 'None' if blocked_lane is None else {'T': 'Left', 'M': 'Middle', 'B': 'Right'}[blocked_lane[0]]

        write_stats_abnormal(test_set)

        write_stats_normal_on_abnormal(test_set)


def compute_stats(logger, seed, random_sample_of_normal_atts, lags, first_lag_minutes_back, regressor):
    def create_dir_and_open_for_writing(path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        return open(path, 'w')

    for known_lanes in (False, True):
        if not os.path.exists(regressor.get_output_dir()):
            os.makedirs((regressor.get_output_dir()))
        with create_dir_and_open_for_writing(os.path.join((regressor.get_output_dir()),
                         ('stats_piecewise_abnormal_%s_lanes.csv' % ('known' if known_lanes else 'unknown')))) as stats_abnormal_file_handle, \
                create_dir_and_open_for_writing(os.path.join((regressor.get_output_dir()),
                         ('stats_normal_on_abnormal_%s_lanes.csv' % ('known' if known_lanes else 'unknown')))) as stats_normal_file_handle:
            header = 'demand_level,blocked_lane_1,blocked_lane_2,location_on_link,piece_index,r2,mae,rmse,rmsne\n'
            stats_abnormal_file_handle.write(header)
            stats_normal_file_handle.write(header)
            parameter_combinations = get_combinations()
            get_fresh_randomizer(seed).shuffle(parameter_combinations)
            for demand, lanes, location in parameter_combinations:
                DistressSignal1Or2Vehicles(logger, regressor, 'copy_of_m_drive', ARBITRARY_SEED, lags, first_lag_minutes_back) \
                    .stats_one_ground_truth_scenario(
                    random_sample_of_normal_atts,
                    DEMAND_NAMES[demand], lanes[0], None if len(lanes) < 2 else lanes[1], LINK_LOCATION_NAMES[location],
                    5,
                    24 // len(lanes),
                    stats_abnormal_file_handle,
                    stats_normal_file_handle,
                    known_lanes
                )
                stats_abnormal_file_handle.flush()
                stats_normal_file_handle.flush()


def various_regression():
    # one_piece_t_range = [range(INCIDENT_START_SECOND, INCIDENT_END_SECOND + 1, 60)]
    two_piece_t_range = [range(INCIDENT_START_SECOND, INCIDENT_START_SECOND + 6 * 60, 60),
                         range(INCIDENT_START_SECOND + 6 * 60, INCIDENT_END_SECOND + 1, 60)]
    num_lags = 2
    minutes_back_to_first_lag = 5

    for pieces_of_time_range in [two_piece_t_range]: # one_piece_t_range
        # LR
        compute_stats(logger, ARBITRARY_SEED, random_sample_of_normal_atts=None, lags=num_lags, first_lag_minutes_back=minutes_back_to_first_lag,
                      regressor=Regressor(
                          'output_LR_5step',
                          logger,
                          pieces_of_time_range,
                          LinearRegression,
                          dict(fit_intercept=False),
                          None, ARBITRARY_SEED, num_lags, minutes_back_to_first_lag))

        # # GP
        # for scale_length in range(9, 16):
        #     compute_stats(logger, ARBITRARY_SEED, random_sample_of_normal_atts=None, lags=num_lags, first_lag_minutes_back=minutes_back_to_first_lag,
        #                   regressor=Regressor(
        #                       'output_GP_5step_s%f' % scale_length,
        #                       logger,
        #                       pieces_of_time_range,
        #                       GaussianProcessRegressor,
        #                       dict(random_state=ARBITRARY_SEED, kernel=RBF(scale_length, (scale_length, scale_length))),
        #                       None, ARBITRARY_SEED, num_lags, minutes_back_to_first_lag))

        # # DNN
        # for num_hidden_layers in (1, 2):
        #     for num_sigmoids_in_each_hidden_layer in (15,):
        #         compute_stats(logger, ARBITRARY_SEED, random_sample_of_normal_atts=None, lags=num_lags, first_lag_minutes_back=minutes_back_to_first_lag,
        #                       regressor=Regressor(
        #                         'output_DNN_5step_h%d_g%d' % (num_hidden_layers, num_sigmoids_in_each_hidden_layer),
        #                         logger,
        #                         pieces_of_time_range,
        #                         DeepNNRegressor,
        #                         dict(num_hidden_layers=num_hidden_layers,
        #                              num_sigmoids_in_each_hidden_layer=num_sigmoids_in_each_hidden_layer,
        #                              loss='mse',
        #                              optimizer='adam',
        #                              mini_batch_size=100,
        #                              num_epochs=100,
        #                              validation_split=0.1),
        #                         None, ARBITRARY_SEED, num_lags, minutes_back_to_first_lag))


if __name__ == '__main__':
    various_regression()
