import datetime
from random import Random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

from common_definitions import *
from get_vectors import get_vectors_from_multiple_att_input_files
from plotter import Plotter

START_SECOND_FOR_TRAINING_M_NORMAL = 60 * 15
END_SECOND_FOR_TRAINING_M_NORMAL = START_SECOND_FOR_TRAINING_M_NORMAL + 60 * 60

START_SECOND_FOR_ABNORMAL_TEST_DATA = INCIDENT_START_SECOND - 60 * 5
END_SECOND_FOR_ABNORMAL_TEST_DATA = INCIDENT_END_SECOND + 60 * 5

INFINITY_SPEED = 10 ** 7 - 1


# TODO: train all models on same train set
# TODO: build different models per incident parameters?


class Regressor(object):
    def __init__(self, output_subdir, logger, t_ranges_for_piecewise_modeling, regression_cls, regressor_init_kwargs,
                 scenario_results_dir, seed, num_lags, earliest_lag_minutes_back):
        self.logger = logger
        self.t_ranges_for_piecewise_modeling = t_ranges_for_piecewise_modeling
        self.regression_cls = regression_cls
        self.regressor_init_kwargs = regressor_init_kwargs
        self.output_dir = os.path.join(output_subdir,
                                       '%d_piece_output_lags_%d_future_%d' % (len(t_ranges_for_piecewise_modeling),
                                                                              num_lags, earliest_lag_minutes_back))
        self.plotter = Plotter(self.output_dir, scenario_results_dir, seed, num_lags, earliest_lag_minutes_back)
        self.seed = seed if seed is not None else int(datetime.datetime.now().strftime('%s'))
        self.earliest_lag_minutes_back = earliest_lag_minutes_back
        self.num_lags = num_lags
        self.scenario_results_dir = scenario_results_dir
        create_dir(self.output_dir)

    def get_fresh_randomizer(self):
        return Random(self.seed)

    def time_series_degradation_of_normal_model_with_and_without_other_links(self, normal_atts, abnormal_atts, output_file_name):
        t_range_for_predictions = range(START_SECOND_FOR_ABNORMAL_TEST_DATA, END_SECOND_FOR_ABNORMAL_TEST_DATA, 60)
        _, m_normal_with_u_d = self.__train_model_with_other_links(normal_atts)
        vectors_test_with_u_d = self.__get_vectors_from_multiple_att_input_files(abnormal_atts)
        _, m_normal_without_u_d = self.__train_model_without_other_links(normal_atts)
        vectors_test_without_u_d = remove_uplink_and_downlink(self.__get_vectors_from_multiple_att_input_files(abnormal_atts))
        self.plotter.plot_with_and_without_u_d_on_average_time_series_no_frills(
            vectors_test_with_u_d, [(t_range_for_predictions, m_normal_with_u_d)],
            vectors_test_without_u_d, [(t_range_for_predictions, m_normal_without_u_d)],
            'no_frills_' + output_file_name,
            title='Degradation of $M_{ordinary}$ on Average Incident Conditions')

    def time_series_degradation_of_normal_model_with_other_links(self, normal_atts, abnormal_atts, output_file_name):
        t_range_for_predictions = range(START_SECOND_FOR_ABNORMAL_TEST_DATA, END_SECOND_FOR_ABNORMAL_TEST_DATA, 60)
        vectors_train, m_normal_with_other_links = self.__train_model_with_other_links(normal_atts)
        vectors_test = self.__get_vectors_from_multiple_att_input_files(abnormal_atts)
        self.plotter.plot_on_average_time_series_no_frills(
            vectors_test, [(t_range_for_predictions, m_normal_with_other_links)], 'no_frills_' + output_file_name,
            title='$M_{ordinary}$ Degradation on Avergage Incident Conditions, with Uplink and Downlink Information')
        self.plotter.plot_on_average_time_series(
            vectors_train, vectors_test, [(t_range_for_predictions, m_normal_with_other_links)], output_file_name,
            title='$M_{ordinary}$ Degradation on Avergage Incident Conditions, with Uplink and Downlink Information')

    def time_series_degradation_of_normal_model_without_other_links(self, normal_atts, abnormal_atts, output_file_name):
        t_range_for_predictions = range(START_SECOND_FOR_ABNORMAL_TEST_DATA, END_SECOND_FOR_ABNORMAL_TEST_DATA, 60)
        vectors_train, m_normal_without_other_links = self.__train_model_without_other_links(
            normal_atts)
        vectors_test = remove_uplink_and_downlink(
            self.__get_vectors_from_multiple_att_input_files(abnormal_atts))
        self.plotter.plot_on_average_time_series_no_frills(
            vectors_test, [(t_range_for_predictions, m_normal_without_other_links)], 'no_frills_' + output_file_name,
            title='$M_{ordinary}$ Degradation on Avergage Incident Conditions, without Uplink and Downlink Information')
        self.plotter.plot_on_average_time_series(
            vectors_train, vectors_test, [(t_range_for_predictions, m_normal_without_other_links)], output_file_name,
            title='$M_{ordinary}$ Degradation on Avergage Incident Conditions, without Uplink and Downlink Information')

    def create_time_series_plots_for_distress_signal(self, normal_atts, abnormal_atts,
                                                     fraction_simulations_for_test_set):
        all_atts_abnormal_train, all_atts_abnormal_test = self.__split_atts_into_test_and_train_sets(
            abnormal_atts, fraction_simulations_for_test_set)

        def single_location_on_link(atts, relative_location_on_link):
            return filter(lambda att: (('location_%s' % relative_location_on_link) in att), atts)

        for relative_location_on_link in LINK_LOCATION_NAMES.values():
            atts_abnormal_train_single_location = single_location_on_link(all_atts_abnormal_train,
                                                                          relative_location_on_link)
            atts_abnormal_test_single_location = single_location_on_link(all_atts_abnormal_test,
                                                                         relative_location_on_link)
            self.train_and_test_piecewise_m_abnormal(
                atts_abnormal_train_single_location,
                atts_abnormal_test_single_location,
                'piecewise_m_abnormal_with_other_links_location_%s.png' % relative_location_on_link,
                title='%d-Piece $M_{abnormal}$ with Uplink and Downlink Information\nOnly Simulations with Blockage at %s of Link' %
                      (len(self.t_ranges_for_piecewise_modeling), self.regression_cls.__name__, relative_location_on_link.upper())
            )
            for use_feature_t_accident in (True, False):
                self.__train_and_test_m_abnormal(
                    use_feature_t_accident,
                    atts_abnormal_train_single_location,
                    atts_abnormal_test_single_location,
                    'm_abnormal_with_other_links_blockage_location_%s_t_accident_%d.png' % (
                        relative_location_on_link, use_feature_t_accident),
                    title='$M_{abnormal}$ with Uplink and Downlink Information\nOnly Simulations with Blockage at %s of Link' % relative_location_on_link.upper())
            self.time_series_degradation_of_normal_model_without_other_links(
                normal_atts,
                atts_abnormal_test_single_location,
                'time_series_m_normal_without_other_links_on_blockage_location_%s.png' %
                relative_location_on_link)

    def memoized_train_m_ordinary(self, normal_atts, memo={}):
        memo_key = tuple(sorted(normal_atts))
        if memo_key not in memo:
            self.logger.info('Training M_ordinary on %d normal_atts' % len(normal_atts))
            vectors_train = self.__get_vectors_from_multiple_att_input_files(normal_atts)
            data_train = get_vectors_for_fitting(vectors_train).as_matrix()
            measured_speed_train = vectors_train.speed.as_matrix()
            trained_model = self.regression_cls(**self.regressor_init_kwargs).fit(data_train, measured_speed_train)
            memo[memo_key] = data_train, trained_model
        return memo[memo_key]

    def model_degradation_on_time_pieces(self, file_name_suffix, normal_atts, abnormal_atts):
        self.logger.info('model_degradation_on_time_pieces')
        normal_data_train, trained_m_ordinary = self.memoized_train_m_ordinary(normal_atts)
        stats = []
        for i, t_range in enumerate(self.__get_t_ranges()):
            vectors_abnormal = self.__get_vectors_from_multiple_att_input_files(abnormal_atts)
            fit_model_normal, \
            data_abnormal, \
            data_normal, \
            r2_normal_model_applied_to_abnormal, \
            mae_normal_model_applied_to_abnormal, \
            rmse_normal_model_applied_to_abnormal, \
            rmsne_normal_model_applied_to_abnormal, \
            measured_abnormal = self.__degradation_of_pretrained_m_ordinary_on_time_piece(
                normal_data_train, trained_m_ordinary, vectors_abnormal, t_range)
            self.plotter.scatter_plot_model_degradation_on_time_pieces(
                len(normal_atts), len(abnormal_atts),
                r2_normal_model_applied_to_abnormal, mae_normal_model_applied_to_abnormal,
                rmse_normal_model_applied_to_abnormal, measured_abnormal, fit_model_normal.predict(data_abnormal),
                'model_degradation_on_time_piece_%s%s' % (i, file_name_suffix), 'model_degradation_on_time_piece_%s%s.png' % (i, file_name_suffix))
            stats.append((i, r2_normal_model_applied_to_abnormal, mae_normal_model_applied_to_abnormal,
                          rmse_normal_model_applied_to_abnormal, rmsne_normal_model_applied_to_abnormal))
        return stats

    def m_ordinary_degradation_on_test_vectors(
            self, with_other_links, normal_atts, abnormal_atts, output_file_path, title):
        vectors_normal = self.__get_vectors_from_multiple_att_input_files(normal_atts)
        vectors_abnormal = self.__get_vectors_from_multiple_att_input_files(abnormal_atts)
        msd_normal_model_applied_to_abnormal, \
        fit_model_normal, \
        data_abnormal, \
        data_normal, \
        r2_normal_model_applied_to_abnormal, \
        mae_normal_model_applied_to_abnormal, \
        rmse_normal_model_applied_to_abnormal, \
        rmsne_normal_model_applied_to_abnormal, \
        measured_abnormal = self.__model_degradation_metrics(with_other_links, vectors_normal, vectors_abnormal)
        self.plotter.hexbin_plot_model_degradation(
            msd_normal_model_applied_to_abnormal,
            fit_model_normal.predict(data_abnormal), len(data_normal), len(data_abnormal),
            r2_normal_model_applied_to_abnormal, mae_normal_model_applied_to_abnormal,
            rmse_normal_model_applied_to_abnormal, measured_abnormal,
            title, output_file_path)
        return r2_normal_model_applied_to_abnormal, mae_normal_model_applied_to_abnormal, \
               rmse_normal_model_applied_to_abnormal

    def cache_key(self):
        return self.seed, self.earliest_lag_minutes_back, self.num_lags

    def get_output_dir(self):
        return self.output_dir

    def __add_feature_minutes_since_accident_start(self, vectors):
        return vectors.assign(inverse_of_minutes_since_accident_start=lambda df: np.where(
            df.t >= INCIDENT_START_SECOND,
            60.0 / (60 + df.t - INCIDENT_START_SECOND),
            INFINITY_SPEED))

    def __get_t_ranges(self):
        return self.t_ranges_for_piecewise_modeling

    def train_and_test_piecewise_m_abnormal(
            self, atts_abnormal_train, atts_abnormal_test, output_file_name, title):
        self.logger.info('train_and_test_piecewise_m_abnormal %s' % output_file_name)

        vectors_abnormal_train = self.__add_feature_minutes_since_accident_start(
            self.__get_vectors_from_multiple_att_input_files(atts_abnormal_train))
        vectors_abnormal_test = self.__add_feature_minutes_since_accident_start(
            self.__get_vectors_from_multiple_att_input_files(atts_abnormal_test))

        def train_per_time_period(t_range_for_training):
            train_vectors_in_t_range_for_training = vectors_abnormal_train[lambda df: df.t.isin(t_range_for_training)]
            return self.regression_cls(**self.regressor_init_kwargs).fit(
                X=get_vectors_for_fitting(train_vectors_in_t_range_for_training).as_matrix(),
                y=train_vectors_in_t_range_for_training.speed.as_matrix())

        t_ranges = self.__get_t_ranges()
        models = map(train_per_time_period, t_ranges)
        return self.plotter.plot_on_average_time_series(vectors_abnormal_train, vectors_abnormal_test,
                                                  zip(t_ranges, models), output_file_name, title)

    def __split_vectors_into_test_and_train_sets(self, vectors, fraction_simulations_for_test_set):
        all_simulations = vectors.simrun.unique()
        simulations_for_test_set = self.get_fresh_randomizer().sample(all_simulations,
                                                          int(fraction_simulations_for_test_set * len(all_simulations)))
        return vectors[~vectors.simrun.isin(simulations_for_test_set)], \
               vectors[vectors.simrun.isin(simulations_for_test_set)]

    def __split_atts_into_test_and_train_sets(self, atts, fraction_simulations_for_test_set):
        atts_test = set(self.get_fresh_randomizer().sample(atts, int(fraction_simulations_for_test_set * len(atts))))
        return set(atts) - atts_test, atts_test

    def __future_delay(self):
        return 60 * (self.earliest_lag_minutes_back + self.num_lags - 1)

    def __train_and_test_m_abnormal(
            self, use_feature_t_accident, atts_abnormal_train, atts_abnormal_test, output_file_name, title):
        add_feature = self.__add_feature_minutes_since_accident_start if use_feature_t_accident else lambda x: x
        vectors_abnormal_train = add_feature(self.__get_vectors_from_multiple_att_input_files(atts_abnormal_train))
        vectors_abnormal_test = add_feature(self.__get_vectors_from_multiple_att_input_files(atts_abnormal_test))
        t_range_for_training = range(INCIDENT_START_SECOND, INCIDENT_END_SECOND, 60)
        train_vectors_in_t_range_for_training = vectors_abnormal_train[lambda df: df.t.isin(t_range_for_training)]
        m_abnormal = self.regression_cls(**self.regressor_init_kwargs).fit(
            X=get_vectors_for_fitting(train_vectors_in_t_range_for_training).as_matrix(),
            y=train_vectors_in_t_range_for_training.speed.as_matrix())
        self.plotter.plot_on_average_time_series(vectors_abnormal_train, vectors_abnormal_test,
                                                 [(t_range_for_training, m_abnormal)], output_file_name, title)

    def compare_scatter_of_normal_model_with_and_without_other_links(self, normal_atts):
        all_vectors_normal = self.__get_vectors_from_multiple_att_input_files(normal_atts) \
            [lambda df: df.t >= START_SECOND_FOR_TRAINING_M_NORMAL + 60 * (
                self.num_lags + self.earliest_lag_minutes_back - 1)] \
            [lambda df: df.t <= END_SECOND_FOR_TRAINING_M_NORMAL]
        self.__fit_model_and_plot(10, all_vectors_normal, '$M_{ordinary}$ with $U$ and $D$, Incident-free Simulations', 'm_normal_with_other_links.png')
        self.__fit_model_and_plot(10, remove_uplink_and_downlink(all_vectors_normal),
                               '$M_{ordinary}$ without $U$ and $D$, Incident-free Simulations', 'm_normal_without_other_links.png')

    def __train_model_with_other_links(self, atts):
        vectors_train = self.__get_vectors_from_multiple_att_input_files(atts) \
            [lambda df: df.t >= START_SECOND_FOR_TRAINING_M_NORMAL + 60 * (
                self.num_lags + self.earliest_lag_minutes_back - 1)] \
            [lambda df: df.t <= END_SECOND_FOR_TRAINING_M_NORMAL]
        return vectors_train, self.regression_cls(**self.regressor_init_kwargs).fit(
            X=get_vectors_for_fitting(vectors_train).as_matrix(),
            y=vectors_train.speed.as_matrix())

    def __train_model_without_other_links(self, atts):
        vectors_train = self.__get_vectors_from_multiple_att_input_files(atts) \
            [lambda df: df.t >= START_SECOND_FOR_TRAINING_M_NORMAL + 60 * (
                self.num_lags + self.earliest_lag_minutes_back - 1)] \
            [lambda df: df.t <= END_SECOND_FOR_TRAINING_M_NORMAL]
        return vectors_train, self.regression_cls(**self.regressor_init_kwargs).fit(
            X=get_vectors_for_fitting(remove_uplink_and_downlink(vectors_train)).as_matrix(),
            y=vectors_train.speed.as_matrix())

    def __get_vectors_from_multiple_att_input_files(self, atts):
        return get_vectors_from_multiple_att_input_files(self.num_lags, self.earliest_lag_minutes_back, atts)

    def __fit_model_and_plot(self, cross_validation_folds, all_vectors, title, output_file_path):
        vectors_for_fitting = get_vectors_for_fitting(all_vectors)
        X = vectors_for_fitting.as_matrix()
        y = all_vectors.speed.as_matrix()
        cv_predictions = cross_val_predict(self.regression_cls(**self.regressor_init_kwargs), X, y, cv=cross_validation_folds)
        cv_r2 = metrics.r2_score(y, cv_predictions)
        cv_msd = sum(y - cv_predictions) / len(y)
        cv_mae = metrics.mean_absolute_error(y, cv_predictions)
        cv_rmse = metrics.mean_squared_error(y, cv_predictions) ** 0.5
        self.plotter.plot_lr(
            cv_predictions, cv_r2, cv_msd, cv_mae, cv_rmse, vectors_for_fitting,
            self.regression_cls(**self.regressor_init_kwargs).fit(X, y), cross_validation_folds, all_vectors,
            title, output_file_path)
        return len(vectors_for_fitting), cv_r2, cv_mae, cv_rmse

    def __build_fixed_test_set_for_combined_model(self, fraction_simulations_for_test_set, vectors_normal,
                                                  vectors_abnormal):
        vectors_train_normal, vectors_test_normal = self.__split_vectors_into_test_and_train_sets(
            vectors_normal, fraction_simulations_for_test_set)
        vectors_train_abnormal, vectors_test_abnormal = self.__split_vectors_into_test_and_train_sets(
            vectors_abnormal, fraction_simulations_for_test_set)
        vectors_test = pd.concat([vectors_test_normal, vectors_test_abnormal])
        return vectors_train_normal, vectors_train_abnormal, vectors_test

    def __combined_model_all_experiments(self, num_repetitions_for_average_measurements,
                                         vectors_train_normal, vectors_train_abnormal, vectors_test,
                                         fraction_simulations_for_test_set):
        def measure_degradation(percent_abnormal_in_train_set):
            sample_size = percent_abnormal_in_train_set * len(vectors_train_normal) / (
                100 - percent_abnormal_in_train_set)
            sampled_vectors_train_abnormal = vectors_train_abnormal.sample(sample_size)
            vectors_train = pd.concat([vectors_train_normal, sampled_vectors_train_abnormal])
            _, _, _, _, \
            r2_trained_model_applied_to_test_set, \
            mae_trained_model_applied_to_test_set, \
            rmse_trained_model_applied_to_test_set, \
            _ = \
                self.__model_degradation_metrics(vectors_train, vectors_test)
            return r2_trained_model_applied_to_test_set, mae_trained_model_applied_to_test_set, rmse_trained_model_applied_to_test_set

        averaged = sum(pd.DataFrame({percent_abnormal_in_train_set: measure_degradation(percent_abnormal_in_train_set)
                                     for percent_abnormal_in_train_set in range(51)})
                       for _ in
                       range(num_repetitions_for_average_measurements)) / num_repetitions_for_average_measurements
        return averaged.transpose().rename(columns=dict(enumerate(['$R^2$', '$MAE (km/h)$', '$RMSE (km/h)$'])))

    def combined_model(self, atts_for_normal_conditions, atts_for_abnormal_conditions,
                       fraction_simulations_for_test_set):
        vectors_train_normal, vectors_train_abnormal, vectors_test = self.__build_fixed_test_set_for_combined_model(
            fraction_simulations_for_test_set,
            self.__get_vectors_from_multiple_att_input_files(atts_for_normal_conditions),
            self.__get_vectors_from_multiple_att_input_files(atts_for_abnormal_conditions))
        self.plotter.plot_combined_model(
            self.__combined_model_all_experiments(
                10, vectors_train_normal, vectors_train_abnormal, vectors_test, fraction_simulations_for_test_set),
            fraction_simulations_for_test_set, len(vectors_test), len(vectors_train_normal))

    def __degradation_of_pretrained_m_ordinary_on_time_piece(self, normal_data_train, trained_m_ordinary, vectors_test, t_range):
        self.logger.info('Now measuring degradation of M_ordinary')

        vectors_test_first_piece = vectors_test[vectors_test.t.isin(t_range)]
        data_test = get_vectors_for_fitting(vectors_test_first_piece).as_matrix()
        measured_speed_test = vectors_test_first_piece.speed.as_matrix()

        r2_trained_model_applied_to_test_set = metrics.r2_score(y_true=measured_speed_test,
                                                                y_pred=trained_m_ordinary.predict(data_test))
        mae_trained_model_applied_to_test_set = \
            metrics.mean_absolute_error(measured_speed_test, trained_m_ordinary.predict(data_test))
        rmse_trained_model_applied_to_test_set = \
            metrics.mean_squared_error(measured_speed_test, trained_m_ordinary.predict(data_test)) ** 0.5
        rmsne_trained_model_applied_to_test_set = \
            np.mean(np.true_divide(np.array(measured_speed_test) - np.array(trained_m_ordinary.predict(data_test)),
                                   np.array(measured_speed_test)) ** 2) ** 0.5
        return trained_m_ordinary, \
               data_test, \
               normal_data_train, \
               r2_trained_model_applied_to_test_set, \
               mae_trained_model_applied_to_test_set, \
               rmse_trained_model_applied_to_test_set, \
               rmsne_trained_model_applied_to_test_set, \
               measured_speed_test

    def __model_degradation_metrics_on_time_piece(self, vectors_train, vectors_test, t_range):
        self.logger.info('Now measuring degradation of M_ordinary')
        data_train = get_vectors_for_fitting(vectors_train).as_matrix()
        measured_speed_train = vectors_train.speed.as_matrix()
        trained_model = self.regression_cls(**self.regressor_init_kwargs).fit(data_train, measured_speed_train)

        vectors_test_first_piece = vectors_test[vectors_test.t.isin(t_range)]
        data_test = get_vectors_for_fitting(vectors_test_first_piece).as_matrix()
        measured_speed_test = vectors_test_first_piece.speed.as_matrix()

        r2_trained_model_applied_to_test_set = metrics.r2_score(y_true=measured_speed_test,
                                                                y_pred=trained_model.predict(data_test))
        mae_trained_model_applied_to_test_set = \
            metrics.mean_absolute_error(measured_speed_test, trained_model.predict(data_test))
        rmse_trained_model_applied_to_test_set = \
            metrics.mean_squared_error(measured_speed_test, trained_model.predict(data_test)) ** 0.5
        return trained_model, \
               data_test, \
               data_train, \
               r2_trained_model_applied_to_test_set, \
               mae_trained_model_applied_to_test_set, \
               rmse_trained_model_applied_to_test_set, \
               measured_speed_test

    def __model_degradation_metrics(self, with_other_links, vectors_train, vectors_test):
        other_links_func = (lambda x: x) if with_other_links else remove_uplink_and_downlink
        data_train = other_links_func(get_vectors_for_fitting(vectors_train)).as_matrix()
        measured_speed_train = vectors_train.speed.as_matrix()
        trained_model = self.regression_cls(**self.regressor_init_kwargs).fit(data_train, measured_speed_train)

        data_test = other_links_func(get_vectors_for_fitting(vectors_test)).as_matrix()
        measured_speed_test = vectors_test.speed.as_matrix()

        msd_trained_model_applied_to_test_set = np.mean(trained_model.predict(data_test) - measured_speed_test)
        r2_trained_model_applied_to_test_set = trained_model.score(data_test, measured_speed_test)
        mae_trained_model_applied_to_test_set = \
            metrics.mean_absolute_error(measured_speed_test, trained_model.predict(data_test))
        rmse_trained_model_applied_to_test_set = \
            metrics.mean_squared_error(measured_speed_test, trained_model.predict(data_test)) ** 0.5
        rmsne_trained_model_applied_to_test_set = \
            np.mean(np.true_divide(np.array(measured_speed_test) - np.array(trained_model.predict(data_test)),
                                   np.array(measured_speed_test)) ** 2) ** 0.5
        return \
            msd_trained_model_applied_to_test_set, \
            trained_model, \
            data_test, \
            data_train, \
            r2_trained_model_applied_to_test_set, \
            mae_trained_model_applied_to_test_set, \
            rmse_trained_model_applied_to_test_set, \
            rmsne_trained_model_applied_to_test_set, \
            measured_speed_test

    def save_vectors_as_csv(self, att_file_paths, output_csv_path):
        self.__get_vectors_from_multiple_att_input_files(att_file_paths).to_csv(output_csv_path, index=False)

    def make_csv_for_weka(self, simulation_results_att_file_path):
        self.__get_vectors_from_multiple_att_input_files([simulation_results_att_file_path]) \
            .to_csv('/home/inonpe/Desktop/for_weka_%s.csv' % os.path.basename(simulation_results_att_file_path),
                    index=False)

    def specialized_abnormal_models(self, incident_atts):
        def do(atts, scenario, output_file_path):
            print 'Plotting for %s' % (scenario,)
            return self.__fit_model_and_plot(
                10,
                self.__get_vectors_from_multiple_att_input_files(atts),
                scenario + ', Predicted vs. Measured Speed by Demand\nOD Matrices Perturbed Element-wise by $N(1, 0.2)$',
                output_file_path)

        output_dir_path = functools.partial(os.path.join, self.output_dir, 'specialized_abnormal_models')
        create_dir(output_dir_path())
        with open(output_dir_path('stats_num_lanes_blocked.csv'), 'w') as stats_how_many_lanes_blocked:
            stats_how_many_lanes_blocked.write('num_lanes_blocked,vectors,cv_r2,cv_mae,cv_rmse\n')
            for num_lanes_blocked in (1, 2):
                num_vectors, cv_r2, cv_mae, cv_rmse = do(
                    filter(
                        lambda path: ('None' in path) == (num_lanes_blocked == 1),
                        incident_atts),
                    'Incident Scenarios, only %d lanes blocked\n' % num_lanes_blocked,
                    output_dir_path('%d_lanes_blocked.png' % num_lanes_blocked))
                stats_how_many_lanes_blocked.write(
                    '%s,%s,%s,%s,%s\n' % (num_lanes_blocked, num_vectors, cv_r2, cv_mae, cv_rmse))

        with open(output_dir_path('stats_link_location_and_which_lanes_blocked.csv'), 'w') as stats_where_blocked:
            stats_where_blocked.write('lane1,lane2,location_on_link,vectors,cv_r2,cv_mae,cv_rmse\n')
            which_lanes_blocked_options = frozenset(map(
                lambda path: re.search('(laneblock1_[^_]*_laneblock2_[^_]*)', path).groups(0)[0],
                incident_atts))
            location_of_blockage_options = frozenset(map(
                lambda path: re.search('(link74_location_[^_]*)', path).groups(0)[0],
                incident_atts))
            for which_lanes in which_lanes_blocked_options:
                for location_on_link in location_of_blockage_options:
                    num_vectors, cv_r2, cv_mae, cv_rmse = do(
                        filter(
                            lambda path: which_lanes in path and location_on_link in path,
                            incident_atts),
                        'Incident Scenarios, %s, %s\n' % (which_lanes, location_on_link),
                        output_dir_path('%s_%s.png' % (which_lanes, location_on_link)))
                    stats_where_blocked.write('%s,%s,%s,%s,%s,%s,%s\n' % (
                        re.search('laneblock1_([^_]*)', which_lanes).groups(0)[0],
                        re.search('laneblock2_([^_]*)', which_lanes).groups(0)[0],
                        re.search('link74_location_([^_]*)', location_on_link).groups(0)[0],
                        num_vectors, cv_r2, cv_mae, cv_rmse))
