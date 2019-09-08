import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FuncFormatter

import datetime
import numpy as np
from sklearn import metrics

from common_definitions import *
from get_vectors import get_vectors_from_multiple_att_input_files


def shorten_column_names(column_names):
    return map(lambda col_name:
               ('$1/T_{accident}$' if col_name == 'inverse_of_minutes_since_accident_start' else
               ('$%s_{%s}$' % (col_name[0].upper(), re.search(r'[0-9]+', col_name).group(0)))),
               column_names)


def plot_45_degrees_line(ax):
    ax.plot(np.linspace(*ax.get_xlim()), np.linspace(*ax.get_xlim()), 'k--', lw=2)


class Plotter(object):
    def __init__(self, output_dir, scenario_results_dir, seed, num_lags, earliest_lag_minutes_back):
        self.seed = seed if seed is not None else int(datetime.datetime.now().strftime('%s'))
        self.earliest_lag_minutes_back = earliest_lag_minutes_back
        self.num_lags = num_lags
        self.scenario_results_dir = scenario_results_dir
        self.output_dir = output_dir
        create_dir(self.output_dir)

    def scatter_plot_model_degradation_on_time_pieces(
            self, num_normal, num_abnormal, r2_normal_model_applied_to_abnormal,
            mae_normal_model_applied_to_abnormal,
            rmse_normal_model_applied_to_abnormal, measured_abnormal, predictions, title, output_file_path):
        matplotlib.rc('font', **{'weight': 'bold', 'size': PLOT_SIZE})
        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
        ax.set_xlabel('Measured Speed (km/h)')
        ax.set_ylabel('Predicted Speed (km/h)')
        ax.set_facecolor('lightgrey')
        ax.set_title(title)
        ax.scatter(measured_abnormal, predictions, s=200, color='b', marker='x')
        plot_45_degrees_line(ax)
        ax.text(0.05, 0.95,
                '|train|=%d, |test|=%d\nMAE$=%.3f$\nRMSE$=%.3f$' %
                (num_normal, num_abnormal,
                 mae_normal_model_applied_to_abnormal,
                 rmse_normal_model_applied_to_abnormal),
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=PLOT_SIZE * 1.3, fontweight='bold')
        self.__savefig_and_close(output_file_path)

    def plot_combined_model(self, measurements,
                            fraction_simulations_for_test_set, num_vectors_test, num_vectors_train_normal):
        matplotlib.rc('font', **{'weight': 'bold', 'size': PLOT_SIZE})
        fig, axarr = plt.subplots(3, figsize=(PLOT_SIZE, PLOT_SIZE))
        for i in range(3):
            axarr[i].plot(measurements.index, measurements.iloc[:, i])
            axarr[i].set_facecolor('lightgrey')
            axarr[i].set_xlabel('%$Vectors_{abnormal}$ in train set')
            axarr[i].xaxis.set_minor_locator(MultipleLocator(2))
            axarr[i].grid(which='both')
            axarr[i].set_ylabel(measurements.columns[i])
        axarr[0].set_title('Improvement of $M_{combined}$ as $Vectors_{abnormal}$ are added to train set')
        axarr[0].text(
            0.35, 0.25,
            ('Test set is %d%% of all simulations,\nnormal and abnormal (%d vectors).' %
             (100 * fraction_simulations_for_test_set, num_vectors_test)) +
            ('\n\nTrain set comprises of %d $Vectors_{normal}$ and\nincreasing percentage of $Vectors_{abnormal}.$' %
             num_vectors_train_normal),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75),
            fontsize=PLOT_SIZE * 1.3, fontweight='bold',
            transform=axarr[0].transAxes)
        self.__savefig_and_close('stats_for_combined_model.png')

    def plot_average_time_series_only_normal_scenarios(self, normal_atts):
        def plt_avg(atts, output_file_path, title):
            vectors = self.__get_vectors_from_multiple_att_input_files(atts)
            speed_stats = vectors.groupby('t').agg(['mean', 'std']).speed
            fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
            matplotlib.rc('font', **{'weight': 'bold', 'size': PLOT_SIZE})
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(PLOT_SIZE * 1.3)
            # TODO: Change X axis to time of day
            # TODO: Add indication of accident occurence?
            ax.set_title(title)
            ax.plot(speed_stats.index, speed_stats['mean'], color='blue', marker='o', markersize=PLOT_SIZE // 1.5)
            plt.legend(handles=[mpatches.Patch(color='lightgrey', label='Standard Deviation')])
            ax.set_ylabel('1min. Mean Speed on Link (km/h)')
            ax.fill_between(speed_stats.index,
                            speed_stats['mean'] - speed_stats['std'],
                            speed_stats['mean'] + speed_stats['std'],
                            facecolor='lightgrey',
                            alpha=0.5)
            ax.text(0.8, 0.85, '%d Data Vectors' % len(vectors),
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=PLOT_SIZE, fontweight='bold', family='monospace')
            self.__savefig_and_close(output_file_path)

        plt_avg(
            normal_atts, 'normal_scenario_all_demand.png', 'Normal Scenarios, All Demand Levels')

        for demand_name in ['low', 'medium', 'high']:
            plt_avg(
                filter(lambda path: 'normal_scenario_demand_%s' % demand_name in path, normal_atts),
                'normal_scenario_demand_%s.png' % demand_name, 'Normal Scenarios, %s Demand' % demand_name)

    def plot_average_time_series_only_abnormal_scenarios(self, abnormal_atts):
        def plot_average_time_series_for_incident_scenarios(atts, output_file_path, title):
            vectors = self.__get_vectors_from_multiple_att_input_files(atts)
            speed_stats = vectors.groupby('t').agg(['mean', 'std']).speed
            fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE * 0.5))
            matplotlib.rc('font', **{'weight': 'bold', 'size': PLOT_SIZE})
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(PLOT_SIZE * 1.3)
            # TODO: Change X axis to time of day
            # TODO: Add legend for mean
            ax.set_title(title)
            ax.plot(speed_stats.index, speed_stats['mean'], color='blue', marker='o', markersize=PLOT_SIZE // 1.5)
            ax.text(0.8, 0.85, '%d Data Vectors' % len(vectors),
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=PLOT_SIZE, fontweight='bold', family='monospace')

            def fill_between_times(t_range, color):
                ax.fill_between(speed_stats.index,
                                speed_stats['mean'] - speed_stats['std'],
                                speed_stats['mean'] + speed_stats['std'],
                                where=speed_stats.index.isin(t_range),
                                facecolor=color,
                                alpha=0.5)

            # for x in [INCIDENT_START_SECOND, INCIDENT_END_SECOND]:
            #     ax.axvline(x, color='red', linewidth=5, alpha=0.7, linestyle='--')
            fill_between_times(range(INCIDENT_START_SECOND), 'green')
            fill_between_times(range(INCIDENT_START_SECOND, INCIDENT_END_SECOND), 'red')
            fill_between_times(range(INCIDENT_END_SECOND, SIMULATION_END_SECOND + 1), 'green')

            # ax.errorbar(speed_stats.index, speed_stats['mean'], yerr=speed_stats['std'])
            plt.legend(handles=[mpatches.Patch(color='lightgrey', label='Standard Deviation')])
            ax.set_ylabel('1min. Mean Speed on Link (km/h)')
            self.__savefig_and_close(output_file_path)

        plot_average_time_series_for_incident_scenarios(
            abnormal_atts, 'incident_scenarios_all_variations.png', 'Incident Scenarios, All Variations')

        plot_average_time_series_for_incident_scenarios(
            filter(lambda att_path: len(set(convert_to_blocked_lanes(att_path))) == 1, abnormal_atts),
            'incident_scenarios_any_single_lane_blocked.png', 'Incident Scenarios, any Single Lane Blocked')

        # for demand, blocked_lanes, location in INCIDENT_OPTIONS:
        #     blocked_lane_1 = blocked_lanes[0]
        #     blocked_lane_2 = blocked_lanes[1] if len(blocked_lanes) > 1 else None
        #     location_name = LINK_LOCATION_NAMES[location]
        #     demand_name = DEMAND_NAMES[demand]
        #     atts = glob.glob(os.path.join(
        #         self.scenario_results_dir,
        #         'incident_scenarios_for_1_or_2_vehicles',
        #         '*location_%s_*_demand_%s_laneblock1_%s_laneblock2_%s_*' % (
        #             location_name, demand_name, blocked_lane_1, blocked_lane_2),
        #         'results.att'))
        #     output_file_path = 'incident_scenarios_location_%s_demand_%s_laneblock1_%s_laneblock2_%s.png' % (
        #         location_name, demand_name, blocked_lane_1, blocked_lane_2)
        #     title = 'incident_scenarios_location_%s_demand_%s_laneblock1_%s_laneblock2_%s' % (
        #         location_name, demand_name, blocked_lane_1, blocked_lane_2)
        #     self.plot_average_time_series_for_incident_scenarios(atts, output_file_path, title)

    def plot_typical_behavior_of_average_speed_under_incident(self, abnormal_atts):
        vectors = self.__get_vectors_from_multiple_att_input_files(abnormal_atts)

        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE * 0.5))
        matplotlib.rc('font', **{'weight': 'bold', 'size': PLOT_SIZE})
        for item in ([ax.title, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(PLOT_SIZE)

        speed_stats = vectors.groupby('t').agg(['mean', 'std']).speed
        ax.plot(speed_stats.index, speed_stats['mean'], color='purple', marker='s', markersize=PLOT_SIZE // 1.5)

        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.set_xlim(60 * 15, 5400 - 15 * 60)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: (datetime.datetime(year=2017, month=1, day=1, hour=6, minute=45) +
                                          datetime.timedelta(seconds=x)).time().strftime('%H:%M')))
        fig.autofmt_xdate()
        ax.xaxis.set_tick_params(width=5)
        ax.grid(which='major', color='black', alpha=0.5)
        ax.grid(which='minor', alpha=0.5)

        ax.set_ylabel('1min. Mean Speed on Link (km/h)')
        plt.title('Typical Mean Speed under Incident Conditions')
        self.__savefig_and_close('typical_average_speed_under_incident.png')

    def plot_with_and_without_u_d_on_average_time_series_no_frills(
            self, test_set_vectors_with_u_d, pairs_of_t_range_and_model_with_u_d,
            test_set_vectors_without_u_d, pairs_of_t_range_and_model_without_u_d,
            output_file_name, title):
        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE * 0.5))
        matplotlib.rc('font', **{'size': PLOT_SIZE * 1.5})  # 'weight': 'bold'
        for item in ([ax.title, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(PLOT_SIZE)

        test_set_speed_stats = test_set_vectors_with_u_d.groupby('t').agg(['mean', 'std']).speed
        ax.plot(test_set_speed_stats.index, test_set_speed_stats['mean'],
                color='blue', marker='o', markersize=PLOT_SIZE // 2, label='Avg. Test Set')

        for test_set_vectors, pairs_of_t_range_and_model, marker, markersize, label in [
            (test_set_vectors_with_u_d, pairs_of_t_range_and_model_with_u_d, 'd', PLOT_SIZE // 1.7, 'with $U, D$'),
            (test_set_vectors_without_u_d, pairs_of_t_range_and_model_without_u_d, 's', PLOT_SIZE // 2, 'without $U, D$')]:
            for i, (t_range_for_predictions, trained_model) in enumerate(pairs_of_t_range_and_model):
                test_vectors_in_t_range = test_set_vectors[lambda df: df.t.isin(t_range_for_predictions)]
                average_test_vectors = get_vectors_for_fitting(test_vectors_in_t_range.groupby('t').mean())
                average_predictions = trained_model.predict(average_test_vectors.as_matrix())
                ax.plot(t_range_for_predictions, average_predictions, color='red',
                        marker=marker, markersize=markersize, linestyle='--',
                        label=('Prediction %s' % label) if i == 0 else None)
                        # markeredgecolor='black', markeredgewidth=3)

        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.set_xlim(60 * 15, 5400 - 15 * 60)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: (datetime.datetime(year=2017, month=1, day=1, hour=6, minute=45) +
                                          datetime.timedelta(seconds=x)).time().strftime('%H:%M')))
        fig.autofmt_xdate()
        ax.xaxis.set_tick_params(width=5)
        ax.grid(which='major', color='black', alpha=0.5)
        ax.grid(which='minor', alpha=0.5)

        ax.set_ylabel('1min. Mean Speed on Link (km/h)')
        plt.legend(facecolor='lightgrey', framealpha=0.9)
        plt.title(title)
        self.__savefig_and_close(output_file_name)

    def plot_on_average_time_series_no_frills(
            self, test_set_vectors, pairs_of_t_range_and_model, output_file_name, title):
        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE * 0.5))
        matplotlib.rc('font', **{'weight': 'bold', 'size': PLOT_SIZE})
        for item in ([ax.title, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(PLOT_SIZE)

        test_set_speed_stats = test_set_vectors.groupby('t').agg(['mean', 'std']).speed
        ax.plot(test_set_speed_stats.index, test_set_speed_stats['mean'],
                color='blue', marker='o', markersize=PLOT_SIZE // 1.5, label='Avg. Test Set')

        for i, (t_range_for_predictions, trained_model) in enumerate(pairs_of_t_range_and_model):
            test_vectors_in_t_range = test_set_vectors[lambda df: df.t.isin(t_range_for_predictions)]
            average_test_vectors = get_vectors_for_fitting(test_vectors_in_t_range.groupby('t').mean())
            average_predictions = trained_model.predict(average_test_vectors.as_matrix())
            ax.plot(t_range_for_predictions, average_predictions, color='red',
                    marker='^', markersize=PLOT_SIZE // 1.5,
                    label='$M_{ordinary}$ Prediction' if i == 0 else None, markeredgecolor='black', markeredgewidth=3)

        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.set_xlim(60 * 15, 5400 - 15 * 60)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: (datetime.datetime(year=2017, month=1, day=1, hour=6, minute=45) +
                                          datetime.timedelta(seconds=x)).time().strftime('%H:%M')))
        fig.autofmt_xdate()
        ax.xaxis.set_tick_params(width=5)
        ax.grid(which='major', color='black', alpha=0.5)
        ax.grid(which='minor', alpha=0.5)

        ax.set_ylabel('1min. Mean Speed on Link (km/h)')
        plt.legend(facecolor='lightgrey', framealpha=0.9)
        plt.title(title)
        self.__savefig_and_close(output_file_name)

    def plot_on_average_time_series(
            self, train_set_vectors, test_set_vectors, pairs_of_t_range_and_model, output_file_name, title):
        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE * 0.5))
        matplotlib.rc('font', **{'size': PLOT_SIZE * 1.2})  # 'weight': 'bold'
        for item in ([ax.title, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(PLOT_SIZE)

        train_set_speed_stats = train_set_vectors.groupby('t').mean().speed
        ax.plot(train_set_speed_stats.index, train_set_speed_stats.values,
                color='purple', marker='s', markersize=PLOT_SIZE // 1.5, label='Avg. Train Set', alpha=0.3)

        test_set_speed_stats = test_set_vectors.groupby('t').agg(['mean', 'std']).speed
        ax.plot(test_set_speed_stats.index, test_set_speed_stats['mean'],
                color='blue', marker='o', markersize=PLOT_SIZE // 1.5, label='Avg. Test Set')

        def fill_between_times(t_range, color, label):
            ax.fill_between(test_set_speed_stats.index,
                            test_set_speed_stats['mean'] - test_set_speed_stats['std'],
                            test_set_speed_stats['mean'] + test_set_speed_stats['std'],
                            where=test_set_speed_stats.index.isin(t_range),
                            facecolor=color,
                            alpha=0.5,
                            label=label)

        fill_between_times(range(INCIDENT_START_SECOND), 'green', 'Std. Dev., No Blockage')
        fill_between_times(range(INCIDENT_START_SECOND, INCIDENT_END_SECOND), 'red', 'Std. Dev., Blockage')
        fill_between_times(range(INCIDENT_END_SECOND, SIMULATION_END_SECOND + 1), 'green', None)

        stats = []
        for i, (t_range_for_predictions, trained_model) in enumerate(pairs_of_t_range_and_model):
            test_vectors_in_t_range = test_set_vectors[lambda df: df.t.isin(t_range_for_predictions)]
            average_test_vectors = get_vectors_for_fitting(test_vectors_in_t_range.groupby('t').mean())
            average_predictions = trained_model.predict(average_test_vectors.as_matrix())
            ax.plot(t_range_for_predictions, average_predictions, color='yellow',
                    marker='^', markersize=PLOT_SIZE // 1.5,
                    label='Prediction' if i == 0 else None, markeredgecolor='black', markeredgewidth=3)

            # TODO: Change to retrieve parameters of any model.
            # if (hasattr(trained_model, 'coef_')):
            #     fit_equation = ('$S_%d$= ' % i) + '\n    '.join(
            #         list(map(lambda coef_var: '%+.3f %s' % coef_var,
            #                  filter(lambda coef_var: coef_var[0] != 0,
            #                         zip(trained_model.coef_, shorten_column_names(average_test_vectors.columns))))) +
            #         ['%+.3f' % trained_model.intercept_])
            #     plt.gca().text(0.05 + 0.2 * (i + 1), 0.98, fit_equation,
            #                    transform=ax.transAxes, verticalalignment='top',
            #                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            #                    fontsize=PLOT_SIZE * 0.77, fontweight='bold', family='monospace')

            individual_predictions = trained_model.predict(
                get_vectors_for_fitting(test_vectors_in_t_range).as_matrix())
            r2 = metrics.r2_score(test_vectors_in_t_range.speed, individual_predictions)
            mae = metrics.mean_absolute_error(test_vectors_in_t_range.speed, individual_predictions)
            rmse = metrics.mean_squared_error(test_vectors_in_t_range.speed, individual_predictions) ** 0.5
            rmsne = np.mean(np.true_divide(np.array(test_vectors_in_t_range.speed) - np.array(individual_predictions),
                              np.array(test_vectors_in_t_range.speed)) ** 2) ** 0.5
            stats.append([i, r2, mae, rmse, rmsne])

        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.set_xlim(60 * 15, 5400 - 15 * 60)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: (datetime.datetime(year=2017, month=1, day=1, hour=6, minute=45) +
                                          datetime.timedelta(seconds=x)).time().strftime('%H:%M')))
        fig.autofmt_xdate()
        ax.xaxis.set_tick_params(width=5)
        ax.grid(which='major', color='black', alpha=0.5)
        ax.grid(which='minor', alpha=0.5)

        ax.set_ylabel('1min. Mean Speed on Link (km/h)')
        plt.legend(facecolor='lightgrey', framealpha=0.9, loc='lower right', fontsize=PLOT_SIZE * 0.8)
        plt.title(title)
        self.__savefig_and_close(output_file_name)
        return stats

    def __savefig_and_close(self, output_file_path):
        output_full_path = os.path.join(self.output_dir, output_file_path)
        create_dir(os.path.dirname(output_full_path))
        plt.savefig(output_full_path)
        plt.close('all')
        print '--------------- Saved figure to %s ---------------' % output_full_path

    def __get_vectors_from_multiple_att_input_files(self, atts):
        return get_vectors_from_multiple_att_input_files(self.num_lags, self.earliest_lag_minutes_back, atts)

    def plot_lr(self, cv_predictions, cv_r2, cv_msd, cv_mae, cv_rmse, vectors_for_fitting, fit_lr,
                cross_validation_folds, all_vectors, title, output_file_path):
        matplotlib.rc('font', **{'size': PLOT_SIZE * 1.8})  # 'weight': 'bold',
        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
        ax.set_xlabel('Measured Speed (km/h)')
        ax.set_ylabel('Predicted Speed (km/h)')
        ax.set_facecolor('lightgrey')

        measured = all_vectors.speed.as_matrix()
        fit_equation = 'S = ' + '\n    '.join(
            list(map(lambda coef_var: '%+.3f %s' % coef_var,
                     filter(lambda coef_var: coef_var[0] != 0,
                            zip(fit_lr.coef_, shorten_column_names(vectors_for_fitting.columns))))) +
            ['%+.3f' % fit_lr.intercept_])

        text = 'MSD = %.3f\nMAE = %.3f\nRMSE = %.3f' % (cv_msd, cv_mae, cv_rmse)
        ax.text(0.05, 0.95, text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=PLOT_SIZE * 2, family='monospace')

        ax.set_title(title)
        for demand, color in [('high', 'r'), ('medium', 'yellow'), ('low', 'g')]:
            match_demand = all_vectors.demand.str.lower() == demand
            if any(match_demand):
                ax.scatter(measured[match_demand], cv_predictions[match_demand],
                           s=400, color=color, label='%s demand' % demand, marker='x')
        plot_45_degrees_line(ax)
        ax.legend(loc='lower right', shadow=True).get_frame().set_facecolor('lightgrey')
        self.__savefig_and_close(output_file_path)

    def hexbin_plot_model_degradation(self, msd_normal_model_applied_to_abnormal,
                                      predictions, num_normal, num_abnormal,
                                      r2_normal_model_applied_to_abnormal,
                                      mae_normal_model_applied_to_abnormal,
                                      rmse_normal_model_applied_to_abnormal,
                                      measured_abnormal, title, output_file_path):

        matplotlib.rc('font', **{'size': PLOT_SIZE * 1.8})  # 'weight': 'bold',
        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

        ax.set_xlabel('Measured Speed (km/h)')
        ax.set_ylabel('Predicted Speed (km/h)')
        ax.set_title(title)

        hb = ax.hexbin(measured_abnormal, predictions, gridsize=200, cmap='PuBu')
        ax.text(0.05, 0.95,
                'MSD$=%.3f$\nMAE$=%.3f$\nRMSE$=%.3f$' %
                (msd_normal_model_applied_to_abnormal,
                 mae_normal_model_applied_to_abnormal,
                 rmse_normal_model_applied_to_abnormal),
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=PLOT_SIZE * 1.9)

        ax.xaxis.set_ticks(np.arange(0, ax.get_xlim()[1], 10))
        ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 10))
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 90)
        ax.grid()
        plot_45_degrees_line(ax)
        plt.colorbar(hb, fraction=0.025, pad=0.05, aspect=38).set_label('Count')
        plt.gca().set_aspect('equal', adjustable='box')
        self.__savefig_and_close(output_file_path)
