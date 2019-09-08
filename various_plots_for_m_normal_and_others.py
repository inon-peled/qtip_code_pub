import logging

from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import glob

from common_definitions import *
from regressor import Regressor

ARBITRARY_SEED = 777


def degradation_m_ordinary_hexbin(regressor, normal_atts, abnormal_atts):
    for with_other_links in (False, True):
        regressor.m_ordinary_degradation_on_test_vectors(
            with_other_links,
            normal_atts,
            abnormal_atts,
            'm_ordinary_degradation_scatter_other_links_%d.png' % with_other_links,
            '$M_{ordinary}$ %s $U$ and $D$, Incident Scenarios' %
                ('with' if with_other_links else 'without'))

    # regressor.combined_model(normal_atts, abnormal_atts, 0.2)


# @multiprocessed
def __plot_various_time_series(args):
    scenario_results_dir, num_lags, earliest_lag_minutes_back = args
    two_piece_t_range = [range(INCIDENT_START_SECOND, INCIDENT_START_SECOND + 6 * 60, 60),
                         range(INCIDENT_START_SECOND + 6 * 60, INCIDENT_END_SECOND + 1, 60)]
    regressor = Regressor(output_subdir='output_plots_degradation_m_ordinary_LR',
                          logger=logger,
                          t_ranges_for_piecewise_modeling=two_piece_t_range,
                          regression_cls=LinearRegression,
                          regressor_init_kwargs=dict(fit_intercept=False),
                          scenario_results_dir=scenario_results_dir,
                          seed=ARBITRARY_SEED,
                          num_lags=num_lags,
                          earliest_lag_minutes_back=earliest_lag_minutes_back)
    # plotter = Plotter(output_dir='output_tmp',
    #                   scenario_results_dir=None,
    #                   seed=ARBITRARY_SEED,
    #                   num_lags=num_lags,
    #                   earliest_lag_minutes_back=earliest_lag_minutes_back)

    normal_atts = glob.glob(os.path.join(
        scenario_results_dir,
        'normal_scenario_demand_*',
        '*',
        '*Link Segment Results_001.att'))

    regressor.compare_scatter_of_normal_model_with_and_without_other_links(normal_atts)

    abnormal_atts = glob.glob(os.path.join(
        scenario_results_dir,
        'incident_scenarios',
        'incident_link74_location_*',
        'results.att'))

    degradation_m_ordinary_hexbin(regressor, normal_atts, abnormal_atts)

    regressor.time_series_degradation_of_normal_model_with_and_without_other_links(
        normal_atts, abnormal_atts, 'time_series_degradation_of_normal_model_with_or_without_other_links.png')

    # regressor.time_series_degradation_of_normal_model_with_other_links(
    #     normal_atts, abnormal_atts, 'time_series_degradation_of_normal_model_with_other_links.png')
    #
    # regressor.time_series_degradation_of_normal_model_without_other_links(
    #     normal_atts, abnormal_atts, 'time_series_degradation_of_normal_model_without_other_links.png')
    #
    # plotter.plot_average_time_series_only_normal_scenarios(normal_atts)
    #
    # plotter.plot_average_time_series_only_abnormal_scenarios(abnormal_atts)
    #
    # plotter.plot_typical_behavior_of_average_speed_under_incident(abnormal_atts)

    # regressor.specialized_abnormal_models(abnormal_atts)
    #
    # regressor.create_time_series_plots_for_distress_signal(normal_atts, abnormal_atts, 0.2)


if __name__ == '__main__':
    # scenario_results_dir = 'for_debugging'
    scenario_results_dir = os.path.join('copy_of_m_drive', 'sim_5sec')
    # scenario_results_dir = 'M:\\vbox_shared\\VISSIM_files\\inon\\' if platform.system() == 'Windows' \
    #     else '/run/user/1000/gvfs/smb-share:server=lynnas.win.dtu.dk,share=inonpe/vbox_shared/VISSIM_files/inon/'
    # scenario_results_dir = os.path.join('.', 'debug_scenario_results_dir')

    # Pool(maxtasksperchild=1).map(multiprocessed_plot_various_time_series,
    #                              itertools.product([scenario_results_dir], range(2, 3), range(5, 6)))

    __plot_various_time_series((scenario_results_dir, 2, 5))
