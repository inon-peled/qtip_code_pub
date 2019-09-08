from common_definitions import *

import collections
import glob
import traceback
import re
import multiprocessing
import random
import datetime
import numpy
import shutil
import win32com.client as com
import os

NUM_PROCESSES = 4
# BASE_DIR = os.path.join('M:', os.sep, 'backup', 'qtip_data', 'VISSIM_files', 'inon')
BASE_DIR = os.path.join('C:', os.sep, 'Users', 'inonpe', 'Desktop', 'sim_batches')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'TemplateForIncidentModelWithTrafficLightFor1Or2VehiclesAnd5StepsPerSimSecond')


def log(msg):
    print '[%s] [%s] %s' % (datetime.datetime.now(), os.getpid(), msg)


class Link74IncidentScenarioCreatorFor1Or2Vehicles(object):
    def __init__(self, vissim, basedir, inpx, layx, od_matrices_dir, demand_factor,
                 blocked_lane_1, blocked_lane_2, incident_location_relative_to_link):
        self.vissim = vissim
        self.basedir = basedir
        self.inpx = inpx
        self.layx = layx
        self.od_matrices_dir = od_matrices_dir
        self.demand_factor = demand_factor
        self.blocked_lane_1 = blocked_lane_1
        self.blocked_lane_2 = blocked_lane_2
        self.incident_location_relative_to_link = incident_location_relative_to_link

    def __load_base_network(self):
        self.vissim.LoadNet(self.inpx, False)
        self.vissim.LoadLayout(self.layx)
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue('QuickMode', 1)
        return self

    def __modify_network(self):
        return self \
            .__set_incident_location() \
            .__set_blocked_lanes() \
            .__set_matrices()

    def __set_matrices(self):
        def load_and_perturb(index, basename, matname):
            mat = self.vissim.Net.Matrices.ItemByKey(index)
            mat.ReadFromFile(os.path.join(self.od_matrices_dir, basename))
            mat.SetAttValue('Name', matname)
            # NOTE: could yield negative elements, though very unlikely.
            for i in range(1, mat.RowCount + 1):
                for j in range(1, mat.ColCount + 1):
                    mat.SetValue(i, j, int(mat.GetValue(i, j) * self.demand_factor * numpy.random.normal(1.0, 0.2)))

        map(lambda b_m: load_and_perturb(*b_m), [
            (1, 'AllModels1stQuarter.fma', 'First15MinutesPerturbed'),
            (2, 'BaseDemand.fma', 'NextHourDemandPerturbed'),
            (3, 'CalibratedBasicModel2ndQuarter.fma', 'Last15MinutesPerturbed')
        ])
        return self

    def __set_blocked_lanes(self):
        def remove_one_traffic_light(lane_name):
            self.vissim.Net.SignalHeads.RemoveSignalHead(
                self.__get_by_name(lane_name, self.vissim.Net.SignalHeads))

        unblocked_lanes = frozenset(NUMBERED_LANE_NAMES) - \
                          frozenset([self.blocked_lane_1, self.blocked_lane_2])
        map(remove_one_traffic_light, unblocked_lanes)
        return self

    @staticmethod
    def __get_by_name(name, iterable):
        return list(filter(lambda r: r.AttValue('Name') == name, iterable))[0]

    def __set_incident_location(self):
        link_length = self.vissim.Net.Links.ItemByKey(74).AttValue('Length2D')
        location_on_link_for_lights = link_length * self.incident_location_relative_to_link

        def reposition_traffic_lights(numbered_lane_name):
            traffic_light_number = int(re.search(r'(\d+$)', numbered_lane_name).groups(0)[0])
            position = location_on_link_for_lights - \
                       NUM_METERS_BETWEEN_CONSECUTIVE_TRAFFIC_LIGHTS_ON_SAME_LANE * (traffic_light_number - 1)
            self.__get_by_name(numbered_lane_name, self.vissim.Net.SignalHeads).SetAttValue('Pos', position)

        def reposition_data_collection_point(lane):
            self.__get_by_name(lane, self.vissim.Net.DataCollectionPoints).SetAttValue(
                'Pos', location_on_link_for_lights + 5)

        def reposition_queue_counters():
            self.__get_by_name('IncidentLocation', self.vissim.Net.QueueCounters).SetAttValue(
                'Pos', location_on_link_for_lights + 2 * 5)

        reposition_queue_counters()
        map(reposition_data_collection_point, LANE_NAMES)
        map(reposition_traffic_lights, NUMBERED_LANE_NAMES)
        return self

    def run_single_simulation(self):
        self.__load_base_network()
        self.__modify_network()
        self.vissim.Simulation.SetAttValue('SimRes', 5)
        self.vissim.Simulation.SetAttValue('NumRuns', 1)
        self.vissim.Simulation.SetAttValue('NumCores', 1)
        self.vissim.Simulation.SetAttValue('RandSeed', random.randint(1, 2 ** 30))
        self.vissim.Simulation.RunContinuous()
        self.vissim.SaveNet()
        return self


class Link74IncidentScenarioRunnerFor1Or2Vehicles(object):
    def __init__(self):
        self.vissim = com.Dispatch('Vissim.Vissim')

    def bootstrap(self,
                  scenarios_output_dir,
                  demand_factor,
                  blocked_lanes,
                  incident_location_on_link):
        blocked_lane_1 = blocked_lanes[0]
        blocked_lane_2 = blocked_lanes[1] if len(blocked_lanes) > 1 else None
        scenario_dir = os.path.join(
            scenarios_output_dir,
            '5_steps_per_sim_second',
            'incident_scenarios_for_1_or_2_vehicles',
            'incident_link74_location_%s_durationmin_%d_demand_%s_laneblock1_%s_laneblock2_%s_simrun_%d' %
            (LINK_LOCATION_NAMES[incident_location_on_link],
             INCIDENT_DURATION_MINUTES,
             DEMAND_NAMES[demand_factor],
             blocked_lane_1,
             blocked_lane_2,
             random.randint(2 ** 31, 2 ** 32)))
        shutil.copytree(TEMPLATE_DIR, scenario_dir)
        return Link74IncidentScenarioCreatorFor1Or2Vehicles(
            self.vissim,
            scenario_dir,
            os.path.join(scenario_dir, 'incident.inpx'),
            os.path.join(scenario_dir, 'incident.layx'),
            os.path.join(BASE_DIR, 'od_matrices'),
            demand_factor,
            blocked_lane_1,
            blocked_lane_2,
            incident_location_on_link
        )


def multiprocessed_run_num_simulations_for_each_scenario_params(
        scenarios_output_dir,
        pipe_parent_to_child,
        pipe_child_to_parent,
        scenario_params,
        num_simulations):
    try:
        log('Start creating VISSIM executor')
        executor = Link74IncidentScenarioRunnerFor1Or2Vehicles()
        log('Finish creating VISSIM executor')
        pipe_child_to_parent[1].send(True)
    except Exception as exc:
        log('Failed to create VISSIM executor')
        traceback.print_exc(exc)
        pipe_child_to_parent[1].send(False)
    if not pipe_parent_to_child[1].recv():
        return
    for params in scenario_params:
        for i in range(num_simulations):
            try:
                log('Start running simulation %d of %d with parameters %s' % (i + 1, num_simulations, str(params)))
                executor \
                    .bootstrap(scenarios_output_dir, *params) \
                    .run_single_simulation()
                log('Finished running simulation %d of %d with parameters %s' % (i + 1, num_simulations, str(params)))
            except Exception as exc:
                log('ERROR: Failed to run simulation %d of %d with parameters %s' %
                    (i + 1, num_simulations, str(params)))
                traceback.print_exc(exc)


def __run_given_combinations_of_incident_parameters(scenarios_output_dir, combinations_of_incident_parameters,
                                                    num_repetitions_of_each_combination):
    random.shuffle(combinations_of_incident_parameters)
    pipes = [(multiprocessing.Pipe(), multiprocessing.Pipe()) for _ in range(NUM_PROCESSES)]
    processes = [multiprocessing.Process(
        target=multiprocessed_run_num_simulations_for_each_scenario_params,
        args=(scenarios_output_dir,
              pipes[i][0],
              pipes[i][1],
              combinations_of_incident_parameters[i::NUM_PROCESSES],
              num_repetitions_of_each_combination))
        for i in range(NUM_PROCESSES)]
    map(lambda p: p.start(), processes)
    for pipe_parent_to_child, pipe_child_to_parent in pipes:
        if not pipe_child_to_parent[0].recv():
            log('Some VISSIM executor failed to create')
            map(lambda pipe_pair: pipe_pair[0][0].send(False), pipes)
            return
    log('All %d VISSIM executors successfully created' % NUM_PROCESSES)
    map(lambda pipe_pair: pipe_pair[0][0].send(True), pipes)
    map(lambda p: p.join(), processes)


def run_all_combinations(scenarios_output_dir, num_simulations):
    __run_given_combinations_of_incident_parameters(scenarios_output_dir, get_combinations(), num_simulations)


def __run_missing_combinations_if_previous_run_failed_midway_through(scenarios_output_dir, dirpath, num_repetitions):
    no_results = \
        frozenset(glob.glob(os.path.join(dirpath, '*'))) - frozenset(
            map(os.path.dirname,glob.glob(os.path.join(dirpath, '*', 'incident_Link Segment Results_001.att'))))
    map(shutil.rmtree, no_results)
    finished = collections.Counter(map(__get_incident_parameters_from_directory_name, glob.glob(os.path.join(dirpath, '*'))))
    expected = collections.Counter(get_combinations() * num_repetitions)
    missing_parameter_combinations = expected - finished
    if missing_parameter_combinations:
        __run_given_combinations_of_incident_parameters(scenarios_output_dir, list(missing_parameter_combinations.elements()), 1)
    return missing_parameter_combinations


def __get_incident_parameters_from_directory_name(directory_path):
    m = re.search(r'incident_link74_location_(?P<location>[^_]+)' +
                  r'_durationmin_' + str(INCIDENT_DURATION_MINUTES) +
                  r'_demand_(?P<demand>[^_]+)' +
                  r'_laneblock1_(?P<laneblock1>[^_]+)' +
                  r'_laneblock2_(?P<laneblock2>[^_]+)', directory_path)
    location_name_to_float = dict(pair[::-1] for pair in LINK_LOCATION_NAMES.items())
    demand_name_to_float = dict(pair[::-1] for pair in DEMAND_NAMES.items())
    blocked_lanes = \
        (m.group('laneblock1'),) if m.group('laneblock2') == 'None' else (m.group('laneblock1'), m.group('laneblock2'))
    return demand_name_to_float[m.group('demand')], \
           blocked_lanes, \
           location_name_to_float[m.group('location')]


def run_multiple_bathces_because_vissim_sometimes_crashes(num_batches):
    for i in range(1, num_batches + 1):
        print('[%s] Starting batch %d/%d' % (datetime.datetime.now(), i, num_batches))
        run_all_combinations(os.path.join(BASE_DIR, 'batch%d' % i), 1)
        print('[%s] Finished batch %d/%d' % (datetime.datetime.now(), i, num_batches))

if __name__ == '__main__':
    # pass
    run_multiple_bathces_because_vissim_sometimes_crashes(10)

    # print(__run_missing_combinations_if_previous_run_failed_midway_through(
    #     os.path.join(BASE_DIR, '5_steps_per_sim_second', 'incident_scenarios_for_1_or_2_vehicles'), 20))

    # Link74IncidentScenarioRunnerFor1Or2Vehicles()\
    #     .bootstrap(LOW_DEMAND_FACTOR, ['TopLane1', 'TopLane2'], START_OF_LINK)\
    #     .run_single_simulation()
