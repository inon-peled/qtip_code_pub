import functools
import itertools
import os
import re

links_ordered = [73, 60, 63, 74, 66, 18, 68]

data_path = functools.partial(os.path.join, 'data')

SIMULATION_END_SECOND = 5400
INCIDENT_START_SECOND = 1500
INCIDENT_DURATION_MINUTES = 30
INCIDENT_END_SECOND = INCIDENT_START_SECOND + (60 * INCIDENT_DURATION_MINUTES)

PLOT_SIZE = 25
START_OF_LINK = 0.05
MIDDLE_OF_LINK = 0.45
END_OF_LINK = 0.90
LINK_LOCATION_NAMES = {START_OF_LINK: 'start', MIDDLE_OF_LINK: 'middle', END_OF_LINK: 'end'}
HIGH_DEMAND_FACTOR = 1.3
MEDIUM_DEMAND_FACTOR = 1.0
LOW_DEMAND_FACTOR = 0.7
DEMAND_NAMES = {HIGH_DEMAND_FACTOR: 'high', MEDIUM_DEMAND_FACTOR: 'medium', LOW_DEMAND_FACTOR: 'low'}

LANE_NAMES = ['TopLane', 'MiddleLane', 'BottomLane']
NUMBER_1_LANE_NAMES = map(lambda lane_name: lane_name + '1', LANE_NAMES)
NUMBER_2_LANE_NAMES = map(lambda lane_name: lane_name + '2', LANE_NAMES)
NUMBERED_LANE_NAMES = NUMBER_1_LANE_NAMES + NUMBER_2_LANE_NAMES
NUM_METERS_BETWEEN_CONSECUTIVE_TRAFFIC_LIGHTS_ON_SAME_LANE = 10


INCIDENT_OPTIONS = list(itertools.product(
            [HIGH_DEMAND_FACTOR, MEDIUM_DEMAND_FACTOR, LOW_DEMAND_FACTOR],
            list(itertools.chain(itertools.combinations(LANE_NAMES, 1), itertools.combinations(LANE_NAMES, 2))),
            [START_OF_LINK, MIDDLE_OF_LINK, END_OF_LINK]))


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_combinations():
    incident_options_for_1_vehicle = list(itertools.product(
        [HIGH_DEMAND_FACTOR, MEDIUM_DEMAND_FACTOR, LOW_DEMAND_FACTOR],
        map(lambda lane_name: (lane_name,), NUMBER_1_LANE_NAMES),
        [START_OF_LINK, MIDDLE_OF_LINK, END_OF_LINK]))
    incident_options_for_2_vehicles_on_different_lanes = list(itertools.product(
        [HIGH_DEMAND_FACTOR, MEDIUM_DEMAND_FACTOR, LOW_DEMAND_FACTOR],
        map(lambda tpl: (tpl[0] + '1', tpl[1] + '1'), itertools.combinations(LANE_NAMES, 2)),
        [START_OF_LINK, MIDDLE_OF_LINK, END_OF_LINK]))
    incident_options_for_2_vehicles_on_same_lane = list(itertools.product(
        [HIGH_DEMAND_FACTOR, MEDIUM_DEMAND_FACTOR, LOW_DEMAND_FACTOR],
        map(lambda lane_name: (lane_name + '1', lane_name + '2'), LANE_NAMES),
        [START_OF_LINK, MIDDLE_OF_LINK, END_OF_LINK]))
    return incident_options_for_1_vehicle + incident_options_for_2_vehicles_on_different_lanes + \
           incident_options_for_2_vehicles_on_same_lane


def convert_to_blocked_lanes(incident_att_path):
    lane_name_no_numbers_no_none = filter(
        lambda path_name: path_name != 'None',
        map(lambda lane: lane[:-1] if {'1', '2'} & set(lane) else lane,
            re.search(r'laneblock1_([^_]+)_laneblock2_([^_]+)', incident_att_path).groups())
    )
    return lane_name_no_numbers_no_none


def remove_uplink_and_downlink(vectors):
    return vectors.drop(filter(lambda colname: re.search(r'downlink|uplink', colname), vectors.columns), axis=1)


def get_vectors_for_fitting(all_vectors):
    return all_vectors.drop(
        filter(lambda colname: re.search(r'before|inverse_of_minutes_since_accident_start', colname) is None,
               all_vectors.columns), axis=1)
