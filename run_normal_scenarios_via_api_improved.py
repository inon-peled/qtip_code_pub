from multiprocessing import Process, Queue
import datetime
import random

import numpy
import shutil
import os
import win32com.client as com

NUM_PROCESSES = 4
BASE_DIR = os.path.join('C:', os.sep, 'Users', 'inonpe', 'PycharmProjects', 'incident_modeling', 'normal_5sec')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'TemplateForNormalScenario_5sec')

HIGH_DEMAND_FACTOR = 1.3
MEDIUM_DEMAND_FACTOR = 1.0
LOW_DEMAND_FACTOR = 0.7
DEMAND_NAMES = {HIGH_DEMAND_FACTOR: 'high', MEDIUM_DEMAND_FACTOR: 'medium', LOW_DEMAND_FACTOR: 'low'}


def log(msg):
    print '[%s] [pid %s] %s' % (datetime.datetime.now(), os.getpid(), msg)


class NormalScenarioCreator(object):
    def __init__(self, vissim, scenario_dir, inpx, layx, od_matrices_dir, demand_factor, num_simulations, num_cores):
        self.vissim = vissim
        self.scenario_dir = scenario_dir
        self.inpx = inpx
        self.layx = layx
        self.od_matrices_dir = od_matrices_dir
        self.demand_factor = demand_factor
        self.num_simulations = num_simulations
        self.num_cores = num_cores

    def __load_base_network(self):
        self.vissim.LoadNet(self.inpx, False)
        self.vissim.LoadLayout(self.layx)
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue('QuickMode', 1)
        return self

    def __set_perturbed_matrices(self, iteration):
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

    def run_simulation_batch(self):
        self.__load_base_network()
        self.vissim.Simulation.SetAttValue('NumRuns', 1)
        for i in range(self.num_simulations):
            self.vissim.Simulation.SetAttValue('RandSeed', random.randint(1, 2 ** 30))
            self.vissim.Simulation.SetAttValue('NumCores', self.num_cores)
            log('Start run of simulation %d of %d' % (i + 1, self.num_simulations))
            self.__set_perturbed_matrices(i)
            self.vissim.Simulation.RunContinuous()
            self.vissim.SaveNet()
            log('Finish run of simulation %d of %d' % (i + 1, self.num_simulations))
        return self


def consume_tasks(tasks_q):
    log('Start consuming tasks')
    vissim = com.Dispatch('Vissim.Vissim')
    while not tasks_q.empty():
        demand_factor = tasks_q.get()
        work_dir = os.path.join(BASE_DIR, 'normal_scenario_demand_%s' % DEMAND_NAMES[demand_factor],
                                str(random.randint(1, 2 ** 30)))
        shutil.copytree(TEMPLATE_DIR, work_dir)
        NormalScenarioCreator(
            vissim,
            work_dir,
            os.path.join(work_dir, 'normal.inpx'),
            os.path.join(work_dir, 'normal.layx'),
            os.path.join(BASE_DIR, 'od_matrices'),
            demand_factor,
            1,
            1
        ).run_simulation_batch()
        log('Finished task for demand %s' % DEMAND_NAMES[demand_factor])


def run_all_combinations(num_simulations_per_demand):
    def create_tasks_queue():
        log('Start creating tasks')
        tasks = reduce(lambda l1, l2: l1 + l2,
                   map(lambda d: [d] * num_simulations_per_demand[d], num_simulations_per_demand))
        random.shuffle(tasks)
        tasks_queue = Queue()
        map(tasks_queue.put, tasks)
        log('Finished creating tasks')
        return tasks_queue, len(tasks)

    tasks_q, num_tasks = create_tasks_queue()
    workers = [Process(target=consume_tasks, args=(tasks_q,))
               for _ in range(min(NUM_PROCESSES, num_tasks))]
    map(lambda p: p.start(), workers)
    map(lambda p: p.join(), workers)


if __name__ == '__main__':
    run_all_combinations({HIGH_DEMAND_FACTOR: 50, MEDIUM_DEMAND_FACTOR: 50, LOW_DEMAND_FACTOR: 50})

    # NormalScenarioRunner()\
    #     .bootstrap(HIGH_DEMAND_FACTOR, 2, 8)\
    #     .run_simulation_batch()
