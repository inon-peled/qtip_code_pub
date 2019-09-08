import multiprocessing
import win32com.client as com
import shutil
import glob
import os


def transfer_all_executed_scenarios_to_one_directory(target_dir, source_glob_masks):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for mask in source_glob_masks:
        for d in glob.glob(mask):
            shutil.copytree(d, os.path.join(target_dir, d.split(os.path.sep)[-1]))


def get_scenario_dirs_with_missing_results(base_dir):
    dirs = filter(lambda d: not os.path.exists(os.path.join(d, 'incident_Link Segment Results_001.att')),
                  glob.glob(os.path.join(base_dir, '*')))
    return (os.path.join(d, 'incident.inpx') for d in dirs)


def open_and_close_vissim_to_save_results(vissim_inpx_path):
    print(vissim_inpx_path)
    try:
        com.Dispatch('Vissim.Vissim-64.10').LoadNet(vissim_inpx_path, False)
    except Exception as e:
        print('ERROR occurred for %s: %s' % (vissim_inpx_path, e))

if __name__ == '__main__':
    dirs = list(get_scenario_dirs_with_missing_results('C:\\Users\\inonpe\\Desktop\\sim_5sec'))
    print(len(dirs))
    multiprocessing.Pool(4).map(open_and_close_vissim_to_save_results, dirs)

# if __name__ == '__main__':
#     transfer_all_executed_scenarios_to_one_directory(
#         'C:\\Users\inonpe\Desktop\sim_5sec', [
#             'C:\\Users\\inonpe\\Desktop\\sim_batches\\batch*\\*\\*\*'])