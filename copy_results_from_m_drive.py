import glob
import os
import shutil


LOCAL_INCIDENT_SCENARIOS_DIR = os.path.join('copy_of_m_drive', 'sim_5sec')
RESULTS_ATT_ORIGINAL_BASENAME = 'incident_Link Segment Results_001.att'
RESULTS_ATT_NEW_BASENAME = 'results.att'


def copy_and_rename_atts():
    for src in (glob.glob(os.path.join('M:', 'backup', 'qtip_data', 'VISSIM_files', 'inon', 'sim_5sec', '*', RESULTS_ATT_ORIGINAL_BASENAME))):
        dst_dir = os.path.join(LOCAL_INCIDENT_SCENARIOS_DIR, os.path.dirname(src).split(os.sep)[-1])
        os.makedirs(dst_dir)
        shutil.copyfile(src, os.path.join(dst_dir, RESULTS_ATT_NEW_BASENAME))


def rename_atts_after_manual_copying(incident_scenarios_dir):
    for att in glob.glob(os.path.join(incident_scenarios_dir, '*', RESULTS_ATT_ORIGINAL_BASENAME)):
        os.rename(att, os.path.join(os.path.dirname(att), RESULTS_ATT_NEW_BASENAME))


def remove_incident_scenarios_without_results(incident_scenarios_dir):
    for dir in glob.glob(os.path.join(incident_scenarios_dir, '*')):
        if not os.path.exists(os.path.join(dir, RESULTS_ATT_NEW_BASENAME)):
            print('Deleting directory without results:', dir)
            shutil.rmtree(dir)


if __name__ == '__main__':
    # copy_and_rename_atts()
    # rename_atts_after_manual_copying(os.path.join('copy_of_m_drive', 'sim_5sec/'))
    remove_incident_scenarios_without_results(os.path.join('copy_of_m_drive', 'sim_5sec/'))
    pass
