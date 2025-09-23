import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import os
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

def analysis(name=None,parameter_name=None):
    trackers = []

    if name is not None:
        name = name
    else:
        name = 'histrack_seq'
    if parameter_name is not None:
        parameter_name = parameter_name
    else:
        parameter_name = 'histrack_seq_384_full_no_heatmap_judg'
    dataset_name = 'satsot'

    """ckpt test"""
    # path = '/home/zhoujiawei/my_model/MemTrack/output/test/tracking_results/histrack_seq'
    # parameter_names = sorted([i for i in os.listdir(path) if 'tp_num' in i and 'memory5' not in i])
    # for parameter_name in parameter_names:
    #     trackers.extend(trackerlist(name=name, parameter_name=parameter_name, dataset_name=dataset_name,
    #                             run_ids=None, display_name=parameter_name))
    #dataset_name = 'lasot'
    """stark"""
    # trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-S50'))
    # trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-ST50'))
    # trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-ST101'))
    """TransT"""
    # trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
    #                             run_ids=None, display_name='TransT_N2', result_only=True))
    # trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
    #                             run_ids=None, display_name='TransT_N4', result_only=True))
    """pytracking"""
    # trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
    # trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
    # trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
    # trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
    # trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
    """ostrack"""
    trackers.extend(trackerlist(name=name, parameter_name=parameter_name, dataset_name=dataset_name,
                                run_ids=None, display_name=parameter_name))

    dataset = get_dataset(dataset_name)
    # dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
    # plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
    #              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
    # print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
if __name__ == '__main__':
    analysis()
