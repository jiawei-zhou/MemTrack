import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8,train=False,tracker_cls=None,save_name=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(*dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id,tracker_cls=tracker_cls,save_name=save_name)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus,train=train)


def main():
    tracker_name = 'memtrack_seq'
    tracker_param = 'memtrack_seq_384_full'
    dataset_name = ['satsot','sv248s']
    threads = 0
    num_gpus = 1
    debug = 1 # 1
    tracker_cls = None
    save_name = 'no_heatmap_judg'
    sequence = None # None
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default=tracker_name,help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str,default=tracker_param, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default= dataset_name,nargs='+',help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=sequence, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=debug, help='Debug level.')
    parser.add_argument('--threads', type=int, default=threads, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=num_gpus)
    args = parser.parse_args()
    print(args.dataset_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus,tracker_cls=tracker_cls,save_name=save_name)


if __name__ == '__main__':
    main()
