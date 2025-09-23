from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/zhoujiawei/clean_code/MemTrack/got10k_lmdb'
    settings.got10k_path = '/home/zhoujiawei/clean_code/MemTrack/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/zhoujiawei/clean_code/MemTrack/itb'
    settings.lasot_extension_subset_path_path = '/home/zhoujiawei/clean_code/MemTrack/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/zhoujiawei/clean_code/MemTrack/lasot_lmdb'
    settings.lasot_path = '/home/zhoujiawei/clean_code/MemTrack/lasot'
    settings.network_path = '/home/zhoujiawei/clean_code/MemTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/zhoujiawei/clean_code/MemTrack/nfs'
    settings.ootb_path = '/home/zhoujiawei/satellite_video_datasets/OOTB'
    settings.otb_path = '/home/zhoujiawei/clean_code/MemTrack/otb'
    settings.prj_dir = '/home/zhoujiawei/clean_code/MemTrack'
    settings.result_plot_path = '/home/zhoujiawei/clean_code/MemTrack/output/test/result_plots'
    settings.results_path = '/home/zhoujiawei/clean_code/MemTrack/output/test/tracking_results'    # Where to store tracking results
    settings.satsot_path = '/home/zhoujiawei/satellite_video_datasets/SatSOT'
    settings.save_dir = '/home/zhoujiawei/clean_code/MemTrack/output'
    settings.segmentation_path = '/home/zhoujiawei/clean_code/MemTrack/output/test/segmentation_results'
    settings.sv248s_path = '/home/zhoujiawei/satellite_video_datasets/SV248S'
    settings.tc128_path = '/home/zhoujiawei/clean_code/MemTrack/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/zhoujiawei/clean_code/MemTrack/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/zhoujiawei/clean_code/MemTrack/trackingnet'
    settings.uav_path = '/home/zhoujiawei/clean_code/MemTrack/uav'
    settings.viso_path = '/home/zhoujiawei/satellite_video_datasets/VISO/SOT/sot'
    settings.vot18_path = '/home/zhoujiawei/clean_code/MemTrack/vot2018'
    settings.vot22_path = '/home/zhoujiawei/clean_code/MemTrack/vot2022'
    settings.vot_path = '/home/zhoujiawei/clean_code/MemTrack/VOT2019'
    settings.youtubevos_dir = ''

    return settings

