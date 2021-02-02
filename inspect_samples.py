"""Generate some plots of generated dataset to inspect samples."""

import logging
import argparse
from precog.preprocess.plot import InspectSamplePlotter
import utility.arguments as uarguments

DEFAULT_DATA_PATH = '/media/external/data/precog_generate/datasets/20210127'
DEFAULT_SAMPLE_PATTERN = 'map/episode/agent'

def parse_arguments():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--data-path',
        default=DEFAULT_DATA_PATH,
        type=uarguments.dir_path,
        help="")
    argparser.add_argument(
        '--suffix',
        default='json',
        type=str,
        help="extension of sample files to match.")
    argparser.add_argument(
        '--plot-path',
        default='out',
        type=uarguments.dir_path,
        dest='save_plot_path',
        help="path of directory to save splits")
    argparser.add_argument(
        '--pattern',
        default=DEFAULT_SAMPLE_PATTERN,
        dest='sample_pattern',
        type=str,
        help='pattern of sample file paths. Use this to tag samples')
    argparser.add_argument(
        '--filter-paths',
        nargs='+',
        type=uarguments.str_kv,
        action=uarguments.ParseKVToDictAction,
        default={},
        help="Labels to (inclusively) filter sample paths by.")
    return argparser.parse_args()

def main():
    logging.basicConfig(
            format='%(levelname)s: %(message)s',
            level=logging.INFO)
    config = parse_arguments()
    plotter = InspectSamplePlotter(config)
    plotter.plot()

if __name__ == '__main__':
    main()