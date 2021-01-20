"""Generate validation split datasets for CARLA."""

import argparse
import utility.arguments as uarguments

DEFAULT_DATA_PATH = '/media/external/data/precog_generate/datasets/20210113'
DEFAULT_SPLIT_PATH = '/media/external/data/precog_generate/splits'
DEFAULT_SAMPLE_PATTERN = 'map/episode'

def parse_arguments():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--n-groups',
        default=12,
        type=int,
        help="Number of groups to generate.")
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
    argparse.add_argument(
        '--split-path',
        default=DEFAULT_SPLIT_PATH,
        type=uarguments.dir_path,
        dest='save_split_path',
        help="path of directory to save splits")
    argparse.add_argument(
        '--pattern',
        default=DEFAULT_SAMPLE_PATTERN,
        dest='sample_pattern',
        type=str,
        help='pattern of sample file paths. Use this to tag samples')
    argparse.add_argument(
        '--filter_labels',
        nargs='+',
        type=uarguments.str_kv,
        action=uarguments.ParseKVToDictAction,
        default={},
        help="filter samples by specific tag.")
    return argparser.parse_args()

def main():
    config = parse_arguments()
    group_creator = SampleGroupCreator(config)
    groups = group_creator.generate_groups()
    for idx, group in groups.items():
        print(f"{idx} has", len(group), "groups")

if __name__ == '__main__':
    main()