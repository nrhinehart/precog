"""Generate validation split datasets for CARLA."""

import logging
import argparse
import precog.preprocess as preprocess
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
    argparser.add_argument(
        '--split-path',
        default=DEFAULT_SPLIT_PATH,
        type=uarguments.dir_path,
        dest='save_split_path',
        help="path of directory to save splits")
    argparser.add_argument(
        '--pattern',
        default=DEFAULT_SAMPLE_PATTERN,
        dest='sample_pattern',
        type=str,
        help='pattern of sample file paths. Use this to tag samples')
    argparser.add_argument(
        '--filter_labels',
        nargs='+',
        type=uarguments.str_kv,
        action=uarguments.ParseKVToDictAction,
        default={},
        help="filter samples by specific tag.")
    return argparser.parse_args()

def try_create_groups(config):
    group_creator = preprocess.SampleGroupCreator(config)

    print(f"found {len(group_creator.sample_ids)} samples.")
    # debug constructor information. Looks good as of 20210120
    print(group_creator.word_to_labels)
    print(group_creator.mapped_ids.keys())
    print("mapped samples to map and episode")
    for k1, val1 in group_creator.mapped_ids.items():
        count = 0
        for k2, val2 in val1.items():
            print(k1, k2, len(val2))
            count += len(val2)
        print(k1, count)

    # debug generate groups. Looks good as of 20210120
    groups = group_creator.generate_groups()
    count = 0
    print("generated groups")
    for idx, group in groups.items():
        print(f"group {idx} has", len(group), "samples")
        count += len(group)
    print(count)

def main():
    logging.basicConfig(
            format='%(levelname)s: %(message)s',
            level=logging.INFO)
    config = parse_arguments()
    group_creator = preprocess.SampleGroupCreator(config)

    logging.info(f"found {len(group_creator.sample_ids)} samples.")
    groups = group_creator.generate_groups()
    for idx, group in groups.items():
        logging.info(f"group {idx} has {len(group)} samples")
    group_creator.generate_cross_validation_splits(groups)

if __name__ == '__main__':
    main()