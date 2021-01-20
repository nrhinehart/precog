from pathlib import Path
import itertools
import functools
import numpy as np

def save_json(filepath, obj):
    """Serialize object as a JSON formatted stream and save to a file.

    Parameters
    ----------
    filepath : str
        Path of file to save JSON.
    obj : dict
        Object to serialize.
    """
    with open(filepath, 'w') as f:
        json.dump(obj, f)

def load_json(filepath):
    """
    Parameters
    ----------
    
    """
    with open(filepath) as f:
        return json.load(f)

def reduce(f, l, i=None):
    """
    Parameters
    ----------
    f : (function v, acc: f(v, acc))
    l : iterable
    i : any
    """
    return functools.reduce(f, l, i)

def merge_list_of_list(ll):
    """Concatenate iterable of iterables into one list."""
    return list(itertools.chain.from_iterable(ll))

def space_list(l):
    return ' '.join(map(str, l))

def underscore_list(l):
    return space_list(l).replace(' ', '_')

def strip_extension(path):
    """Function to strip file extension

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = Path(path)
    return str(p.with_suffix(''))

def create_sample_pattern(sample_pattern):
    """Given a string of '/' separated words, create a dict of the words and their ordering in the string. Idempotent.

    Parameters
    ----------
    sample_pattern : str or (list of str)
        String of '/' separated words

    Returns
    -------
    dict of str: int
        Empty dict if sample pattern is ''.
        Otherwise each key is a word with value that is the index in the patch ID containing the label corresponding to the word.
    """
    if sample_pattern == '':
        return { }
    else if isinstance(sample_pattern, str):
        sample_pattern = sample_pattern.split('/')
        return {k: i for i,k in enumerate(sample_pattern)}
    else:
        return sample_pattern

def create_sample_id(path, sample_pattern=None, rootpath=None):
    """Create sample ID from path either by
    1) sample_pattern to find the words to use for ID
    2) rootpath to clip the patch path from the left to form patch ID

    Parameters
    ----------
    path : string
        Absolute path to a patch
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch path.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
    rootpath : str
        The root directory path containing sample to clip from sample file path.
        Assumes file path contains rootpath.

    Returns
    -------
    str
        Sample ID generated from path.
    """
    if sample_pattern is not None:
        len_of_patch_id = -(len(sample_pattern) + 1)
        patch_id = strip_extension(path).split('/')[len_of_patch_id:]
        return '/'.join(patch_id)
    elif rootpath is not None:
        return strip_extension(path[len(rootpath):].lstrip('/'))
    else:
        return ValueError("Either sample_pattern or rootpath should be set.")

def create_sample_ids(paths, sample_pattern=None, rootpath=None):
    """Apply create_sample_id() for a list of paths.

    Parameters
    ----------
    path : string
        Absolute path to a patch
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch path.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
    rootpath : str
        The root directory path containing sample to clip from sample file path.
        Assumes file path contains rootpath.

    Returns
    -------
    str
        Sample ID generated from path.
    """
    ids = [None]*len(paths)
    for idx, path in paths:
        ids[idx] = create_sample_id(path,
                sample_pattern=sample_pattern,
                rootpath=rootpath)
    return ids

def label_from_id(sample_id, word, sample_pattern):
    """Get label corresponding to word from sample ID.

    Parameters
    ----------
    sample_id : str
        Sample ID get label from
    word : str
        Word to the label corresponds to.
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch ID.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    Returns
    -------
    int
        Patch size
    """
    return int(patch_id.split('/')[patch_pattern[word]])

def group_ids(ids, words, sample_pattern):
    """Group IDs by in the order of the words in the words array.
    For example if sample_pattern of IDs is 'annotation/subtype/slide/patch_size/magnification' and we have IDs like

    Stroma/MMRd/VOA-1000A/512/20/0_0
    Stroma/MMRd/VOA-1000A/512/10/0_0
    Stroma/MMRd/VOA-1000A/512/20/2_2
    Stroma/MMRd/VOA-1000A/256/20/0_0
    Stroma/MMRd/VOA-1000A/256/10/0_0
    Tumor/POLE/VOA-1000B/256/10/0_0

    Setting words=['patch_size', 'magnification'] gives

    512: {
        20: {
            Stroma/MMRd/VOA-1000A/512/20/0_0
            Stroma/MMRd/VOA-1000A/512/20/2_2
        },
        10: {
            Stroma/MMRd/VOA-1000A/512/10/0_0
        }
    },
    256: {
        20: {
            Stroma/MMRd/VOA-1000A/256/20/0_0
        },
        10: {
            Stroma/MMRd/VOA-1000A/256/10/0_0
            Tumor/POLE/VOA-1000B/256/10/0_0
        }
    }

    Parameters
    ----------
    ids : iterable of str
        List of sample IDs to group.
    words : list of str
        Words to group IDs by. Order of nested labels correspond to order of words array.
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch ID.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
    
    Returns
    -------
    dict
        The grouped IDs
    dict of str: np.array
        Labels corresponding to each word in words array
    """
    id_nd = np.char.asarray([[*id.split('/'), id] for id in ids])
    word_to_labels = { }
    for word in words:
        word_to_labels[word] = np.unique(id_nd[:, sample_pattern[word]])
    id_groups = { }
    def traverse_words(part_id_nd, idx=0):
        if idx >= len(words):
            return part_id_nd[:, -1].tolist()
        else:
            word = words[idx]
            out = { }
            for label in word_to_labels[word]:
                selector = part_id_nd[:, sample_pattern[word]] == label
                out[label] = traverse_words(
                        part_id_nd[selector, :],
                        idx=idx + 1)
            return out
    return traverse_words(id_nd), word_to_labels
    

def index_ids(ids, sample_pattern, include=[], exclude=[]):
    """Index IDs by sample pattern words.
    For example if sample_pattern of IDs is 'annotation/subtype/slide/patch_size/magnification' and we have IDs like

    Stroma/MMRd/VOA-1000A/512/20/0_0
    Stroma/MMRd/VOA-1000A/512/10/0_0
    Stroma/MMRd/VOA-1000A/512/20/2_2
    Stroma/MMRd/VOA-1000A/256/20/0_0
    Stroma/MMRd/VOA-1000A/256/10/0_0
    Tumor/POLE/VOA-1000B/256/10/0_0

    Setting include=['patch_size'] gives

    512/0_0: [
        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
    ],
    512/2_2: [
        Stroma/MMRd/VOA-1000A/512/20/2_2
    ],
    256/0_0: [
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
        Tumor/POLE/VOA-1000B/256/10/0_0
    ]

    So here we create meta IDs of form 'patch_size/patch_id' that sample IDs are grouped into.
    Setting exclude=['patch_size', 'magnification'] gives

    Stroma/MMRd/VOA-1000A/0_0: [
        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
    ],
    Stroma/MMRd/VOA-1000A/2_2: [
        Stroma/MMRd/VOA-1000A/512/20/2_2
    ],
    Tumor/POLE/VOA-1000B: [
        Tumor/POLE/VOA-1000B/256/10/0_0
    ]

    Parameters
    ----------
    patch_ids : list of str

    sample_pattern : dict
        Dictionary describing the directory structure of the patch paths.
        The words are 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'

    include : iterable of str
        The words to group by. By default includes all words.
    
    exclude : iterable of str
        The words to exclude.

    Returns
    -------
    dict of str: list
        The patch IDs grouped by words.
    
    TODO: need to convert to using np.char.asarray()
    """
    id_nd = np.array([[*id.split('/'), id] for id in ids])
    words = set(sample_pattern) - set(exclude)
    if include:
        words = words & set(include)
    indices = sorted([sample_pattern[word] for word in words] + [
            id_nd.shape[1] - 2, id_nd.shape[1] - 1])
    id_nd = id_nd[:,indices]
    id_nd = np.apply_along_axis(lambda r: np.array(['/'.join(r[:-1]), r[-1]]),
            1, id_nd)
    group = { }
    for common_id, id in id_nd:
        if common_id not in group:
            group[common_id] = []
        group[common_id].append(id)
    return group
