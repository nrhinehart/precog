
import git
import inspect
import functools
import logging
import os
import time
import pdb

log = logging.getLogger(__file__)

def get_sha_and_dirty(dir_path=os.path.dirname(os.path.realpath(__file__))):
    repo = git.Repo(dir_path, search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha, repo.is_dirty()

def log_wrap(wrapped_func, logfun, argprint):
    """Simple decorator to log the start and end of a function.

    :param wrapped_func: 
    :returns: 
    :rtype: 

    """
    assert(logfun in ('info', 'debug', 'error', 'warning'))
    names, varargs, keywords, defaults = inspect.getargspec(wrapped_func)
    # sig = inspect.signature(wrapped_func)
    log = logging.getLogger(wrapped_func.__module__)    

    @functools.wraps(wrapped_func)
    def log_wrapper(*args, **kwargs):
        
        if not argprint:
            getattr(log, logfun)("Starting {}".format(wrapped_func.__qualname__))
        else:
            if names[0] == 'self':
                names_fix = names[1:]
            else:
                names_fix = names

            argstr = ', '.join(['{}={}'.format(n, a) for n, a in zip(names_fix, args)])
            
            getattr(log, logfun)("Starting {}({})".format(wrapped_func.__qualname__, argstr))

        start = time.time()
            
        # Call the function, print later.
        ret = wrapped_func(*args, **kwargs)

        end = time.time()

        elapsed = "{:.2f}".format(end - start)
        
        if not argprint:
            getattr(log, logfun)("Finished {} in {}s.".format(wrapped_func.__qualname__, elapsed))
        return ret
    return log_wrapper

def log_wrapd(argprint=False):
    return functools.partial(log_wrap, logfun='debug', argprint=argprint)

def log_wrapi(argprint=False):
    return functools.partial(log_wrap, logfun='info', argprint=argprint)

def log_wrapw(wf, argprint=False):
    return functools.partial(log_wrap, logfun='warning', argprint=argprint)

def log_wrape(wf, argprint=False):
    return functools.partial(log_wrap, logfun='error', argprint=argprint)

def query_purge_directory(directory):
    import logging
    import os
    import shutil
    import time
    directory = os.path.realpath(directory)
    
    def query():
        return input('\n***Should we remove "{}"\n(AKA: "{}") (y/n)***?'.format(directory, os.path.basename(directory))).lower()
    answer = None
    while answer not in ('y', 'n'):
        answer = query()
    if answer == 'y'and os.path.exists(directory):
        print("Removing '{}'".format(directory))
        # Close the loggers.
        logging.shutdown()
        try:
            shutil.rmtree(directory)
        except OSError:
            try:
                logging.shutdown()
                time.sleep(1)
                shutil.rmtree(directory)
            except OSError as e:
                print(e)
                print("Can't fully remove directory: '{}'".format(directory))
                print("Is the directory on NFS, and another process is still accessing it (e.g. tensorboard)?")
    else:
        log.info("Not removing '{}'".format(directory))
