import shutil
import gzip
import os
import pickle
from functools import wraps
from multiprocessing import Lock
from random import randint


MAXIMUM_KEY_LENGTH = 10 * 1024 ** 2
CACHE_DIRECTORY = os.path.join('data', 'cache')
INDEX_FILE = os.path.join(CACHE_DIRECTORY, 'index')
LOCK = Lock()


def __generate_file_path(f):
    return os.path.join(CACHE_DIRECTORY, f.__name__ + '.' + str(randint(10 ** 12, 2 * 10 ** 12)) + '.pkl')


def __read_cache_index_under_lock():
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'wb') as fw_index:
            pickle.dump({}, fw_index)
    with open(INDEX_FILE, 'rb') as fr_index:
        return pickle.load(fr_index)


def __make_key(f, args, kwargs):
    key = ''.join([
        '[FUNC %s]' % f.__name__,
        '[ARGS %s]' % ','.join([str(a) for a in args]),
        '[KWARGS %s]' % ','.join([str(k) for k in kwargs.items()])
    ])
    if len(key) > MAXIMUM_KEY_LENGTH:
        raise ValueError('Too long cache key (%d > %d)' % (len(key), MAXIMUM_KEY_LENGTH))
    else:
        return key


def __add_to_cache(f, args, kwargs, df):
    df_file = __generate_file_path(f)
    with open(df_file, 'wb') as fw_df:
        pickle.dump(df, fw_df)
    with LOCK:
        index_dict = __read_cache_index_under_lock()
        index_dict[__make_key(f, args, kwargs)] = df_file
        tmp_index_path = os.path.join(CACHE_DIRECTORY, 'index_tmp_%s' % randint(10 ** 12, 2 * 10 ** 12))
        with open(tmp_index_path, 'wb') as fp_index:
            pickle.dump(index_dict, fp_index)
        shutil.move(tmp_index_path, INDEX_FILE)
    return df_file


def __find_cached(f, args, kwargs):
    with LOCK:
        index_dict = __read_cache_index_under_lock()
    if __make_key(f, args, kwargs) in index_dict:
        cached_file_path = index_dict[__make_key(f, args, kwargs)]
        file_path_maybe_gzipped = cached_file_path if os.path.exists(cached_file_path) else (cached_file_path + '.gz')
        if os.path.exists(file_path_maybe_gzipped):
            with (gzip.open(file_path_maybe_gzipped) if file_path_maybe_gzipped.endswith('.gz')
                  else open(file_path_maybe_gzipped, 'rb')) as fr_df:
                return pickle.load(fr_df)


def cache(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        cached = __find_cached(f, args, kwargs)
        if cached is not None:
            return cached
        df = f(*args, **kwargs)
        __add_to_cache(f, args, kwargs, df)
        return df
    return wrapped


def cache_method(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        args_for_caching = args[0].cache_key() + args[1:]
        cached = __find_cached(f, args_for_caching, kwargs)
        if cached is not None:
            return cached
        df = f(*args, **kwargs)
        __add_to_cache(f, args_for_caching, kwargs, df)
        return df
    return wrapped
