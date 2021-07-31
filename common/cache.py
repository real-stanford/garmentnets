import pathlib
import hashlib
import pickle

def file_attr_cache(target_file, cache_dir='~/local/.cache/file_attr_cache'):
    cache_dir_path = pathlib.Path(cache_dir).expanduser()
    target_file_path = pathlib.Path(target_file).expanduser()
    assert(target_file_path.exists())
    target_key = hashlib.md5(
        str(target_file_path.absolute()).encode()).hexdigest()
    def decorator(func):
        def wrapped(*args, **kwargs):
            if not cache_dir_path.exists():
                cache_dir_path.mkdir(parents=True, exist_ok=True)
            else:
                assert(cache_dir_path.is_dir())
            cache_file_path = cache_dir_path.joinpath(target_key)
            if cache_file_path.exists():
                target_time = target_file_path.stat().st_mtime
                cache_time = cache_file_path.stat().st_mtime
                if target_time < cache_time:
                    # exist and older than target
                    obj = pickle.load(cache_file_path.open('rb'))
                    return obj
            
            # run function
            obj = func(*args, **kwargs)
            pickle.dump(obj, cache_file_path.open('wb'))
            return obj
        return wrapped
    return decorator
