import traceback
from functools import wraps


def multiprocessed(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            ret = f(*args, **kwargs)
            return ret
        except Exception as exc:
            print 'args = %s, kwargs = %s' % (args, kwargs)
            traceback.print_exc()
            raise exc
    return wrapped
