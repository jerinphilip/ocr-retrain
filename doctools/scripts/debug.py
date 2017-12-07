from functools import wraps
from time import time as _timenow 
from sys import stderr

def Popen(argv):
     t = lambda x: x
     setattr(t, "wait", lambda : None)
     print(' '.join(argv))
     return t

def time(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        start = _timenow()
        result = f(*args, **kwargs)
        end = _timenow()
        print('[time] {}: {}'.format(f.__name__, end-start),
                file=stderr)
        return result
    return _wrapped

# if __name__ == '__main__':
#     @time
#     def f(x, y):
#         return x + y

# f(3, 2)