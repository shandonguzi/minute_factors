import time
from functools import wraps

def timeit(prefix):
    def decorator(func):
        @wraps(func)
        def robust_func(*args, **kwargs):
            start_time = time.time()
            print(f"\n[+] {prefix} start at {time.strftime('%c')}")
            func(*args,**kwargs)
            print(f"[=] {prefix} finish at {time.strftime('%c')}")
            print(f"[=] Cost {round(time.time() - start_time, 1)} seconds\n")
        return robust_func
    return decorator