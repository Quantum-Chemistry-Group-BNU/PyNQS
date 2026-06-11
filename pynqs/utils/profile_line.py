import sys
from functools import wraps
from contextlib import contextmanager
from line_profiler import LineProfiler


class LineProfileContext:
    def __init__(self, *functions, enable=True):
        self.enable = enable
        if self.enable:
            self.profiler = LineProfiler()
            unique_funcs = set()
            for func in functions:
                if func not in unique_funcs:
                    unique_funcs.add(func)
                    self.profiler.add_function(func)

    def __enter__(self):
        if self.enable:
            self.profiler.enable_by_count()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.profiler.disable_by_count()
            self.profiler.print_stats(sys.stdout, output_unit=1e-6)
            sys.stdout.flush()
        return False


def line_profile(func=None, enable=True):
    if func is None:
        return lambda f: line_profile(f, enable=enable)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not enable:
            return func(*args, **kwargs)

        with LineProfileContext(func, enable=enable):
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def block_profile(*functions, enable=True):
    ctx = LineProfileContext(*functions, enable=enable)
    try:
        with ctx:
            yield ctx
    finally:
        pass
