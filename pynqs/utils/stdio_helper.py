from __future__ import annotations

import sys

from contextlib import contextmanager
from functools import wraps
from pathlib import Path


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_stdio_file(
    save_name: str, source_suffix: str = ".py", result_suffix: str = ".integral"
) -> tuple[str, str]:
    source_file = save_name + source_suffix
    result_file = save_name + result_suffix
    return source_file, result_file


@contextmanager
def save_stdio_context(
    save_name: str,
    source_file: str | None = None,
    source_in_result: bool = True,
    source_position: str = "begin",
    source_suffix: str = ".py",
    result_suffix: str = ".integral",
):
    _, save_result_file = get_stdio_file(save_name, source_suffix, result_suffix)

    source_text = None
    if source_file is not None:
        source_text = Path(source_file).read_text(encoding="utf-8")

    stdout0 = sys.stdout
    stderr0 = sys.stderr
    result_fp = open(save_result_file, "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(stdout0, result_fp)
    sys.stderr = TeeStream(stderr0, result_fp)

    try:
        if source_text is not None and source_in_result and source_position == "begin":
            source_name = Path(source_file).name
            print(f"===== source begin: {source_name} =====")
            print(source_text, end="" if source_text.endswith("\n") else "\n")
            print(f"===== source end: {source_name} =====")
        yield save_result_file
    finally:
        if source_text is not None and source_in_result and source_position == "end":
            source_name = Path(source_file).name
            print(f"===== source begin: {source_name} =====")
            print(source_text, end="" if source_text.endswith("\n") else "\n")
            print(f"===== source end: {source_name} =====")
        sys.stdout = stdout0
        sys.stderr = stderr0
        result_fp.close()


def save_stdio(
    save_name: str,
    source_file: str | None = None,
    source_in_result: bool = True,
    source_position: str = "begin",
    source_suffix: str = ".py",
    result_suffix: str = ".integral",
):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with save_stdio_context(
                save_name=save_name,
                source_file=source_file,
                source_in_result=source_in_result,
                source_position=source_position,
                source_suffix=source_suffix,
                result_suffix=result_suffix,
            ):
                return func(*args, **kwargs)

        return wrapper

    return deco
