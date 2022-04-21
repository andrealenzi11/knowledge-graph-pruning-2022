import os
from typing import Union


def print_and_write(out_file, text: Union[str, float, int, bool, None, list, dict, set, tuple]):
    print(text)
    out_file.write(str(text) + os.linesep)
    out_file.flush()
