# LibHINT = False
LibHINT = True

def hint(*args):
    if LibHINT:
        print("\033[91;1m", *args, "\033[0m")

try:
    import cupy as mypy
    hint("[GPU Mode]")
except Exception:
    import numpy as mypy
    hint("[CPU Mode]")

hint("=======        Lib Loading        =======")