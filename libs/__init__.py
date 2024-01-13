# LibHINT = False
LibHINT = True

def hint(*args):
    if LibHINT:
        print("\033[91;1m", *args, "\033[0m")

hint("=======        Lib Loading        =======")