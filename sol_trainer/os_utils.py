import shutil
import os


def untuple(a):
    if len(a) == 1:
        a = a[0]
    return a


def path_join(a, *p):
    a = untuple(a)
    return os.path.join(a, *p)


def makedirs(a):
    a = untuple(a)
    return os.mkdir(a)


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
