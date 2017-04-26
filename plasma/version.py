import re

__version__ = "0.1.0"

version = __version__

version_info = tuple(re.split(r"[-\.]", __version__))

specification = ".".join(version_info[:2])

def compatible(serializedVersion):
    selfMajor, selfMinor = map(int, version_info[:2])
    otherMajor, otherMinor = map(int, re.split(r"[-\.]", serializedVersion)[:2])
    if selfMajor >= otherMajor:
        return True
    elif selfMinor >= otherMinor:
        return True
    else:
        return False
