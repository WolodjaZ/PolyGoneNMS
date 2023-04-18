import os


def read_version():
    version_file_path = os.path.join(os.path.dirname(__file__), "..", "VERSION")
    with open(version_file_path, "r") as version_file:
        return version_file.read().strip()


__version__ = read_version()
