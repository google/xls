# Bazel runfiles to access data from the package
from rules_python.python.runfiles import runfiles


_RES = runfiles.Create()


def _get_path(file_name: str) -> str:
    p = _RES.Rlocation(
        "com_google_xls/xls/modules/dbe/scripts/dbe_data/data/" + file_name)
    if not p:
        raise FileNotFoundError(
            f"Data file {file_name!r} is not known to Bazel runfiles")
    return p


class DataFiles:
    DICKENS_TXT = _get_path("dickens.txt")
