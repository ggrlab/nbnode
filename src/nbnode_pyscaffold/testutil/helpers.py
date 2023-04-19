import os
import inspect  # to find the caller filename, see find_dirname_above_currentfile()


def find_tests_directory(
    dirpath: str, max_iterations: int = 30, verbose: bool = False
) -> str:
    """
    You want probably rather use :func:`find_dirname_above` as it can do the same, just better.

    In many tests I need the tests/ directory first because the test might be dependent on data.

    Args:
        dirpath:
            The directory where you start from searching
        max_iterations:
            Maximum number of upwards iterations until stop
        verbose:


    Returns:
        The tests/ directory relative to the given dirpath.

    """
    TESTS_DIR = dirpath
    i = 0
    if verbose:
        print(TESTS_DIR)
    while not (
        os.path.realpath(TESTS_DIR).endswith("\\tests")
        or os.path.realpath(TESTS_DIR).endswith("/tests")
    ):
        TESTS_DIR = os.path.join(TESTS_DIR, os.pardir)
        if verbose:
            print(TESTS_DIR, os.path.realpath(TESTS_DIR))
        i += 1
        if i > max_iterations:
            raise ValueError("There seems to be an error finding the test directory.")
    return TESTS_DIR


def find_dirname_above(
    dirpath: str,
    dirname: str = "tests",
    max_iterations: int = 30,
    verbose: bool = False,
    realpath: bool = True,
) -> str:
    """
    Goes directory-wise upwards starting at dirpath and
    0) current_directory = dirpath
    1) list content of current_directory and search for 'dirname'.
    2)  If found
            Return the path
        If not found
            current_directory = os.path.join(current_directory, '..')  # Ascend one directory, go to 1)

    Args:
        dirpath:
            current path
        dirname:
            what to search for
        max_iterations:
            Maximum number of going upwards in the directory-hierarchy (with os.pardir)
        verbose:
            To be or not to be verbose
        realpath:
            If true, return the os.path.realpath() of the returned path

    Returns:
        The first found path to the directory 'dirname'

    """

    TESTS_DIR = dirpath
    i = 0
    if verbose:
        print(TESTS_DIR)

    while not dirname in os.listdir(TESTS_DIR):
        TESTS_DIR = os.path.join(TESTS_DIR, os.pardir)
        if verbose:
            print(TESTS_DIR)
        i += 1
        if i > max_iterations:
            raise ValueError("There seems to be an error finding the test directory.")
    TESTS_DIR = os.path.join(TESTS_DIR, dirname)
    if realpath:
        TESTS_DIR = os.path.realpath(TESTS_DIR)
    if verbose:
        print(TESTS_DIR)

    return TESTS_DIR


def find_dirname_above_currentfile(
    dirname: str = "tests",
    max_iterations: int = 30,
    verbose: bool = False,
    realpath: bool = True,
) -> str:
    """
    Finds the given dirname above the file which called this function.

    Args:
        dirname:
            See find_dirname_above
        max_iterations:
            See find_dirname_above
        verbose:
            See find_dirname_above
        realpath:
            See find_dirname_above

    Returns:
        The first found path to the directory 'dirname'
    """
    # The following two lines extract the filename which called this function
    frame = inspect.stack()[1]
    filename = frame[0].f_code.co_filename

    return find_dirname_above(
        os.path.dirname(filename),
        dirname=dirname,
        max_iterations=max_iterations,
        verbose=verbose,
        realpath=realpath,
    )
