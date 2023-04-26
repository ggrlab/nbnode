"""
    Dummy conftest.py for nbnode_pyscaffold.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
# https://stackoverflow.com/questions/17801300/how-to-run-a-method-before-all-tests-in-all-classes

def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    import shutil
    shutil.rmtree("tests_output", ignore_errors=True)
    

def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    

def pytest_unconfigure(config):
    """
    called before test process is exited.
    """