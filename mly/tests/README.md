# MLY test suite

This test suite is structure for pytest to run.
Each module under the `/mly/` directory should have its own
test module in here, named as `test_<module>.py`, e.g. for
the `mly.tools` module, we should end up with a corresponding
`mly.tests.test_tools` module (`/mly/tests/test_tools.py`) that
contains the tests.

Within each module should be test functions, or classes containing
test methods, each following the standard [good practices for test
discovery](https://docs.pytest.org/en/stable/goodpractices.html).

For gitlab users, the test harness in the gitlab CI will take
care of running the test suite over each proposed change.
