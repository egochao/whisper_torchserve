import pytest
from tests.utils import start_torchserve, stop_torchserve


def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False, help="run tests connect with external resource"
    )


def pytest_collection_modifyitems(config, items):
    """Modify default config from pytest."""
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def start2serve():
    status = start_torchserve(
        ncs=True,
        model_store="model_store", 
        models="asr_model.mar", 
        config_file="model_store/torchserve_config.properties",
        wait_for=25)

    yield status
    stop_torchserve()
