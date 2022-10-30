import os
import time
import logging
import uuid

import torch

from ts.metrics.metrics_store import MetricsStore


def start_torchserve(
    ncs=False,
    model_store="model_store",
    workflow_store="",
    models="",
    config_file="",
    log_file="",
    wait_for=15,
):
    """Initialize a local torchserve endpoint"""
    logging.info("## Starting TorchServe")
    cmd = f"torchserve --start --model-store={model_store}"
    if models:
        cmd += f" --models={models}"
    if workflow_store:
        cmd += f" --workflow-store={workflow_store}"
    if ncs:
        cmd += " --ncs"
    if config_file:
        cmd += f" --ts-config={config_file}"
    if log_file:
        logging.info(f"## Console logs redirected to file: {log_file}")
        cmd += f" >> {log_file}"
    logging.info(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    status = os.system(cmd)
    if status == 0:
        logging.info("## Successfully started TorchServe")
        time.sleep(wait_for)
        return True
    else:
        logging.info("## TorchServe failed to start !")
        return False


def stop_torchserve(wait_for=10):
    """Stop any torchserve endpoint in local"""
    logging.info("## Stopping TorchServe")
    cmd = f"torchserve --stop"
    logging.info(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
    status = os.system(cmd)
    if status == 0:
        logging.info("## Successfully stopped TorchServe")
        time.sleep(wait_for)
        return True
    else:
        logging.info("## TorchServe failed to stop !")
        return False


class MockContext:
    """
    Mock class to replicate the context passed into model initialize
    """

    def __init__(
        self,
        model_name,
        model_type: str = None,
        model_dir: str = None,
    ):
        self.manifest = {"model": {}}
        self.manifest["model"]["modelName"] = model_name
        self.manifest["model"]["modelType"] = model_type
        self.manifest["model"]["modelDir"] = model_dir
        self.metrics = MetricsStore(uuid.uuid4(), model_name)

        self.explain = False

    def get_request_header(self, idx, exp):
        if idx and exp:
            if self.explain:
                return True
        return False
