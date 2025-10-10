# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import queue
import shutil
import threading

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", "--router-mode", "round-robin"]

        log_dir = f"{request.node.name}_frontend"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )


def send_completion_request(
    prompt: str, max_tokens: int = 50, timeout: int = 10
) -> requests.Response:
    """Send a completion request to the frontend"""
    payload = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending completion request with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    try:
        response = requests.post(
            f"http://localhost:{FRONTEND_PORT}/v1/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        logger.info(f"Received response with status code: {response.status_code}")
        return response
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed with error: {e}")
        raise


def validate_openai_response(response: requests.Response) -> None:
    """Validate that the response is a proper OpenAI completion response"""
    assert (
        response.status_code == 200
    ), f"Request failed with status {response.status_code}: {response.text}"

    try:
        data = response.json()
    except ValueError:
        pytest.fail(f"Response is not valid JSON: {response.text}")

    # Validate OpenAI completion response structure
    assert "choices" in data, f"Response missing 'choices' field: {data}"
    assert len(data["choices"]) > 0, f"Response has empty 'choices': {data}"
    assert "text" in data["choices"][0], f"Response choice missing 'text' field: {data}"
    assert data["choices"][0]["text"], f"Response text is empty: {data}"

    logger.info(
        f"Received valid completion response: {data['choices'][0]['text'][:100]}..."
    )


def check_worker_received_request(worker_process: ManagedProcess) -> bool:
    """Check if the worker logs contain 'New Request ID:' message indicating it received a request"""
    log_path = worker_process._log_path
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
                return "New Request ID: " in log_content
        except Exception as e:
            logger.warning(f"Could not read worker log file {log_path}: {e}")
    return False


def determine_worker_roles(worker1: ManagedProcess, worker2: ManagedProcess):
    """Determine primary and backup workers based on which worker handled the test request"""
    worker1_received_test = check_worker_received_request(worker1)
    worker2_received_test = check_worker_received_request(worker2)

    if worker1_received_test and not worker2_received_test:
        primary_worker = (worker2, "Worker 2")
        backup_worker = (worker1, "Worker 1")
        logger.info("Test request was handled by Worker 1")
        return primary_worker, backup_worker
    elif worker2_received_test and not worker1_received_test:
        primary_worker = (worker1, "Worker 1")
        backup_worker = (worker2, "Worker 2")
        logger.info("Test request was handled by Worker 2")
        return primary_worker, backup_worker
    else:
        pytest.fail(
            f"Could not determine which worker handled the test request. Worker1: {worker1_received_test}, Worker2: {worker2_received_test}"
        )


def start_completion_request():
    """
    Start a request in a separate thread.

    Returns:
        tuple: (request_thread, response_queue)
    """
    response_queue: queue.Queue[requests.Response] = queue.Queue()

    def send_formal_request():
        response = send_completion_request(
            "Tell me a long long long story about yourself?",
            8000,
            timeout=240,  # Extended timeout for long request
        )
        response_queue.put(response)

    request_thread = threading.Thread(target=send_formal_request)
    request_thread.start()

    return request_thread, response_queue


def validate_completion_response(
    request_thread: threading.Thread, response_queue: queue.Queue
):
    """
    Wait for and validate the completion response after worker failure.

    Args:
        request_thread: The thread running the completion request
        response_queue: Queue containing the response from the request
    """
    request_thread.join(timeout=300)
    if request_thread.is_alive():
        pytest.fail("Request did not complete within timeout")

    # Get the response
    if response_queue.empty():
        pytest.fail("No response received for request")
    response = response_queue.get()

    # Validate the response
    validate_openai_response(response)
    logger.info("✓ Request completed successfully after worker failure")


def verify_migration_occurred(frontend_process: DynamoFrontendProcess) -> None:
    """
    Verify that migration occurred by checking frontend logs for stream disconnection message.

    Args:
        frontend_process: The frontend process to check logs for

    Raises:
        pytest.fail: If migration message is not found in logs
    """
    log_path = frontend_process._log_path
    if not log_path or not os.path.exists(log_path):
        pytest.fail(f"Frontend log file not found at {log_path}")

    try:
        with open(log_path, "r") as f:
            log_content = f.read()
            if "Stream disconnected... recreating stream..." in log_content:
                logger.info(
                    "✓ Migration detected: Found migration message in frontend logs"
                )
                return
            else:
                pytest.fail(
                    "Expected migration did not occur - migration message not found in frontend logs"
                )
    except Exception as e:
        pytest.fail(f"Could not read frontend log file {log_path}: {e}")
