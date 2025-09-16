# Copyright 2025 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import signal

from typing import Any

from hivemind.dht import DHT
from hivemind.utils import get_logger
from hivemind.utils.timed_storage import ValueWithExpiration

from node0.security.validation import RunParameters


logger = get_logger(__name__)


def get_parameter_store(dht: DHT, prefix: str) -> tuple[Any, ...]:
    """
    Retrieve the most recent run parameters from the distributed hash table (DHT).

    This function fetches run parameters stored by peers in the DHT, validates them
    using RSA signature verification, and returns the averaging target batch size
    from the most recently updated entry.

    Args:
        dht (DHT): The distributed hash table instance to retrieve parameters from.
        prefix (str): Prefix string used to construct the DHT storage key for lookup.

    Returns:
        int: The averaging target batch size from the most recent valid peer entry.
    """

    param_store_key = f"{prefix}_paramstore"
    param_store_result = dht.get(param_store_key, latest=True)

    if not isinstance(param_store_result, ValueWithExpiration):
        logger.error("Could not retrieve run parameters from peers. Exiting run.")
        os.killpg(os.getpgrp(), signal.SIGTERM)
        raise RuntimeError("Could not retrieve run parameters from peers")

    metadata = param_store_result.value

    valid_peer_entries = [
        RunParameters.parse_obj(peer_value.value) for peer_value in metadata.values() if peer_value.value is not None
    ]

    last_time = -float("inf")
    averaging_target_batch_size = 0
    scheduler = ""
    num_warmup_steps = 0
    num_training_steps = 0
    averaging_timeout = 0
    matchmaking_time = 0
    request_timeout = 0
    load_state_timeout = 0

    for val in valid_peer_entries:
        if val.time > last_time:
            averaging_target_batch_size = val.averaging_target_batch_size
            scheduler = val.scheduler
            num_warmup_steps = val.num_warmup_steps
            num_training_steps = val.num_training_steps
            averaging_timeout = val.averaging_timeout
            matchmaking_time = val.matchmaking_time
            request_timeout = val.request_timeout
            load_state_timeout = val.load_state_timeout
            last_time = val.time

    if (
        averaging_target_batch_size <= 0
        or scheduler not in ["linear", "cosine"]
        or num_warmup_steps <= 0
        or num_training_steps <= 0
        or averaging_timeout <= 0
        or matchmaking_time <= 0
        or request_timeout <= 0
        or load_state_timeout <= 0
    ):
        logger.error("Could not retrieve run parameters from peers. Exiting run.")
        os.killpg(os.getpgrp(), signal.SIGTERM)
        return

    logger.info(
        f"Got runtime training parameters: "
        f"averaging_target_batch_size = {averaging_target_batch_size}, "
        f"scheduler = {scheduler}, "
        f"num_warmup_steps = {num_warmup_steps}, "
        f"num_training_steps = {num_training_steps}, "
        f"averaging_timeout = {averaging_timeout}, "
        f"matchmaking_time = {matchmaking_time}, "
        f"request_timeout = {request_timeout}, "
        f"load_state_timeout = {load_state_timeout}"
    )
    return (
        averaging_target_batch_size,
        scheduler,
        num_warmup_steps,
        num_training_steps,
        averaging_timeout,
        matchmaking_time,
        request_timeout,
        load_state_timeout,
    )
