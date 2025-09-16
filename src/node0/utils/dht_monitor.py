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

import threading
import time

from hivemind.dht.protocol import DHTProtocol
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)

_rpc_store_count = 0
_rpc_find_count = 0
_last_log_time = time.time()
_counter_lock = threading.Lock()


def patch_dht_protocol_logging():
    """Monkey patch DHTProtocol to add RPC call counting"""
    # Import here to avoid issues with module loading order
    from hivemind.p2p import P2PContext
    from hivemind.proto import dht_pb2

    global _rpc_store_count, _rpc_find_count, _last_log_time

    # Store original methods
    original_rpc_store = DHTProtocol.rpc_store
    original_rpc_find = DHTProtocol.rpc_find

    logger.extra("[DHT Monitor] Patching DHTProtocol to monitor RPC calls")

    # Create wrapped methods
    async def counted_rpc_store(self, request: dht_pb2.StoreRequest, context: P2PContext) -> dht_pb2.StoreResponse:
        global _rpc_store_count, _rpc_find_count, _last_log_time

        with _counter_lock:
            _rpc_store_count += 1
            current_time = time.time()

            # Log every 60 seconds
            if current_time - _last_log_time >= 60:
                logger.extra(f"[DHT RPC Stats] Last 60s - rpc_store: {_rpc_store_count}, rpc_find: {_rpc_find_count}")
                _rpc_store_count = 0
                _rpc_find_count = 0
                _last_log_time = current_time

        return await original_rpc_store(self, request, context)

    async def counted_rpc_find(self, request: dht_pb2.FindRequest, context: P2PContext) -> dht_pb2.FindResponse:
        global _rpc_store_count, _rpc_find_count, _last_log_time

        with _counter_lock:
            _rpc_find_count += 1
            current_time = time.time()

            # Log every 60 seconds
            if current_time - _last_log_time >= 60:
                logger.extra(f"[DHT RPC Stats] Last 60s - rpc_store: {_rpc_store_count}, rpc_find: {_rpc_find_count}")
                _rpc_store_count = 0
                _rpc_find_count = 0
                _last_log_time = current_time

        return await original_rpc_find(self, request, context)

    # Copy over important attributes from original methods
    counted_rpc_store.__name__ = "rpc_store"
    counted_rpc_store.__qualname__ = "DHTProtocol.rpc_store"
    counted_rpc_find.__name__ = "rpc_find"
    counted_rpc_find.__qualname__ = "DHTProtocol.rpc_find"

    # Patch with the new methods
    DHTProtocol.rpc_store = counted_rpc_store
    DHTProtocol.rpc_find = counted_rpc_find

    logger.extra("[DHT Monitor] RPC monitoring active - will log stats every 60 seconds")

    return None
