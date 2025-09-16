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

import asyncio
import re
import threading
import time

from typing import List, Optional, Tuple

import numpy as np

from hivemind.averaging.control import StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHT
from hivemind.moe.expert_uid import ExpertUID
from hivemind.p2p import PeerID
from hivemind.utils import DHTExpiration, TimedStorage, ValueWithExpiration, get_dht_time, get_logger


GroupKey = str
Endpoint = str
GROUP_PATTERN = re.compile("^(([^.])+)[.]0b[01]*$")  # e.g. bert_exp4_averaging.0b01001101
logger = get_logger(__name__)


def is_valid_group(maybe_group: str) -> bool:
    """A group identifier must contain group type, followed by one or more .-separated indices, and any ?metadata"""
    return bool(GROUP_PATTERN.fullmatch(maybe_group))


class GroupKeyManager:
    """
    Simplified utility class that manages a single fixed group for all nodes
    """

    def __init__(
        self,
        dht: DHT,
        prefix: str,
        fixed_group_key: Optional[str] = None,
    ):
        self.dht = dht
        self.prefix = prefix
        self.peer_id = dht.peer_id

        # Use a fixed group key - either provided or default to "global"
        if fixed_group_key:
            if not is_valid_group(fixed_group_key):
                raise ValueError(f"Invalid fixed group key: {fixed_group_key}")
            self._fixed_key = fixed_group_key
        else:
            # Default fixed group key with empty bits (all nodes use same group)
            self._fixed_key = f"{self.prefix}.0b"
        self.group_bits = ""

    @property
    def current_key(self) -> GroupKey:
        """Return the fixed group key that all nodes use"""
        return self._fixed_key

    async def declare_averager(self, peer_id: PeerID, expiration_time: float, looking_for_group: bool = True) -> bool:
        """
        Add (or remove) the averager to the fixed group

        :param peer_id: averager public peer_id for incoming requests
        :param expiration_time: intent to run allreduce before this timestamp
        :param looking_for_group: by default (True), declare the averager as "looking for group";
          If False, mark that the averager is no longer looking for group
        :return: True if declared, False if declaration was rejected by DHT peers
        """
        expiration_time = expiration_time if looking_for_group else float(np.nextafter(expiration_time, float("inf")))
        return await self.dht.store(
            key=self._fixed_key,
            subkey=peer_id.to_bytes(),
            value=looking_for_group,
            expiration_time=expiration_time,
            return_future=True,
        )

    async def get_averagers(self, only_active: bool = True) -> List[Tuple[PeerID, DHTExpiration]]:
        """
        Find and return averagers in the fixed group

        :param only_active: if True, return only active averagers that are looking for group
            if False, return all averagers in the group regardless of status
        :return: peer_ids and expirations of every matching averager
        """
        result = await self.dht.get(self._fixed_key, latest=True, return_future=True)
        if result is None or not isinstance(result.value, dict):
            logger.debug(f"Fixed group not found: {self._fixed_key}, creating new group")
            return []

        averagers = []
        for key, looking_for_group in result.value.items():
            try:
                if only_active and not looking_for_group.value:
                    continue
                averagers.append((PeerID(key), looking_for_group.expiration_time))
            except Exception as e:
                logger.warning(f"Could not parse peer key {key} ({looking_for_group}, exc={e})")
        return averagers

    async def join_group(self, expiration_time: float) -> bool:
        """
        Join the fixed group (convenience method)

        :param expiration_time: when this declaration expires
        :return: True if successfully joined
        """
        return await self.declare_averager(self.peer_id, expiration_time, looking_for_group=True)

    async def leave_group(self, expiration_time: float) -> bool:
        """
        Leave the fixed group (convenience method)

        :param expiration_time: original expiration time used when joining
        :return: True if successfully left
        """
        return await self.declare_averager(self.peer_id, expiration_time, looking_for_group=False)


class Matchmaking:
    """
    Simplified matchmaking that works with a single fixed group

    :type dht: an instance of hivemind.DHT. Server will use DHT for all network interactions.
    :param prefix: Prefix of the stage. i.e head, body0, tail.
    :param request_timeout: This timeout is backward compatible with HM Matchmaking. It is used to cancel matchmaking in case of no responses.
    :param min_matchmaking_time: How long before matchmaking operation is cancelled
    :param check_interval: How often to poll the DHT to check for new peers in the group
    :param update_period: How often to update the peer table with all peers in the stage
    """

    def __init__(
        self,
        dht: DHT,
        prefix: str,
        request_timeout: float = 5.0,
        min_matchmaking_time: float = 10.0,
        check_interval: float = 1.0,
        update_period: float = 3.0,
    ):
        self.group_key_manager = GroupKeyManager(dht, prefix)

        self.dht = dht
        self.prefix = prefix
        self.peer_id = self.group_key_manager.peer_id
        self.request_timeout = request_timeout
        self.min_matchmaking_time = min_matchmaking_time
        self.check_interval = check_interval

        # Parameters for update peer table
        self.update_period = update_period
        self.peer_table = TimedStorage[ExpertUID, PeerID]()
        self.is_alive = threading.Event()
        self.is_alive.set()
        self.update_trigger, self.update_finished = threading.Event(), threading.Event()
        self.update_period, self.last_update = update_period, get_dht_time()
        self.update_thread = threading.Thread(target=self.update_peers_in_background, daemon=True)
        self.update_thread.start()

    @property
    def max_peers(self):
        return len(self.peer_table)

    @property
    def peer_set(self):
        res = set()
        for index, expert_info in self.peer_table.items():
            res.add(expert_info.value._bytes)
        return res

    def update_peers_in_background(self):
        while self.is_alive.is_set():
            time_to_next_update = max(0.0, self.last_update + self.update_period - get_dht_time())
            try:
                self.update_trigger.wait(timeout=time_to_next_update)
                # update triggered by main thread
            except TimeoutError:
                pass  # update triggered by refresh_period

            self.update_trigger.clear()
            response = self.dht.get(self.prefix.split("_")[0] + ".0.", latest=True)
            if isinstance(response, ValueWithExpiration) and isinstance(response.value, dict):
                for index, expert_info in response.value.items():
                    try:
                        (uid, endpoint), expiration_time = expert_info
                        self._add_peer(uid, endpoint, expiration_time)
                    except Exception as e:
                        logger.warning(f"Skipping malformed peer info {expert_info} (exc={e})")
            else:
                logger.warning(
                    f"Could not refresh peer, dht info key contains {response}, will retry in {time_to_next_update}s"
                )
            self.last_update = get_dht_time()
            self.update_finished.set()

    def _add_peer(self, uid: ExpertUID, endpoint: Endpoint, expiration_time: DHTExpiration):
        self.peer_table.store(uid, PeerID(endpoint), expiration_time)
        logger.debug(f"Storing peer: {uid}, expiration time = {expiration_time:.3f}.")

    async def look_for_group(self, step: StepControl) -> Optional[GroupInfo]:
        """
        Look for peers in the fixed group and form a group if enough peers are available

        :param step: To get the step schedule time
        :return: GroupInfo if group formed successfully, None otherwise
        """
        timeout = self.min_matchmaking_time

        new_expiration_time = float(get_dht_time() + timeout)
        await self.group_key_manager.join_group(new_expiration_time)

        # Wait and retry logic
        start_time = time.time()

        # Accumulate all peers that issue join_group. Wait to match peer_table
        all_peerIds = set()
        while time.time() - start_time < timeout:
            # Get all active averagers in the group
            averagers = await self.group_key_manager.get_averagers(only_active=True)

            _peerIds = {peer_id.to_bytes() for peer_id, _ in averagers}
            all_peerIds = all_peerIds.union(_peerIds)

            # We have enough peers, proceed with group formation
            if (len(all_peerIds) == self.max_peers) and (self.max_peers > 0):
                break

            # Wait for either the peer_table to populate or to find all peers in the table
            logger.debug(f"Not enough peers yet: {len(all_peerIds)} < {self.max_peers}, waiting...")
            await asyncio.sleep(self.check_interval)

        if len(all_peerIds) == 0:
            # Timeout reached without finding enough peers
            logger.info(f"Timeout: Not any peers in group")
            return None

        # Create group info with all available peers
        all_peerIds = sorted(list(all_peerIds))
        peer_ids = [PeerID(peer_id) for peer_id in all_peerIds]

        # Create a deterministic group ID based on sorted peer IDs
        sorted_peer_ids = sorted([str(pid) for pid in peer_ids])
        group_id = b"O[\x9aU\xcf%\xf0(\x90Nq\xdf!\x8b\x85)&\x0c\xe9r"
        gathered = tuple(step.data_for_gather for peer_id in sorted_peer_ids)

        group_info = GroupInfo(group_id=group_id, peer_ids=tuple(peer_ids), gathered=gathered)

        end_time = time.time()
        logger.extra(
            f"Formed group with {len(peer_ids)} peers out of {self.max_peers} in {end_time - start_time:.3f} secs"
        )
        return group_info

    async def leave_group(self, expiration_time: float) -> bool:
        """
        Leave the current group

        :param expiration_time: original expiration time
        :return: True if successfully left
        """
        return await self.group_key_manager.leave_group(expiration_time)


class MatchmakingException(Exception):
    """An internal exception that marks undesired edge cases during averaging"""
