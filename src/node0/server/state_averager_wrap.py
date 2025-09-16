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
import math
import os
import random
import signal
import time

from typing import Any, Optional

import torch

from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.compression import CompressionInfo, deserialize_torch_tensor
from hivemind.p2p import PeerID
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger
from hivemind.utils.asyncio import aiter_with_timeout, enter_asynchronously
from hivemind.utils.streaming import combine_from_streaming
from hivemind.utils.tensor_descr import TensorDescriptor
from hivemind.utils.timed_storage import ValueWithExpiration

from node0.server.HM_state_averager import (
    TrainingStateAverager as HivemindTrainingStateAverager,
)
from node0.server.matchmaking import MatchmakingException


GatheredData = Any
logger = get_logger(__name__)


class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    def get_indices(self, param):
        return torch.ones(param.shape).bool()


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p, param):
        super().__init__(p)
        self.state[param] = {}
        self._set_partition(param)

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["num_partitions"] = min(math.ceil(1 / self.p), param.numel())
        param_state["partitions"] = (
            torch.rand(param.numel(), device=param.device).argsort().view(param.shape) % param_state["num_partitions"]
        )

    def get_indices(self, param, curr_partition):
        curr_partition = curr_partition % self.state[param]["num_partitions"]
        indices = (self.state[param]["partitions"] == curr_partition).bool()

        return indices


class TrainingStateAverager(HivemindTrainingStateAverager):
    """
    A class that extends Hivemind TrainingStateAverager and prevents too many
    consecutive calls to load_state_from_peers.
    """

    def __init__(
        self,
        sparse_avg=0.0,
        average_state_every=1,
        call_limit: int = 1,
        *args,
        **kwargs,
    ):
        kwargs["start"] = False
        super().__init__(*args, **kwargs)

        self.zclip_warmup = None
        if hasattr(self, "zclip"):
            self.zclip_warmup = self.zclip.warmup_steps

        self._call_limit = call_limit
        self._consecutive_fails = 0
        self._request_timeout = kwargs["request_timeout"]
        self.sparse_avg = sparse_avg
        self.average_state_every = average_state_every
        self.partition_selector = []

    def set_sparta_partitions(self):
        with self.get_tensors() as local_tensors:
            torch.manual_seed(1337)
            for i, p in enumerate(local_tensors):
                self.partition_selector.append(PartitionedIndexSelector(self.sparse_avg, p))

    async def _load_state_from_peers(self, future: MPFuture, timeout: Optional[float] = None):
        # Adapted from /hivemind/averaging/averager.py
        if timeout is not None:
            timeout = self.next_chunk_timeout if self.next_chunk_timeout is not None else self.request_timeout
        try:
            key_manager = self._matchmaking.group_key_manager
            peer_priority, _ = self.dht.get(f"{key_manager.prefix}.all_averagers", latest=True) or ({}, None)
            peer_priority = {
                PeerID(peer_id): (float(info.value), random.random())  # using randomness as a tie breaker
                for peer_id, info in peer_priority.items()
                if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))
            }

            if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
                logger.info(f"Averager could not load state from peers: peer dict empty or corrupted {peer_priority}")
                future.set_result(None)
                return

            metadata = None
            for peer in sorted(peer_priority.keys(), key=peer_priority.get, reverse=True):
                if peer != self.peer_id:
                    t0 = time.monotonic()
                    logger.info(f"Downloading parameters from peer {peer}")
                    try:
                        stub = self.get_stub(self._p2p, peer, namespace=self.prefix)
                        stream = await stub.rpc_download_state(averaging_pb2.DownloadRequest())
                        current_tensor_parts, tensors = [], []

                        async for message in aiter_with_timeout(stream, timeout=timeout):
                            if message.metadata:
                                metadata = self.serializer.loads(message.metadata)
                            if message.tensor_part.dtype and current_tensor_parts:
                                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                                tensor = deserialize_torch_tensor(combine_from_streaming(current_tensor_parts))
                                tensors.append(tensor)
                                current_tensor_parts = []
                            current_tensor_parts.append(message.tensor_part)

                        if current_tensor_parts:
                            tensor = deserialize_torch_tensor(combine_from_streaming(current_tensor_parts))
                            tensors.append(tensor)

                        if not metadata:
                            logger.debug(f"Peer {peer} did not send its state")
                            continue

                        t1 = time.monotonic()
                        logger.info(f"Finished downloading state in {t1 - t0:.3f}s from {peer}")
                        self._consecutive_fails = 0
                        # Check if any gradient contains NaN or inf values
                        has_nans = any(not torch.isfinite(t).all() for t in tensors)
                        if has_nans:
                            logger.error(f"Failed to load state from peer.")
                            logger.error(f"Downloaded state contains invalid values. Exiting the run.")
                            os.killpg(os.getpgrp(), signal.SIGTERM)
                        future.set_result((metadata, tensors))
                        return
                    except Exception as e:
                        self._consecutive_fails = self._consecutive_fails + 1

                        if isinstance(e, TimeoutError) and self._consecutive_fails < self._call_limit:
                            logger.info(
                                f"{self._consecutive_fails}/{self._call_limit} load state timeout before ending session."
                            )

                        if isinstance(e, TimeoutError) and self._consecutive_fails >= self._call_limit:
                            logger.error(
                                f"Failed to load state from peers. "
                                "Too many TimeoutErrors were caught when trying to _load_state_from_peers. "
                                "This problem may occur due to slow internet connection, or temporary overload of the peer-to-peer network. Exiting run."
                            )
                            os.killpg(os.getpgrp(), signal.SIGTERM)
                        else:
                            logger.error(
                                f"Failed to load state from {peer} - {repr(e)}. Exiting run.",
                                exc_info=logger.getEffectiveLevel() <= 15,
                            )
                            os.killpg(os.getpgrp(), signal.SIGTERM)

        finally:
            if not future.done():
                future.set_result(None)

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run sparse aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths;
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            epoch = self.global_epoch
            snapped_epoch = int(round(epoch / self.average_state_every))
            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                if self.sparse_avg > 0:
                    tensor_idxs = []  # index of tensor in local_tensors
                    sparse_idxs = []  # the sparse indices that index into the averaged tensor
                    sparse_tensor = []  # the sparse tensor to be averaged

                    tensor_infos = kwargs["tensor_infos"]
                    for i, val in enumerate(zip(local_tensors, tensor_infos)):
                        p, ti = val
                        should_include = False

                        # Determine if tensor should be included
                        if i < len(self.main_parameters):
                            if self.main_parameters[i].requires_grad:
                                should_include = True
                        else:
                            should_include = True
                            if self.zclip_warmup:
                                # Start averaging zclip (mean,var) after 2*zclip_warmup to account for
                                # peers that joined between step 0 and zclip warmup
                                epoch_round = int(snapped_epoch * self.average_state_every)
                                zclip_in_warmup = epoch_round <= 2 * self.zclip_warmup
                                is_zclip = ti.key == "zclip_mean" or ti.key == "zclip_var"
                                if zclip_in_warmup and is_zclip:
                                    should_include = False

                        # Only process tensors that should be included
                        if should_include:
                            tensor_idxs.append(i)
                            _idx = self.partition_selector[i].get_indices(p, snapped_epoch)
                            sparse_tensor.append(p[_idx].contiguous())
                            sparse_idxs.append(_idx)

                    # Build tensor info using proper indexing
                    tensor_infos_sparse = []
                    for sparse_idx, tensor_idx in enumerate(tensor_idxs):
                        desc = TensorDescriptor(
                            size=sparse_tensor[sparse_idx].shape,
                            dtype=sparse_tensor[sparse_idx].dtype,
                            device=sparse_tensor[sparse_idx].device,
                            requires_grad=local_tensors[tensor_idx].requires_grad,
                        )
                        tensor_infos_sparse.append(CompressionInfo(key=sparse_idx, descriptor=desc))

                    tensor_infos_sparse = tuple(tensor_infos_sparse)
                    kwargs["tensor_infos"] = tensor_infos_sparse
                    await self._run_allreduce_inplace_(
                        sparse_tensor, group_info, peer_fractions=peer_fractions, **kwargs
                    )

                    # Copy results back using proper indexing
                    for sparse_idx, tensor_idx in enumerate(tensor_idxs):
                        local_tensors[tensor_idx][sparse_idxs[sparse_idx]] = sparse_tensor[sparse_idx]
                else:
                    await self._run_allreduce_inplace_(
                        local_tensors, group_info, peer_fractions=peer_fractions, **kwargs
                    )
                return user_gathered
        except BaseException as e:
            if isinstance(e, Exception):
                logger.error(e, exc_info=logger.getEffectiveLevel() <= 15)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")
