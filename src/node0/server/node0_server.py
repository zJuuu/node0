# This file contains code originally from Hivemind under MIT License
# Original: Copyright 2020 Learning@home authors and collaborators
# Modified by: Pluralis Research 2025
#
# Original code: MIT License (see THIRD_PARTY_LICENSES)
# Modifications: Apache 2.0 License (see LICENSE)
#
# Licensed under the Apache License, Version 2.0 (the "License") for modifications only;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import multiprocessing as mp

from functools import partial
from typing import Type

import torch

from hivemind.compression import CompressionBase, NoCompression
from hivemind.dht import DHT
from hivemind.moe.expert_uid import UID_DELIMITER
from hivemind.moe.server import Server
from hivemind.moe.server.layers import (
    add_custom_models_from_file,
    name_to_block,
    name_to_input,
)
from hivemind.moe.server.server import _generate_uids
from hivemind.optim import Optimizer
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor

from node0.models.arguments import ModelArguments
from node0.models.lr_schedule import schedule_name_to_scheduler
from node0.server.HM_gradient_averager import GradientAveragerFactory
from node0.server.module_collab import ModuleCollab
from node0.utils import MonitorWorker
from node0.utils.common import load_ss_components
from node0.utils.dht_monitor import patch_dht_protocol_logging
from node0.utils.get_parameters import get_parameter_store


logger = get_logger(__name__)


def get_ss_dim(expert_cls: str, model_conf: ModelArguments) -> int:
    """Utility function to get the input dimension to a stage."""
    if "head" in expert_cls:
        input_dim = model_conf.hidden_dim
    else:
        compression_length = int(model_conf.hidden_dim // model_conf.compression_rate)
        input_dim = compression_length + 1  # +1 for tokens
    return input_dim


class Node0Server(Server):
    def __init__(self, optim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        optim._start_monitor()

    @classmethod
    def create(
        cls,
        num_experts: int = 1,
        expert_uids: str = None,
        expert_pattern: str = None,
        expert_cls: str = "lm_body",
        model_conf: ModelArguments = None,
        optim_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        scheduler: str = "none",
        num_warmup_steps: int | None = None,
        num_training_steps: int | None = None,
        clip_grad_norm: float | None = None,
        weight_decay: float | None = None,
        num_stages: int | None = None,
        num_handlers: int | None = None,
        min_batch_size: int = 1,
        max_batch_size: int = 4096,
        averaging_target_batch_size: int = 256,
        reuse_grad_buffers: bool = False,
        use_local_updates: bool = False,
        use_offloading: bool = False,
        matchmaking_time: float = 5.0,
        averaging_timeout: float = 10.0,
        request_timeout: float = 3.0,
        next_chunk_timeout: float = 10.0,
        load_state_timeout: float = 600,
        average_state_every: int = 1,
        sparse_avg: float = 0.0,
        max_allowed_stale: int = 0,
        grad_avg_factory: GradientAveragerFactory | None = None,
        optim_collab_cls: Type[torch.optim.Optimizer] = Optimizer,
        grad_averaging_compression: CompressionBase = NoCompression(),
        load_state_compression: CompressionBase = NoCompression(),
        device: str | None = None,
        initial_peers: list = [],
        compression: CompressionType = CompressionType.NONE,
        stats_report_interval: int = 60,
        custom_module_path: str | None = None,
        update_period: float = 30,
        expiration: float | None = None,
        monitor: MonitorWorker | None = None,
        upload_bw: float | None = None,
        *,
        start: bool,
        **kwargs,
    ) -> Server:
        """Instantiate a server for collaborative optimization.

        Args:
            start (bool): if True, starts the server right away
            num_experts (int, optional): run this many identical experts. Defaults to 1.
            expert_uids (str, optional): spawn experts with these exact uids, overrides num_experts and expert_pattern. Defaults to None.
            expert_pattern (str, optional): a string pattern for experts uids. Defaults to None.
            expert_cls (str, optional): expert type. Defaults to "lm_body".
            model_conf (BaseModel, optional): model config class. Defaults to None.
            optim_cls (Type[torch.optim.Optimizer], optional): optimizer class. Defaults to torch.optim.AdamW.
            scheduler (str, optional): if not `none`, the name of the expert LR scheduler. Defaults to "none".
            num_warmup_steps (int | None, optional): the number of warmup steps for LR scheduler. Defaults to None.
            num_training_steps (int | None, optional): the total number of steps for LR scheduler. Defaults to None.
            clip_grad_norm (float | None, optional): maximum gradient norm used for clipping. Defaults to None.
            num_handlers (int | None, optional): server will use this many parallel processes to handle incoming requests. Defaults to None.
            min_batch_size (int, optional): total num examples in the same batch will be greater than this value. Defaults to 1.
            max_batch_size (int, optional): total num examples in the same batch will not exceed this value. Defaults to 4096.
            averaging_target_batch_size (int): number of examples to accumulate across all peers before averaging. Defaults to 256.
            reuse_grad_buffers (bool, optional): if True, use model's .grad buffers for gradient accumulation. Defaults to False.
            use_local_updates (bool, optional): whether each node performs local weights updates between the allreduce. Defaults to False.
            use_offloading (bool, optional): perform gradient offloading. Defaults to False.
            matchmaking_time (float, optional): time window for nodes to find other nodes for allreduce. Defaults to 5.0.
            averaging_timeout (float, optional): timeout for nodes to perform the allreduce. Defaults to 10.0.
            optim_collab_cls (Type[torch.optim.Optimizer], optional): collaborative optimizer class. Defaults to Optimizer.
            device (str | None, optional): cuda or cpu. Defaults to None.
            initial_peers (list, optional): multiaddrs of one or more active DHT peers. Defaults to [].
            compression (CompressionType, optional): compression type. Defaults to CompressionType.NONE.
            stats_report_interval (int | None, optional): interval between two reports of batch processing performance statistics. Defaults to None.
            custom_module_path (str | None, optional): path of a file with custom nn.modules. Defaults to None.
            update_period (float, optional): server will report experts to DHT once in this many seconds. Defaults to 30.
            expiration (float | None, optional): DHT entries will expire after this many seconds. Defaults to None.
            monitor (MonitorWorker | None, optional): monitor instance. Defaults to None.
            upload_bw (float | None, optional): upload bandwidth. Defaults to None.
            kwargs: any other params will be forwarded to DHT upon creation

        Returns:
            Server: collaborative training server
        """
        # Add custom layers
        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)

        try:
            assert expert_cls in name_to_block
        except:
            logger.error(
                f"Expert class {expert_cls} is not supported. Make sure you provided correct custom_module_path: {custom_module_path}"
            )
            raise

        # Connect to DHT
        _ = patch_dht_protocol_logging()
        dht = DHT(initial_peers=initial_peers, start=True, startup_timeout=30, **kwargs)
        visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
        logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

        # Connect to monitor
        if monitor is not None:
            monitor.connect_dht(dht)

        # Generate uids
        try:
            assert (expert_pattern is None and num_experts is None and expert_uids is not None) or (
                num_experts is not None and expert_uids is None
            )
        except:
            logger.error(
                "Please provide either expert_uids *or* num_experts (possibly with expert_pattern), but not both)"
            )
            raise

        if expert_uids is None:
            expert_uids = []

            uids_to_generate = num_experts - len(expert_uids)
            if uids_to_generate > 0:
                logger.info(f"Generating {uids_to_generate} expert uids from pattern {expert_pattern}")
                expert_uids.extend(_generate_uids(uids_to_generate, expert_pattern, dht))

        # Get parameter store
        stage = expert_uids[0].split(".")[0]
        (
            averaging_target_batch_size,
            scheduler,
            num_warmup_steps,
            num_training_steps,
            averaging_timeout,
            matchmaking_time,
            request_timeout,
            load_state_timeout,
        ) = get_parameter_store(dht, stage)

        num_experts = len(expert_uids)
        num_handlers = num_handlers if num_handlers is not None else num_experts * 8
        device = device or (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Scheduler
        scheduler_cls = schedule_name_to_scheduler[scheduler]
        if scheduler_cls is not None:
            scheduler_cls = partial(
                scheduler_cls, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

        # Initialize experts
        input_dim = model_conf.hidden_dim if not model_conf.use_compression else get_ss_dim(expert_cls, model_conf)
        sequence_length = model_conf.max_seq_len
        sample_input = name_to_input[expert_cls](DUMMY_BATCH_SIZE, sequence_length, input_dim)
        if isinstance(sample_input, tuple):
            args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
        else:
            args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)

        experts = {}
        for expert_uid in expert_uids:
            expert = name_to_block[expert_cls](model_conf)
            if monitor:
                active_period_timeout = (
                    max(load_state_timeout, averaging_timeout + matchmaking_time + request_timeout) + 10
                )
                monitor.monitor_callback(expert, model_conf, active_period_timeout)

            assert averaging_target_batch_size is not None

            if model_conf.use_compression:
                exclude_params = ["rcv", "fixed_tok_embeddings"]
            else:
                exclude_params = []
            if weight_decay:
                no_decay = ["tok_embeddings.weight"]
                params = [
                    {
                        "params": [
                            p
                            for n, p in expert.named_parameters()
                            if not any(nd in n for nd in no_decay) and not any(ex in n for ex in exclude_params)
                        ],
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in expert.named_parameters()
                            if any(nd in n for nd in no_decay) and not any(ex in n for ex in exclude_params)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                params = [p for n, p in expert.named_parameters() if not any(ex in n for ex in exclude_params)]

            if model_conf.use_compression:
                ss_comps = load_ss_components(model_conf.ss_component)
                expert.load_comp(ss_comps)
                logger.info("Succeded loading remote subspace components")
                expert.ss_regularize()

            optimizer_lock = mp.Lock()

            backend = ModuleCollab(
                optimizer_lock=optimizer_lock,
                name=expert_uid,
                module=expert,
                args_schema=args_schema,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )

            backend.module.to(device)

            optim_collab = optim_collab_cls(
                model=expert,
                optimizer_lock=optimizer_lock,
                sparse_avg=sparse_avg,
                max_allowed_stale=max_allowed_stale,
                optimizer=optim_cls,
                params=params,
                dht=dht,
                run_id=expert_uid.split(UID_DELIMITER)[0],
                scheduler=scheduler_cls,
                target_batch_size=averaging_target_batch_size,
                matchmaking_time=matchmaking_time,
                averaging_timeout=averaging_timeout,
                load_state_timeout=load_state_timeout,
                average_state_every=average_state_every,
                grad_averager_factory=grad_avg_factory,
                grad_compression=grad_averaging_compression,
                reuse_grad_buffers=reuse_grad_buffers,
                use_local_updates=use_local_updates,
                offload_optimizer=use_offloading,
                delay_state_averaging=False,
                next_chunk_timeout=next_chunk_timeout,
                verbose=True,
                averager_opts={"bandwidth": upload_bw, "request_timeout": request_timeout},
            )
            optim_collab.load_state_from_peers(wait_for_end_round=True)
            backend.optimizer = optim_collab
            experts[expert_uid] = backend

        return cls(
            optim_collab,
            dht,
            experts,
            num_connection_handlers=num_handlers,
            device=device,
            checkpoint_dir=None,
            stats_report_interval=stats_report_interval,
            update_period=update_period,
            expiration=expiration,
            start=start,
        )
