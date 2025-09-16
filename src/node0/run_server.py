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

import argparse
import json
import multiprocessing as mp
import os
import platform

from functools import partial
from pathlib import Path

import torch
import yaml

from hivemind.compression import NoCompression, ScaledFloat16Compression
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger

from node0.security.authorization import authorize_with_pluralis
from node0.security.validation import make_validators
from node0.server.HM_gradient_averager import GradientAverager
from node0.server.node0_server import Node0Server
from node0.server.optim import AutoStepOptimizer
from node0.utils import (
    MonitorWorker,
    Node0Logger,
    build_cls,
    clean_tmp,
    infer_expert_params,
    update_initial_peers,
)
from node0.utils.mem_monitor import MemoryTracker
from node0.utils.network_throughput import NetworkMonitor
from node0.utils.node_info import get_node_info


logger = get_logger(__name__)

if platform.system().lower() == "darwin":
    # Necessary for forks to work properly on macOS, see https://github.com/kevlened/pytest-parallel/issues/93
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    mp.set_start_method("fork", force=True)


def parse_args():
    parser = argparse.ArgumentParser()
    # Connection arguments
    parser.add_argument(
        "--host_maddrs",
        type=str,
        help="Multiaddrs to listen for external connections from other p2p instances",
    )
    parser.add_argument(
        "--announce_maddrs",
        type=str,
        help="Visible multiaddrs the host announces for external connections from other p2p instances",
    )
    parser.add_argument(
        "--initial_peers",
        type=str,
        nargs="*",
        required=True,
        default=[],
        help="Multiaddrs of one or more active DHT peers",
    )

    # Experiment arguments
    parser.add_argument("--run_config", type=str, required=True, help="Run configuration file.")
    parser.add_argument(
        "--custom_module_path",
        type=str,
        help="Path of a file with custom nn.modules, wrapped into special decorator",
    )
    parser.add_argument(
        "--num_handlers",
        type=int,
        default=5,
        help="Server will use this many processes to handle incoming requests",
    )

    # Authorization parameters
    parser.add_argument(
        "--identity_path",
        type=str,
        default="private.key",
        help="Path to identity file to be used in P2P",
    )
    parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--email", type=str, default="", help="Email address")
    parser.add_argument("--auth_server", type=str, required=True, help="Authentication server URL")

    args = parser.parse_args()
    args = vars(args)

    # Read run config file
    with open(args.pop("run_config")) as f:
        run_config = yaml.safe_load(f)

    # Combine arguments
    run_config.update({k: v for k, v in args.items() if v is not None})
    return run_config


def main():
    if not torch.backends.openmp.is_available():
        # Necessary to prevent the server from freezing after forks
        torch.set_num_threads(1)

    # Parse arguments
    args = parse_args()

    # Set up logging
    log_level = args.pop("log_level", "info").upper()
    node0_logger = Node0Logger(log_level=log_level)

    # Logging and monitoring
    save_clean_logs = args.pop("save_clean_logs", False)
    terminate_AR_fail = args.pop("terminate_AR_fail", True)
    experiment_prefix = args.pop("experiment_prefix", "pluralis")
    monitor = MonitorWorker(
        stats_report_interval=args["stats_report_interval"] if "stats_report_interval" in args else 60,
        experiment_prefix=experiment_prefix,
        log_file="logs/server.log",
        save_clean_logs=save_clean_logs,
        terminate_AR_fail=terminate_AR_fail,
    )
    node0_logger.add_monitor_handler(monitor.queue_handler)
    monitor.start()

    logger.info(f"Running with configuration: {json.dumps(args, indent=4)}")

    # Check pytorch version
    if not torch.__version__.startswith("2.7"):
        logger.error("Wrong pytorch version. Please install torch 2.7")
        exit(1)

    # Clean tmp folder
    clean_tmp()

    # Collect information about the node
    node_info = get_node_info()

    # Authorize
    check_integrity = args.pop("check_integrity", True)
    authorizer = authorize_with_pluralis(
        node_info=node_info,
        user_token=args.pop("token"),
        user_email=args.pop("email"),
        role="worker",
        auth_server=args.pop("auth_server"),
        identity_path=args["identity_path"],
        current_path=Path(__file__).resolve().parent,
        announce_maddrs=args["announce_maddrs"],
        host_port=int(args["host_maddrs"].split("/")[4]),
        check_integrity=check_integrity,
    )
    pipeline_stage = str(authorizer.pipeline_stage)
    args["host_maddrs"] = [args["host_maddrs"]]
    args["announce_maddrs"] = [args["announce_maddrs"]]

    expert_params = infer_expert_params(pipeline_stage)
    stage_name = expert_params.pop("stage_name")
    args.update(expert_params)

    # Add validators
    validators, public_key = make_validators(
        experiment_prefix=experiment_prefix,
        peer_id=authorizer.peer_id,
        stage=stage_name,
    )

    # Add expert parameters to args
    num_worker_dhts = args.pop("num_dht") - 1
    args["initial_peers"] = update_initial_peers(
        args["initial_peers"], pipeline_stage, args["num_stages"], num_dht=num_worker_dhts
    )

    # Add authorization details to log monitor
    monitor.add_auth_info(
        authorizer=authorizer,
        peer_id=authorizer.peer_id,
        stage=stage_name,
        local_public_key=public_key,
    )

    # Set BW for the peer
    default_bandwidth = args.pop("bandwidth", 20)
    if node_info.upload_speed:
        bandwidth = node_info.upload_speed
    else:
        bandwidth = default_bandwidth

    # Build model arguments
    model_config = args.pop("model_config")
    model_args = build_cls(model_config["class_name"], model_config["init_args"])
    model_args.stage = args.pop("stage")

    # Build optimizer
    optim_config = args.pop("optim_config")
    optim_cls = build_cls(optim_config["class_name"], optim_config["init_args"], partial_init=True)

    # Build gradient averager
    grad_avg_config = args.pop("grad_avg_config", None)
    is_tail = "tail" in expert_params["expert_pattern"]
    if "request_timeout" in args:
        request_timeout = args["request_timeout"]
    else:
        request_timeout = 3.0
    if (
        grad_avg_config
        and ("averager_rank" in grad_avg_config["init_args"])
        and grad_avg_config["init_args"]["averager_rank"] > 0
        and not is_tail
    ):
        # If detected to be mac, replace psgd
        if platform.system().lower() == "darwin":
            grad_avg_config["class_name"] = "node0.server.power_sgd_averager_mac.PowerSGDGradientAverager"
        grad_avg_config["init_args"]["request_timeout"] = request_timeout
        grad_avg_factory = build_cls(grad_avg_config["class_name"], grad_avg_config["init_args"], partial_init=True)
    elif (
        grad_avg_config
        and ("grad_compression_factor" in grad_avg_config["init_args"])
        and grad_avg_config["init_args"]["grad_compression_factor"] > 0
        and not is_tail
    ):
        grad_avg_config["init_args"]["request_timeout"] = request_timeout
        grad_avg_factory = build_cls(grad_avg_config["class_name"], grad_avg_config["init_args"], partial_init=True)
    else:
        grad_avg_factory = partial(GradientAverager, bandwidth=bandwidth, request_timeout=request_timeout)

    # Optimizer compressions
    grad_averaging_compression = args.pop("grad_averaging_compression", "NoCompression")
    if grad_averaging_compression == "NoCompression":
        grad_averaging_compression = NoCompression()
    elif grad_averaging_compression == "Float16Compression":
        grad_averaging_compression = ScaledFloat16Compression()
    else:
        raise ValueError("grad_averaging_compression must be NoCompression or Float16Compression")

    load_state_compression = args.pop("load_state_compression", "NoCompression")
    if load_state_compression == "NoCompression":
        load_state_compression = NoCompression()
    elif load_state_compression == "Float16Compression":
        load_state_compression = ScaledFloat16Compression()
    else:
        raise ValueError("load_state_compression must be NoCompression or Float16Compression")

    # Compression
    compression_type = args.pop("compression")
    compression = getattr(CompressionType, compression_type)

    # Select device
    if node_info.device_name == "cpu":
        device = "cpu"
    elif node_info.device_name.startswith("mps"):
        device = "mps"
    else:
        device = "cuda"

    logger.info(f"Using {device} device")

    # Log machine usage
    log_machine_usage = args.pop("log_machine_usage", False)
    if log_machine_usage:
        NetworkMonitor()
        MemoryTracker()

    # Start server
    server = Node0Server.create(
        model_conf=model_args,
        optim_cls=optim_cls,
        grad_avg_factory=grad_avg_factory,
        optim_collab_cls=AutoStepOptimizer,
        grad_averaging_compression=grad_averaging_compression,
        load_state_compression=load_state_compression,
        start=True,
        compression=compression,
        record_validators=validators,
        authorizer=authorizer,
        monitor=monitor,
        upload_bw=bandwidth,
        device=device,
        **args,
    )

    try:
        server.join()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")


if __name__ == "__main__":
    main()
