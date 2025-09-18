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

import multiprocessing as mp
import os
import signal
import threading
import time

from typing import Callable, Optional

import torch

from hivemind import Optimizer
from hivemind.optim.grad_scaler import GradScaler
from hivemind.utils import get_dht_time, get_logger

from node0.server.HM_gradient_averager import GradientAverager, GradientAveragerFactory
from node0.server.state_averager_wrap import TrainingStateAverager


logger = get_logger(__name__)


class AutoStepOptimizer(Optimizer):
    """
    A class that extends Hivemind Optimizer and ensures step() is called at least once per auto_step_time.
    If step() hasn't been called externally within the auto_step_time window, it will be called automatically.
    """

    def __init__(
        self,
        model,
        optimizer_lock,
        sparse_avg: float = 0.0,
        auto_step_time: float = 3.0,
        max_allowed_stale: int = 0,
        grad_schedule_buffer: float = 5.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._auto_step_time = auto_step_time
        self.model = model
        self.max_allowed_stale = max_allowed_stale
        self.grad_schedule_buffer = grad_schedule_buffer
        self.optimizer_lock = optimizer_lock
        self._last_step_time: float = time.time()
        self._step_lock = mp.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()
        self.in_update = False

        # Set state avg parameters
        self.state_averager.average_state_every = self.average_state_every
        self.state_averager.sparse_avg = sparse_avg
        if sparse_avg:
            self.state_averager.set_sparta_partitions()
        self.state_averager.run_in_background(await_ready=True)

    def _resync_state(self):
        if self._should_load_state_from_peers():
            logger.log(self.status_loglevel, "Peer is out of sync")
            self.load_state_from_peers()
            return True  # local gradients were computed with out-of-sync parameters, must start over
        elif self._catchup_epoch():
            with self.tracker.pause_updates():
                logger.log(self.status_loglevel, f"Catching up with collaboration step {self.tracker.global_epoch}")
                self.state_averager.local_epoch = self.tracker.global_epoch
                self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)
        return False

    def _should_load_state_from_peers(self) -> bool:
        return self.local_epoch < (self.tracker.global_epoch - self.max_allowed_stale)

    def _catchup_epoch(self) -> bool:
        return (self.tracker.global_epoch - self.max_allowed_stale) <= self.local_epoch < self.tracker.global_epoch

    def _make_state_averager(self, **kwargs) -> TrainingStateAverager:
        return TrainingStateAverager(
            dht=self.dht,
            prefix=f"{self.run_id}_state_averager",
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            offload_optimizer=self.offload_optimizer,
            custom_gradients=self.offload_optimizer,
            status_loglevel=self.status_loglevel,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            **kwargs,
        )

    def _make_gradient_averager(self, factory: Optional[GradientAveragerFactory], **kwargs) -> GradientAverager:
        assert hasattr(self, "state_averager"), "must initialize state averager first"
        factory = factory if factory is not None else GradientAverager
        grad_averager = factory(
            dht=self.dht,
            prefix=f"{self.run_id}_grad_averager",
            parameters=self.state_averager.main_parameters,
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            **kwargs,
        )
        if self.offload_optimizer:
            optimized_param_groups = self.state_averager.optimizer.param_groups
            optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
            with grad_averager.get_tensors() as averaged_gradients:
                assert len(averaged_gradients) == len(optimized_parameters)
                for opt_param, averaged_grad in zip(optimized_parameters, averaged_gradients):
                    opt_param.grad = averaged_grad
        return grad_averager

    def _start_monitor(self) -> None:
        """Start the monitoring thread if it's not already running."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._should_stop.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_step, daemon=True)
            self._monitor_thread.start()

    def _monitor_step(self) -> None:
        """Monitor thread that checks and calls step() if it wasn't called within auto_step_time window."""
        while not self._should_stop.is_set():
            time.sleep(1.0)  # Check every 1s

            current_time = time.time()
            time_since_last_step = current_time - self._last_step_time

            if time_since_last_step >= self._auto_step_time:
                with self._step_lock:
                    # Check again after acquiring lock in case step() was called
                    if time.time() - self._last_step_time >= self._auto_step_time:
                        self._auto_step()

    def _auto_step(self) -> None:
        """Internal method to call step() automatically."""
        try:
            # Call the parent class's step method with one batch size
            logger.debug(f"AutoStepOptimizer step at {time.strftime('%H:%M:%S')}")
            self._last_step_time = time.time()
            # self._check_update_version()

            if self._resync_state():
                return None

            self._maybe_schedule_gradient_averaging()

            if self.in_update and self.tracker.ready_to_update_epoch:
                batch_size = 1
                self._step(batch_size=batch_size)

            self.state_averager.allow_state_sharing = True
        except Exception as e:
            logger.error(f"Error in auto step: {e}")

    def step(self, batch_size: Optional[int] = None) -> None:
        """
        Override of the step method that updates the last step time.
        This should be called by external code.
        """
        with self._step_lock:
            # self._check_update_version()

            if self._resync_state():
                return None

            self._step(batch_size=batch_size)
            self.state_averager.allow_state_sharing = True

            self._last_step_time = time.time()

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self._should_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1)
            self._monitor_thread = None

    def __del__(self):
        """Ensure the monitoring thread is stopped when the object is destroyed."""
        self.stop_monitoring()

    @property
    def ready_to_update_epoch(self) -> bool:
        """Whether or not this peer can increment epoch right away."""
        return (
            self.tracker.global_epoch > self.tracker.local_progress.epoch
            or self.tracker.global_progress.samples_accumulated >= self.tracker.target_batch_size
        )

    def _check_and_accumulate_gradients(self, batch_size: int) -> bool:
        """Check if gradients are valid, accumulate and return True; otherwise, reset and return False"""
        if self.grad_averager.has_nan_grads():
            self.tracker.report_local_progress(self.local_epoch, samples_accumulated=0)
            logger.error("Encountered incorrect value in grads, exiting run")
            os.killpg(os.getpgrp(), signal.SIGTERM)

        self.grad_averager.accumulate_grads_(batch_size)
        self.tracker.report_local_progress(self.local_epoch, self.grad_averager.local_samples_accumulated)
        return True

    def _maybe_schedule_gradient_averaging(self) -> None:
        """If next epoch is coming soon, schedule the next gradient averaging round at the estimated end of epoch"""
        assert self.use_gradient_averaging
        if not self.in_update and self.ready_to_update_epoch:
            if self.scheduled_grads is None or self.scheduled_grads.triggered or self.scheduled_grads.done():
                self.in_update = True
                eta_seconds = self.tracker.estimated_next_update_time - get_dht_time()
                logger.log(self.status_loglevel, f"Pre-scheduling gradient averaging round in {eta_seconds:.2f} sec")
                self.scheduled_grads = self.grad_averager.schedule_step(timeout=self.averaging_timeout)

    def _step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        grad_scaler: Optional[GradScaler] = None,
    ):
        """
        Update training progress after accumulating another local batch size. Depending on the configuration, this will
        report progress to peers, run global or local optimizer step, average parameters or schedule background tasks.

        :param closure: A closure that reevaluates the model and returns the loss.
        :param batch_size: optional override for batch_size_per_step from init.
        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        if grad_scaler is not None and not isinstance(grad_scaler, GradScaler):
            raise ValueError("hivemind.Optimizer requires a hivemind-aware gradient scaler (hivemind.GradScaler)")
        if self.batch_size_per_step is None and batch_size is None and not self.auxiliary:
            raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
        if self.auxiliary and (closure is not None or batch_size is not None):
            raise ValueError("Auxiliary peers should not have batch size, run closures, or use grad_scaler")
        batch_size = batch_size if batch_size is not None else self.batch_size_per_step

        # if delayed updates finished before step, apply these updates; otherwise do nothing
        self.state_averager.step(apply_delayed_updates=True)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # accumulate gradients toward target batch size, then aggregate with peers and run optimizer
        self._check_and_accumulate_gradients(batch_size)

        self._maybe_schedule_gradient_averaging()
        # self._maybe_schedule_state_averaging()

        if self.in_update and self.tracker.ready_to_update_epoch:
            self.state_averager.allow_state_sharing = False  # Prevent state sharing during AR.
            self.grad_averager.processed_batches.copy_(float(self.tracker.local_progress.samples_accumulated))
            self.state_averager.global_epoch = self.tracker.global_epoch
            self.grad_averager.global_epoch = self.tracker.global_epoch
            with self.optimizer_lock:
                self._update_global_epoch(grad_scaler)

            if self.model.model_args.use_compression:
                self.model.ss_regularize()

            self.in_update = False
        return loss

    def load_state_from_peers(self, wait_for_end_round=False, **kwargs):
        # Wait for a while grad accumulation round before requesting state from peers
        logger.info(f"Waiting for peers in stage to finish step before joining")
        while True:
            if (
                self.tracker.fetched_global_progress_this_epoch.is_set()
                and self.tracker.global_progress.samples_accumulated < self.target_batch_size * 0.1
            ):
                break
            else:
                logger.info(f"Waiting for peers in stage to finish step before joining")
                time.sleep(self.tracker.max_refresh_period)

        self._load_state_from_peers(**kwargs)

        if wait_for_end_round:
            self.tracker.fetched_global_progress_this_epoch.clear()
            while True:
                if (
                    self.tracker.fetched_global_progress_this_epoch.is_set()
                    and self.tracker.global_progress.samples_accumulated < self.target_batch_size * 0.2
                ):
                    break
                else:
                    logger.info(f"Downloaded state, waiting for start of new round")
                    time.sleep(0.5)
        logger.info(f"Joining run")

    def _load_state_from_peers(self, **kwargs):
        """
        Attempt to load the newest collaboration state from other peers within the same run_id.

        If successful, this will update parameters, optimizer state, local epoch and learning rate schedule in-place.
        """
        # note: we tag along for the next all-reduce because the run may have already started and cancelling it
        # will cause peers to restart matchmaking and may  stall the entire collaboration for a few seconds.
        if self.scheduled_grads is not None and not self.scheduled_grads.done():
            self._tag_along_with_zero_weight(self.scheduled_grads)
            self.scheduled_grads = None

        with self.tracker.pause_updates():
            while True:
                try:
                    self.state_averager.load_state_from_peers(timeout=self.load_state_timeout, **kwargs)
                    if self.grad_averager is not None:
                        self.grad_averager.load_state_from_peers(timeout=self.load_state_timeout, **kwargs)
                    break
                except KeyboardInterrupt:
                    raise
                except BaseException as e:
                    logger.error(
                        f"Failed to load state from peers: {e}, retrying ...",
                        exc_info=logger.getEffectiveLevel() <= 15,
                    )
                    continue

            if self.tracker.global_epoch - 1 <= self.local_epoch < self.tracker.global_epoch:
                logger.log(self.status_loglevel, f"Catching up with collaboration step {self.tracker.global_epoch}")
                self.state_averager.local_epoch = self.tracker.global_epoch

            self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch

            if self.use_gradient_averaging:
                self.grad_averager.reset_accumulated_grads_()
                if not self.client_mode:
                    self.grad_averager.state_sharing_priority = self.local_epoch

            if hasattr(self.state_averager, "zclip") and self.state_averager.zclip.var.item() > 0.0:
                self.state_averager.zclip.initialized = True
