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

from typing import Tuple

import torch

from hivemind.moe.server import ModuleBackend
from hivemind.utils.nested import nested_compare, nested_flatten, nested_pack


class ModuleCollab(ModuleBackend):
    def __init__(self, optimizer_lock, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.optimizer_lock = optimizer_lock

    def backward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply backward pass to an aggregated batch of requests. Used by Runtime, do not call this manually
        To submit a request for asynchronous processing, please use ``ModuleBackend.backward_pool.submit_task``.

        Subclassing:
           This method receives a sequence of torch tensors following ``nested_flatten(self.backward_schema)``;

           It should return gradients w.r.t. inputs that follow ``nested_flatten(self.forward_schema)``;

           Runtime doesn't guarantee that backward will be performed in the same order and for the same data
           as forward, so we recommend stateless backward pass that re-runs expert forward pass inside backward.

           Please make sure to call ``ModuleBackend.on_backward`` after each call to backward
        """
        (args, kwargs), grad_outputs = nested_pack(inputs, structure=self.backward_schema)

        with torch.enable_grad():
            with self.optimizer_lock:
                args = [
                    tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()
                    for tensor in args
                ]
                kwargs = {
                    input_key: (
                        tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()
                    )
                    for input_key, tensor in kwargs.items()
                }

                batch_size = args[0].size(0)

                outputs = self.module(*args, **kwargs)
                assert nested_compare(outputs, grad_outputs), "outputs and grad_outputs must have the same structure"

                outputs_flat = tuple(nested_flatten(outputs))

                grad_outputs_flat = tuple(
                    map(
                        lambda grad, out: grad.to(device=out.device, dtype=out.dtype, non_blocking=True),
                        nested_flatten(grad_outputs),
                        outputs_flat,
                    )
                )
                torch.autograd.backward(
                    outputs_flat, grad_tensors=grad_outputs_flat, create_graph=False, retain_graph=False
                )
            self.on_backward(batch_size)

        return tuple(
            x.grad if isinstance(x.grad, torch.Tensor) else torch.zeros_like(x) for x in nested_flatten((args, kwargs))
        )

    def on_backward(self, batch_size: int) -> None:
        """
        Train the expert for one step. This method is called by ``ModuleBackend.backward`` after computing gradients.
        """
        if self.optimizer is not None:
            self.optimizer.step(batch_size=batch_size)
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()
