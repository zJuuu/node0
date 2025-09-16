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

from typing import TYPE_CHECKING

from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from pydantic.v1 import BaseModel, StrictFloat, StrictInt, StrictStr


if TYPE_CHECKING:
    # For type checkers, treat it as bytes
    BytesWithPublicKeyType = bytes
else:
    # At runtime, use the actual validator
    BytesWithPublicKeyType = BytesWithPublicKey


class WorkerMetricsV1(BaseModel):
    """Schema for worker metrics."""

    peer_id: str
    num_flop: StrictFloat
    active_time: StrictFloat


class WorkerPortV1(BaseModel):
    """Schema for worker port reachability."""

    peer_id: str
    is_open: bool


class RunParameters(BaseModel):
    peer_id: bytes
    averaging_target_batch_size: StrictInt
    scheduler: StrictStr
    num_warmup_steps: StrictInt
    num_training_steps: StrictInt
    averaging_timeout: StrictFloat
    matchmaking_time: StrictFloat
    request_timeout: StrictFloat
    load_state_timeout: StrictFloat
    time: StrictFloat


class MetricSchema(BaseModel):
    """Force metrics keys to have signed subkeys."""

    worker_metrics: dict[BytesWithPublicKeyType, WorkerMetricsV1]


class PortSchema(BaseModel):
    """Force port keys to have signed subkeys."""

    worker_ports: dict[BytesWithPublicKeyType, WorkerPortV1]


class RunParametersSchema(BaseModel):
    paramstore: dict[BytesWithPublicKeyType, RunParameters | None]


def make_validators(experiment_prefix: str, peer_id: str, stage: str) -> tuple[list[RecordValidatorBase], bytes]:
    """Create all validators"""
    metric_validator = SchemaValidator(MetricSchema, prefix=f"{experiment_prefix}_{stage}")
    port_validator = SchemaValidator(PortSchema, prefix=f"{experiment_prefix}_{peer_id}")
    param_validator = SchemaValidator(RunParametersSchema, prefix=stage.split(".")[0])
    signature_validator = RSASignatureValidator()

    validators = [metric_validator, port_validator, param_validator, signature_validator]
    return validators, signature_validator.local_public_key
