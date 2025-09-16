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

from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import requests

from cryptography.hazmat.primitives import serialization
from hivemind import PeerID
from hivemind.proto import crypto_pb2
from hivemind.proto.auth_pb2 import AccessToken
from hivemind.utils.auth import TokenAuthorizerBase
from hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey
from hivemind.utils.logging import get_logger

from node0.security.integrity_check import verify_integrity
from node0.utils.connection_test_server import TestServer
from node0.utils.node_info import (
    BadRequestError,
    IntegrityError,
    NodeInfo,
    NotInAllowlistError,
    ServerUnavailableError,
    TestServerError,
    call_with_retries,
)


logger = get_logger(__name__)


class PluralisAuthorizer(TokenAuthorizerBase):
    def __init__(
        self,
        peer_id: str,
        user_token: str,
        user_email: str,
        role: str,
        auth_server: str,
        node_info: NodeInfo,
        current_path: Path,
        announce_maddrs: str,
        host_port: int,
        check_integrity: bool,
    ):
        super().__init__()

        self.peer_id = peer_id
        self._user_token = user_token
        self._user_email = user_email
        self._role = role
        self._auth_server = auth_server
        self._node_info = node_info
        self._current_path = current_path
        self._authority_public_key = None
        self._check_integrity = check_integrity
        self.pipeline_stage = None
        self.reachable = "unknown"
        self.monitor_public_key = None

        # Parse announce address
        address_parts = announce_maddrs.split("/")
        self._announce_ip_address = address_parts[2]
        self._announce_port = int(address_parts[4])
        self._host_port = host_port

    async def get_token(self) -> AccessToken:
        """Hivemind calls this method to refresh the access token when necessary."""

        try:
            self.join_experiment()
            return self._local_access_token
        except NotInAllowlistError as e:
            logger.error(f"Authorization failed: {e}. Exiting run.")
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except BadRequestError as e:
            logger.error(f"Authorization failed: {e}. Exiting run.")
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except IntegrityError:
            logger.error("Authorization failed: verification failed. Exiting run.")
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except Exception as e:
            logger.error(f"Authorization failed: {e}. Exiting run.")
            os.killpg(os.getpgrp(), signal.SIGTERM)

    def join_experiment(
        self,
        reset_reachability: bool = False,
        initial_join: bool = False,
        n_retries: int = 10,
    ) -> None:
        """Join experiment with retries."""
        call_with_retries(
            partial(self._join_experiment, reset_reachability, initial_join), n_retries=n_retries, initial_delay=3
        )

    def _join_experiment(self, reset_reachability: bool = False, initial_join: bool = False) -> None:
        """Send authorization request to join the experiment and receive access token."""
        try:
            # Check integrity of files
            if self._check_integrity:
                try:
                    integrity_hash = verify_integrity(self._current_path, self._local_private_key)
                except Exception:
                    raise IntegrityError("Verification failed.") from None
            else:
                integrity_hash = b"hash"

            url = f"{self._auth_server}/api/join"
            headers = {
                "Authorization": f"Bearer {self._user_token}",
                "request-type": "initial" if initial_join else "update",
            }
            json_body = {
                "peer_id": self.peer_id,
                "role": self._role,
                "peer_public_key": self.local_public_key.to_bytes().decode(),
                "device": self._node_info.device_name,
                "gpu_memory": self._node_info.gpu_memory,
                "ram": self._node_info.ram,
                "download_speed": self._node_info.download_speed,
                "upload_speed": self._node_info.upload_speed,
                "latency": self._node_info.latency,
                "integrity_hash": integrity_hash.decode(),
                "reset_reachability": reset_reachability,
                "email": self._user_email,
                "announce_ip_address": self._announce_ip_address,
                "announce_port": self._announce_port,
            }

            if initial_join:
                with TestServer(port=self._host_port) as server:
                    response = requests.put(
                        url,
                        headers=headers,
                        json=json_body,
                    )

                    response.raise_for_status()

                    # Receive server message
                    if not server.get_message():
                        if not server.wait_for_message(timeout=5):
                            raise TestServerError(
                                "Port test failed. Make sure your port forwarding is correct"
                            ) from None

                    # Verify message content
                    if not server.verify_message():
                        raise TestServerError(
                            "Port test failed, wrong message received. Please wait for few minutes before trying to join again"
                        ) from None

            else:
                response = requests.put(
                    url,
                    headers=headers,
                    json=json_body,
                )

                response.raise_for_status()

            response = response.json()

            self._authority_public_key = RSAPublicKey.from_bytes(response["auth_server_public_key"].encode())
            self.monitor_public_key = str(response["monitor_public_key"])
            self.pipeline_stage = response["stage_type"]
            self.reachable = response["reachable"]

            access_token = AccessToken()
            access_token.username = response["username"]
            access_token.public_key = response["peer_public_key"].encode()
            access_token.expiration_time = str(datetime.fromisoformat(response["expiration_time"]))
            access_token.signature = response["signature"].encode()
            self._local_access_token = access_token

            logger.info(
                f"Access for user {access_token.username} has been granted until {access_token.expiration_time} UTC"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [401, 403, 429]:  # Unauthorized, blacklisted, blocked or IP rate limited
                try:
                    error_detail = e.response.json()["detail"]
                except Exception:
                    error_detail = "Request is blocked"
                raise NotInAllowlistError(error_detail) from None
            if e.response.status_code in [400, 413, 418, 422, 424]:  # wrong request information
                raise BadRequestError(e.response.json()["detail"]) from None
            if e.response.status_code == 503:  # can't join due to rate limiting
                raise ServerUnavailableError(e.response.json()["detail"]) from None
            raise e
        except Exception as e:
            raise e

    def is_token_valid(self, access_token: AccessToken) -> bool:
        """Verify that token is valid."""
        data = self._token_to_bytes(access_token)
        if not self._authority_public_key or not self._authority_public_key.verify(data, access_token.signature):
            logger.error("Access token has invalid signature")
            return False

        try:
            expiration_time = datetime.fromisoformat(access_token.expiration_time)
        except ValueError:
            logger.error(f"datetime.fromisoformat() failed to parse expiration time: {access_token.expiration_time}")
            return False
        if expiration_time < datetime.now(timezone.utc):
            logger.error("Access token has expired")
            return False

        return True

    _MAX_LATENCY = timedelta(minutes=1)

    def does_token_need_refreshing(self, access_token: AccessToken) -> bool:
        """Check if token has expired."""
        expiration_time = datetime.fromisoformat(access_token.expiration_time)
        return expiration_time < datetime.now(timezone.utc) + self._MAX_LATENCY

    @staticmethod
    def _token_to_bytes(access_token: AccessToken) -> bytes:
        """Convert access token to bytes."""
        return f"{access_token.username} {access_token.public_key} {access_token.expiration_time}".encode()


def save_identity(private_key: RSAPrivateKey, identity_path: str) -> None:
    """Save private key to file.

    Args:
        private_key (RSAPrivateKey): local private key
        identity_path (str): path to save the key

    Raises:
        FileNotFoundError: can't create file
    """
    protobuf = crypto_pb2.PrivateKey(key_type=crypto_pb2.KeyType.RSA, data=private_key.to_bytes())

    try:
        with open(identity_path, "wb") as f:
            f.write(protobuf.SerializeToString())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The directory `{os.path.dirname(identity_path)}` for saving the identity does not exist"
        ) from e
    os.chmod(identity_path, 0o400)


def authorize_with_pluralis(
    node_info: NodeInfo,
    user_token: str,
    user_email: str,
    role: str,
    auth_server: str,
    identity_path: str,
    current_path: Path,
    announce_maddrs: str,
    host_port: int,
    check_integrity: bool = True,
) -> PluralisAuthorizer:
    """Generate local keys and send authorization request to join the run.

    Args:
        node_info (NodeInfo): information about the node
        user_token (str): authentication token
        user_email (str): email address
        role (str): role in the swarm
        auth_server (str): authorization server URL
        identity_path (str): path to save/load private key
        current_path (Path): path to src/node0
        announce_maddrs (str): announce address
        host_port: (int): host port
        check_integrity (bool): flag to check integrity

    Returns:
        PluralisAuthorizer: authorizer instance
    """
    logger.info("Authorization started...")

    # Generate private key or read from file
    local_private_key = RSAPrivateKey.process_wide()

    if os.path.exists(identity_path):
        with open(identity_path, "rb") as f:
            key_data = crypto_pb2.PrivateKey.FromString(f.read()).data
            private_key = serialization.load_der_private_key(key_data, password=None)
            if local_private_key._process_wide_key:
                local_private_key._process_wide_key._private_key = private_key
            else:
                logger.error("Failed to initialize process-wide private key")
                raise RuntimeError("Process-wide key is None")
    else:
        save_identity(local_private_key, identity_path)

    # Get static peer id
    with open(identity_path, "rb") as f:
        peer_id = str(PeerID.from_identity(f.read()))

    # Authorize
    authorizer = PluralisAuthorizer(
        peer_id,
        user_token,
        user_email,
        role,
        auth_server,
        node_info,
        current_path,
        announce_maddrs,
        host_port,
        check_integrity,
    )

    try:
        authorizer.join_experiment(reset_reachability=True, initial_join=True)
        logger.info("Authorization completed")
        return authorizer
    except NotInAllowlistError as e:
        logger.error(f"Authorization failed: {e}. Exiting run.")
        exit(1)
    except BadRequestError as e:
        logger.error(f"Authorization failed: {e}. Exiting run.")
        exit(1)
    except IntegrityError:
        logger.error("Authorization failed: verification failed. Exiting run.")
        exit(1)
    except Exception as e:
        logger.error(f"Authorization failed: {e}. Exiting run.")
        exit(1)
