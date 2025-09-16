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

import socket
import threading
import time

from hivemind.utils.logging import get_logger

from node0.utils.node_info import TestServerError


logger = get_logger(__name__)


class TestServer:
    def __init__(self, host="0.0.0.0", port=49200):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.thread = None
        self.received_message = None  # Store single message
        self.message_received = threading.Event()  # Signal when message arrives

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """Start the server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1)
            self.running = True
        except OSError as e:
            logger.error(f"Failed to bind to {self.host}:{self.port}: {e}")
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            raise TestServerError(f"Failed to start server on {self.host}:{self.port}: {e}") from None

        def server_loop():
            logger.info(f"Test server listening on {self.host}:{self.port}")
            while self.running and not self.message_received.is_set():
                try:
                    client_socket, addr = self.server_socket.accept()
                    logger.info("Test server connection received")

                    # Receive the message
                    try:
                        client_socket.settimeout(2)
                        data = client_socket.recv(1024)
                        if data:
                            message = data.decode("utf-8", errors="ignore")
                            self.received_message = {"message": message, "from": "", "timestamp": time.time()}
                            self.message_received.set()  # Signal message received
                            logger.info(f"Received message: {message}")
                        else:
                            logger.info("No data received")
                    except TimeoutError:
                        logger.info("Timeout waiting for data")
                    except Exception as e:
                        logger.error(f"Error receiving data: {e}")
                    finally:
                        client_socket.close()

                except TimeoutError:
                    continue
                except OSError:
                    break

        self.thread = threading.Thread(target=server_loop)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)  # Let server start

    def close(self):
        """Close the server"""
        if self.running:
            self.running = False
            if self.server_socket:
                self.server_socket.close()
            if self.thread:
                self.thread.join(timeout=1)
            logger.info("Test server closed")

    def get_message(self):
        """Get the received message (or None if not received yet)"""
        return self.received_message

    def wait_for_message(self, timeout=10):
        """Wait for the message to be received. Returns True if received, False if timeout"""
        return self.message_received.wait(timeout)

    def verify_message(self):
        """Verify the received message contains expected content"""
        if self.received_message and "auth" in self.received_message["message"]:
            return True
        return False
