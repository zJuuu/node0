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
import threading
import time

import psutil

from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class NetworkMonitor(threading.Thread):
    """
    Monitor network throughput for a specific network interface or all interfaces.

    :param interface: Specific network interface to monitor (e.g., 'eth0')
    :param interval: Time between measurements in seconds
    :param duration: Total monitoring duration in seconds (None for continuous monitoring)
    """

    def __init__(
        self,
        interface: str | None = None,
        interval: int = 60,
        duration: int | None = None,
    ):
        super().__init__()

        self.interface = interface
        self.interval = interval
        self.duration = duration

        self.start()

    def get_tcp_connection_count(self) -> int:
        """
        Get the count of TCP connections from host system.
        """
        try:
            tcp_file = "/proc/1/root/proc/net/tcp"
            if os.path.exists(tcp_file):
                with open(tcp_file, "r") as f:
                    return len(f.readlines()) - 1  # Subtract 1 for header line
        except (OSError, PermissionError):
            pass

        # method fails, return -1 to indicate unavailable
        return -1

    def run(self):
        # Store initial network counters
        logger.info("Running network bandwidth monitor")
        initial_counters = psutil.net_io_counters(pernic=True)
        start_time = time.time()

        try:
            # Determine interfaces to monitor
            if self.interface:
                interfaces = [self.interface]
            else:
                interfaces = [iface for iface in initial_counters.keys() if iface != "lo"]

            # Monitoring loop
            while True:
                # Wait for the interval
                time.sleep(self.interval)

                # Get current network counters
                current_counters = psutil.net_io_counters(pernic=True)
                current_time = time.time()

                # Calculate time elapsed
                elapsed = current_time - start_time

                # Process each interface
                bytes_sent = 0
                bytes_recv = 0
                for iface in interfaces:
                    if iface not in current_counters:
                        continue

                    # Calculate throughput
                    initial = initial_counters.get(iface, None)
                    if not initial:
                        continue

                    bytes_sent += current_counters[iface].bytes_sent - initial.bytes_sent
                    bytes_recv += current_counters[iface].bytes_recv - initial.bytes_recv

                # Calculate megabytes sent and received per second
                bytes_sent = (bytes_sent) / elapsed / (1024 * 1024)
                bytes_recv = (bytes_recv) / elapsed / (1024 * 1024)

                bits_sent = bytes_sent * 8
                bits_recv = bytes_recv * 8

                # Get current TCP connection counts
                tcp_count = self.get_tcp_connection_count()

                # Log results
                logger.info(
                    f"Time {time.strftime('%Y-%m-%d %H:%M:%S'):<20} "
                    f"Interface Agg "
                    f"Sent {bits_sent:>10.2f} Mbps "
                    f"Rcv {bits_recv:>10.2f} Mbps "
                )

                logger.info(f"Time {time.strftime('%Y-%m-%d %H:%M:%S'):<20} Number open connections {tcp_count}")

                # Update initial counters and start time
                initial_counters = current_counters
                start_time = current_time

                # Check if monitoring duration is specified
                if self.duration and current_time - start_time >= self.duration:
                    break

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
