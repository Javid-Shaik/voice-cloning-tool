"""
Custom handlers for the logging system
"""

import logging
import logging.handlers
import os
import time
import gzip
import requests
from typing import Optional
from threading import Lock
import queue
from concurrent.futures import ThreadPoolExecutor

class GZipRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Handler that compresses rotated log files
    """
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        if os.path.exists(self.baseFilename + ".1"):
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.baseFilename}.{i}.gz"
                dfn = f"{self.baseFilename}.{i + 1}.gz"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)

        dfn = self.baseFilename + ".1.gz"
        if os.path.exists(dfn):
            os.remove(dfn)

        with open(self.baseFilename, 'rb') as f_in:
            with gzip.open(dfn, 'wb') as f_out:
                f_out.writelines(f_in)

        if not self.delay:
            self.stream = self._open()

class AsyncHttpHandler(logging.Handler):
    """
    Asynchronous HTTP handler for sending logs to a remote endpoint
    """
    def __init__(self, url: str, max_workers: int = 1):
        super().__init__()
        self.url = url
        self.queue = queue.Queue()
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.start()

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.queue.put(msg)
        except Exception:
            self.handleError(record)

    def _send_log(self, msg: str):
        try:
            response = requests.post(self.url, json={'message': msg})
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send log to remote endpoint: {e}")

    def start(self):
        def worker():
            while True:
                try:
                    msg = self.queue.get()
                    if msg is None:
                        break
                    self._send_log(msg)
                except Exception as e:
                    print(f"Error in async handler worker: {e}")
                finally:
                    self.queue.task_done()

        self.executor.submit(worker)

    def close(self):
        self.queue.put(None)
        self.executor.shutdown()
        super().close()

class PerformanceHandler(logging.Handler):
    """
    Handler that tracks performance metrics
    """
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.start_times = {}

    def start_operation(self, operation: str):
        self.start_times[operation] = time.time()

    def end_operation(self, operation: str):
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]

    def get_metrics(self, operation: str) -> dict:
        if operation not in self.metrics:
            return {}
        
        durations = self.metrics[operation]
        return {
            'count': len(durations),
            'average': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations)
        }

    def emit(self, record: logging.LogRecord):
        if hasattr(record, 'operation_start'):
            self.start_operation(record.operation_start)
        if hasattr(record, 'operation_end'):
            self.end_operation(record.operation_end)