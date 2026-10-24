"""Bounded RPC transport. Only the I/O thread ever touches Pika objects.

No request is retried implicitly: an uncertain publish must not duplicate work
on a shared accelerator. The next request reconnects after a transport failure.
"""

import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass


@dataclass
class Request:
    id: str
    body: bytes | None
    deadline: float
    future: Future


class AmqpPipeline:
    def __init__(self, owner, limit=1):
        self.owner = owner
        self.limit = limit
        self.slots = threading.BoundedSemaphore(limit)
        self.queue = queue.Queue(maxsize=limit)
        self.pending: dict[str, Request] = {}
        self.stopping = threading.Event()
        self._consumer_channel = None
        self.thread = threading.Thread(target=self._run, name="linq-amqp-io", daemon=True)
        self.thread.start()

    def call(self, body: bytes | None):
        timeout = self.owner.timeout_seconds
        if self.stopping.is_set():
            raise RuntimeError("LinQ transport is closed.")
        # One wall-clock budget for acquire + wait (not 2× timeout).
        deadline = time.monotonic() + timeout
        if not self.slots.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise RuntimeError("LinQ backpressure: in-flight limit reached.")
        future = Future()
        future.add_done_callback(lambda _: self.slots.release())
        item = Request(uuid.uuid4().hex, body, deadline, future)
        try:
            if self.stopping.is_set():
                raise RuntimeError("LinQ transport is closed.")
            self.queue.put_nowait(item)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            return future.result(timeout=remaining)
        except TimeoutError:
            future.cancel()
            raise RuntimeError(f"LinQ TPU не ответил за {timeout:.1f} с.") from None
        except BaseException:
            future.cancel()
            raise

    @staticmethod
    def _fail(item, message):
        if not item.future.done() and item.future.set_running_or_notify_cancel():
            item.future.set_exception(RuntimeError(message))

    def _drain(self, message):
        for item in list(self.pending.values()):
            self._fail(item, message)
        self.pending.clear()
        while True:
            try:
                self._fail(self.queue.get_nowait(), message)
            except queue.Empty:
                break

    def _reply(self, _channel, _method, props, body):
        from app.inference.tpu_wire import MESSAGE_ID_ATTR_NAME

        headers = props.headers or {}
        key = headers.get(MESSAGE_ID_ATTR_NAME) or getattr(props, "correlation_id", None)
        item = self.pending.pop(key, None) if key else None
        if item is None or item.future.done():
            return
        if time.monotonic() >= item.deadline:
            self._fail(item, "LinQ TPU не ответил: timeout.")
        elif item.future.set_running_or_notify_cancel():
            item.future.set_result(body)

    def _run(self):
        import json

        import pika

        from app.inference.tpu_wire import (
            MESSAGE_ID_ATTR_NAME,
            RESPONSE_ROUTING_KEYS_ATTR_NAME,
            TARGET_URI_ATTR_NAME,
        )

        try:
            while not self.stopping.is_set():
                current = None
                try:
                    for key, item in list(self.pending.items()):
                        if item.future.done() or time.monotonic() >= item.deadline:
                            self.pending.pop(key, None)
                            self._fail(item, "LinQ TPU не ответил: timeout.")
                    # Publish all admitted requests before waiting for replies.
                    for _ in range(self.limit):
                        try:
                            current = self.queue.get_nowait()
                        except queue.Empty:
                            break
                        if current.future.done() or time.monotonic() >= current.deadline:
                            self._fail(current, "LinQ TPU не ответил: timeout.")
                            continue
                        channel = self.owner._connect()
                        if (
                            hasattr(channel, "basic_consume")
                            and channel is not self._consumer_channel
                        ):
                            channel.basic_consume(
                                queue=self.owner._reply_queue,
                                on_message_callback=self._reply,
                                auto_ack=True,
                            )
                            self._consumer_channel = channel
                        if current.body is None:  # broker-only health probe
                            if current.future.set_running_or_notify_cancel():
                                current.future.set_result(None)
                            continue
                        self.pending[current.id] = current
                        started = time.perf_counter()
                        channel.basic_publish(
                            exchange=self.owner.amqp_exchange,
                            routing_key=self.owner.amqp_input_queue,
                            body=current.body,
                            properties=pika.BasicProperties(
                                correlation_id=current.id,
                                expiration=str(
                                    max(1, int((current.deadline - time.monotonic()) * 1000))
                                ),
                                headers={
                                    TARGET_URI_ATTR_NAME: self.owner.program_uri,
                                    RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps(
                                        [self.owner._reply_queue]
                                    ),
                                    MESSAGE_ID_ATTR_NAME: current.id,
                                },
                            ),
                        )
                        self.owner.metrics.observe(
                            "publish_ms", (time.perf_counter() - started) * 1000
                        )
                    if self.pending and not hasattr(self.owner._channel, "basic_consume"):
                        # Polling keeps compatibility with the vendor exchange protocol.
                        # Every reply is dispatched by ID, including out-of-order replies.
                        for _ in range(self.limit):
                            method, props, body = self.owner._channel.basic_get(
                                self.owner._reply_queue, auto_ack=True
                            )
                            if method is None:
                                break
                            self._reply(None, method, props, body)
                    connection = self.owner._connection
                    if connection is not None and hasattr(connection, "process_data_events"):
                        connection.process_data_events(time_limit=0)
                except Exception:
                    if current is not None:
                        self._fail(current, "LinQ TPU transport unavailable.")
                    self._drain("LinQ TPU transport unavailable.")
                    self.owner._close()
                self.stopping.wait(0.005)
        finally:
            self._drain("LinQ transport stopped.")
            self.owner._close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=self.owner.timeout_seconds + 1)
