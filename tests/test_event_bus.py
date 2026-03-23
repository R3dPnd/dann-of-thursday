"""Unit tests for the EventBus."""

import threading
import pytest
from src.event_bus import EventBus


@pytest.fixture()
def eb():
    return EventBus()


class TestSubscribeAndEmit:
    def test_subscriber_receives_event(self, eb):
        received = []
        eb.subscribe(lambda t, p: received.append((t, p)))
        eb.emit("foo", {"x": 1})
        assert received == [("foo", {"x": 1})]

    def test_multiple_subscribers_all_notified(self, eb):
        calls = []
        eb.subscribe(lambda t, p: calls.append(1))
        eb.subscribe(lambda t, p: calls.append(2))
        eb.emit("bar", {})
        assert sorted(calls) == [1, 2]

    def test_emit_with_no_payload_passes_empty_dict(self, eb):
        received = []
        eb.subscribe(lambda t, p: received.append(p))
        eb.emit("ping")
        assert received == [{}]

    def test_subscribe_is_idempotent(self, eb):
        calls = []
        cb = lambda t, p: calls.append(1)
        eb.subscribe(cb)
        eb.subscribe(cb)  # second subscribe should be ignored
        eb.emit("test", {})
        assert calls == [1]


class TestUnsubscribe:
    def test_unsubscribed_callback_not_called(self, eb):
        calls = []
        cb = lambda t, p: calls.append(1)
        eb.subscribe(cb)
        eb.unsubscribe(cb)
        eb.emit("test", {})
        assert calls == []

    def test_unsubscribe_unknown_callback_is_noop(self, eb):
        eb.unsubscribe(lambda t, p: None)  # should not raise

    def test_other_subscribers_unaffected_by_unsubscribe(self, eb):
        calls = []
        cb1 = lambda t, p: calls.append(1)
        cb2 = lambda t, p: calls.append(2)
        eb.subscribe(cb1)
        eb.subscribe(cb2)
        eb.unsubscribe(cb1)
        eb.emit("test", {})
        assert calls == [2]


class TestFaultIsolation:
    def test_bad_subscriber_does_not_crash_emit(self, eb):
        good_calls = []
        eb.subscribe(lambda t, p: (_ for _ in ()).throw(RuntimeError("boom")))
        eb.subscribe(lambda t, p: good_calls.append(1))
        eb.emit("test", {})  # must not raise
        assert good_calls == [1]


class TestThreadSafety:
    def test_concurrent_emits_all_received(self, eb):
        received = []
        lock = threading.Lock()

        def cb(t, p):
            with lock:
                received.append(p["n"])

        eb.subscribe(cb)

        threads = [
            threading.Thread(target=eb.emit, args=("n", {"n": i}))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(received) == list(range(50))
