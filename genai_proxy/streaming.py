"""等待上游事件时维持 SSE 连接。"""

from queue import Empty, Full, Queue
from threading import Event, Thread


def iter_with_heartbeat(events, interval=5):
    """立即产出 None，并在上游静默时定期产出 None 作为心跳。

    队列限制为一个事件；客户端断开后停止读取，关闭上游迭代器。
    正在阻塞的网络读取仍由上游的读取超时结束。
    """
    pending = Queue(maxsize=1)
    stopped = Event()

    def publish(item):
        while not stopped.is_set():
            try:
                pending.put(item, timeout=0.1)
                return
            except Full:
                pass

    def consume():
        try:
            for event in events:
                if stopped.is_set():
                    break
                publish(("event", event))
        except Exception as exc:
            publish(("error", exc))
        finally:
            close = getattr(events, "close", None)
            if close:
                close()
            publish(("end", None))

    try:
        yield None
        Thread(target=consume, daemon=True).start()
        while True:
            try:
                kind, value = pending.get(timeout=interval)
            except Empty:
                yield None
                continue
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()
