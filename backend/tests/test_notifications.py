import io
import threading

from app.core.config import Settings
from app.services.notifications import Notifier
from app.services.storage import Storage


def test_telegram_multipart_and_delivery_status_without_network(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "test.db")
    config = Settings(
        _env_file=None,
        telegram_enabled=True,
        telegram_bot_token="test-token",
        telegram_chat_id="123",
    )
    record = storage.create_incident("cam", "Цех", (1, "no_helmet"), 0.91, 2)
    sent = []
    done = threading.Event()

    def request(req, timeout):
        sent.append(req)
        assert b"Content-Type: image/jpeg" in req.data
        assert b"jpeg-evidence" in req.data
        assert timeout == 15
        return io.BytesIO(b'{"ok":true}')

    original = storage.update

    def update(incident_id, **values):
        original(incident_id, **values)
        if values.get("notification") in {"sent", "failed"}:
            done.set()

    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setattr(storage, "update", update)
    notifier = Notifier(config, storage)
    try:
        notifier.submit(record, b"jpeg-evidence")
        assert done.wait(2)
        assert len(sent) == 1
        assert storage.get(record["id"])["notification"] == "sent"
    finally:
        notifier.close()


def test_telegram_failure_is_visible_and_token_not_logged(tmp_path, monkeypatch, caplog):
    storage = Storage(tmp_path / "test.db")
    config = Settings(
        _env_file=None,
        telegram_enabled=True,
        telegram_bot_token="secret-value",
        telegram_chat_id="123",
    )
    record = storage.create_incident("cam", "Цех", (1, "no_vest"), 0.91, 2)
    done = threading.Event()

    def fail(req, timeout):
        raise OSError("Do not log this URL: secret-value")

    original = storage.update

    def update(incident_id, **values):
        original(incident_id, **values)
        if values.get("notification") == "failed":
            done.set()

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr(storage, "update", update)
    notifier = Notifier(config, storage)
    try:
        notifier.submit(record, b"photo")
        assert done.wait(2)
        assert storage.get(record["id"])["notification"] == "failed"
        assert "secret-value" not in caplog.text
    finally:
        notifier.close()


def test_caption_matches_pdf_section_10_format(tmp_path, monkeypatch):
    from app.services.notifications import LABELS

    storage = Storage(tmp_path / "test.db")
    config = Settings(
        _env_file=None,
        telegram_enabled=True,
        telegram_bot_token="test-token",
        telegram_chat_id="123",
    )
    record = storage.create_incident("cam", "Цех №1", (17, "no_helmet"), 0.94, 3.2)
    bodies = []

    def request(req, timeout):
        bodies.append(req.data)
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr("urllib.request.urlopen", request)
    notifier = Notifier(config, storage)
    try:
        notifier._send(record, b"photo")
    finally:
        notifier.close()
    assert len(bodies) == 1
    text = bodies[0].decode("utf-8", "ignore")
    assert LABELS["no_helmet"] == "Нет каски"
    for line in [
        "Нарушение промышленной безопасности",
        "Тип: Нет каски",
        "Камера: Цех №1",
        "Worker: #17",
        "Время:",
        "Длительность: 3.2 сек.",
        "Уверенность: 94%",
    ]:
        assert line in text
