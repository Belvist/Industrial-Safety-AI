import json
import logging
import queue
import threading
import urllib.request
from uuid import uuid4

LABELS = {
    "no_helmet": "Нет каски",
    "no_vest": "Нет жилета",
    "no_mask": "Нет респиратора",
    "smoking": "Курение в запрещённой зоне",
    "danger_zone": "Опасная зона",
}


class Notifier:
    def __init__(self, settings, storage):
        self.settings, self.storage = settings, storage
        self.jobs = queue.Queue(maxsize=100)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def ready(self):
        return bool(
            self.settings.telegram_enabled
            and self.settings.telegram_bot_token
            and self.settings.telegram_chat_id
        )

    def submit(self, incident, photo):
        if not self.ready:
            self.storage.update(
                incident["id"],
                notification="not_configured" if self.settings.telegram_enabled else "disabled",
            )
            return
        self.storage.update(incident["id"], notification="pending")
        try:
            self.jobs.put_nowait((incident, photo))
        except queue.Full:
            self.storage.update(incident["id"], notification="failed")

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            incident, photo = job
            try:
                self._send(incident, photo)
                self.storage.update(incident["id"], notification="sent")
            except Exception:  # noqa: BLE001 - delivery must not stop the worker or expose bot credentials
                # Never log the request URL: it includes the Telegram bot token.
                logging.getLogger(__name__).warning(
                    "Telegram delivery failed for incident %s", incident["id"]
                )
                self.storage.update(incident["id"], notification="failed")

    def _send(self, incident, photo):
        boundary = uuid4().hex
        # Caption format per project description §10 (Telegram).
        caption = (
            "Нарушение промышленной безопасности\n"
            f"Тип: {LABELS.get(incident['type'], incident['type'])}\n"
            f"Камера: {incident['camera_name']}\n"
            f"Worker: #{incident['track_id']}\n"
            f"Время: {incident['started_at']}\n"
            f"Длительность: {float(incident.get('duration') or 0):.1f} сек.\n"
            f"Уверенность: {incident['confidence']:.0%}"
        )
        fields = {
            "chat_id": self.settings.telegram_chat_id,
            "caption": caption,
        }
        body = bytearray()
        for key, value in fields.items():
            body.extend(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            )
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="incident.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
        )
        body.extend(photo)
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendPhoto",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            if not json.load(response).get("ok"):
                raise RuntimeError("Telegram rejected delivery")

    def close(self):
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=1)
