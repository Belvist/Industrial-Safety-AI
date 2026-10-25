"""Camera control protocol. Only explicit events can change lifecycle state."""

from enum import StrEnum


class CameraState(StrEnum):
    STOPPED = "stopped"
    UPLOADING = "uploading"
    STARTING = "starting"
    RUNNING = "running"
    STALE = "stale"
    STOPPING = "stopping"
    ENDED = "ended"
    ERROR = "error"


class CameraEvent(StrEnum):
    UPLOAD = "upload"
    START = "start"
    READY = "ready"
    FRAME = "frame"
    STALE = "stale"
    STOP = "stop"
    STOPPED = "stopped"
    EOF = "eof"
    FAIL = "fail"


TRANSITIONS = {
    CameraState.STOPPED: {
        CameraEvent.START: CameraState.STARTING,
        CameraEvent.UPLOAD: CameraState.UPLOADING,
    },
    CameraState.UPLOADING: {
        CameraEvent.STOP: CameraState.STOPPING,
        CameraEvent.FAIL: CameraState.ERROR,
    },
    CameraState.STARTING: {
        CameraEvent.READY: CameraState.RUNNING,
        CameraEvent.STOP: CameraState.STOPPING,
        CameraEvent.FAIL: CameraState.ERROR,
    },
    CameraState.RUNNING: {
        CameraEvent.FRAME: CameraState.RUNNING,
        CameraEvent.STALE: CameraState.STALE,
        CameraEvent.STOP: CameraState.STOPPING,
        CameraEvent.EOF: CameraState.ENDED,
        CameraEvent.FAIL: CameraState.ERROR,
    },
    CameraState.STALE: {
        CameraEvent.FRAME: CameraState.RUNNING,
        CameraEvent.EOF: CameraState.ENDED,
        CameraEvent.STOP: CameraState.STOPPING,
        CameraEvent.FAIL: CameraState.ERROR,
    },
    CameraState.STOPPING: {
        CameraEvent.STOPPED: CameraState.STOPPED,
        CameraEvent.FAIL: CameraState.ERROR,
    },
    CameraState.ENDED: {
        CameraEvent.START: CameraState.STARTING,
        CameraEvent.UPLOAD: CameraState.UPLOADING,
        CameraEvent.STOP: CameraState.STOPPING,
    },
    CameraState.ERROR: {
        CameraEvent.START: CameraState.STARTING,
        CameraEvent.UPLOAD: CameraState.UPLOADING,
        CameraEvent.STOP: CameraState.STOPPING,
    },
}


class CameraMachine:
    def __init__(self):
        self.state = CameraState.STOPPED
        self.version = 0

    def dispatch(self, event: CameraEvent):
        target = TRANSITIONS[self.state].get(event)
        if target is None:
            raise RuntimeError(f"Недопустимый переход камеры: {self.state} → {event}.")
        self.state = target
        self.version += 1
        return target
