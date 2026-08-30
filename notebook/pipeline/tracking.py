"""추적기 어댑터.

supervision 0.28 에서 `sv.ByteTrack` 이 deprecated 됐고 0.30 에서 제거된다.
대체재는 별도 `trackers` 패키지의 `ByteTrackTracker` 인데,
2026-08 현재 이 패키지가 supervision 0.29 와 호환되지 않는다
(`box_iou_batch` import 실패).

Colab 의 패키지 버전을 통제할 수 없으므로 **새 것을 먼저 시도하고 안 되면 기존 것**을 쓴다.
메서드 이름도 다르다 (update vs update_with_detections) — 여기서 흡수한다.
"""

import warnings
from typing import Any


class Tracker:
    """어느 구현이 쓰이든 같은 인터페이스."""

    def __init__(self, impl: Any, call: str, backend: str):
        self._impl = impl
        self._call = call
        self.backend = backend

    def update(self, detections):
        return getattr(self._impl, self._call)(detections)

    def reset(self):
        if hasattr(self._impl, "reset"):
            self._impl.reset()


def make(minimum_consecutive_frames: int = 2) -> Tracker:
    """가용한 ByteTrack 구현을 찾아 감싼다."""
    # 1) 새 패키지
    try:
        from trackers import ByteTrackTracker
        try:
            impl = ByteTrackTracker(minimum_consecutive_frames=minimum_consecutive_frames)
        except TypeError:
            impl = ByteTrackTracker()
        return Tracker(impl, "update", "trackers.ByteTrackTracker")
    except Exception:
        pass

    # 2) supervision 내장 (deprecated 경고는 이미 아는 사실이라 억제)
    import supervision as sv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        impl = sv.ByteTrack(minimum_consecutive_frames=minimum_consecutive_frames)
    return Tracker(impl, "update_with_detections", "supervision.ByteTrack")
