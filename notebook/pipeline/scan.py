"""사전 스캔: 영상을 2fps로 싸게 훑어 프레임별 프로파일을 만든다.

전체 영상을 10fps로 정밀 처리하면 대부분이 버려질 구간이라 낭비다.
여기서 "어디가 쓸 만한가"를 먼저 알아낸 뒤, 채택 구간에만 비용을 쓴다.

Phase 0에서는 이 프로파일 자체가 산출물이다 — 프로젝트가 가능한지를 이 숫자가 결정한다.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from . import geometry, models as M


@dataclass
class VideoMeta:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float


def _keypoints_of(result) -> tuple:
    """추론 결과에서 (xy, confidence) 추출. 실패하면 (None, None)."""
    import supervision as sv

    try:
        kp = sv.KeyPoints.from_inference(result)
    except Exception:
        return None, None
    if kp.xy is None or len(kp.xy) == 0:
        return None, None
    conf = kp.confidence
    if conf is None or len(conf) == 0:
        return None, None
    return kp.xy[0], conf[0]


def scan(
    video_path: str,
    mdl: M.Models,
    preset,
    start_s: float = 0.0,
    max_seconds: Optional[float] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """영상을 preset.scan_fps 로 훑어 프레임별 지표를 수집한다.

    Returns:
        {"video": {...}, "preset": str, "stride": int, "dt": float, "rows": [ {...}, ... ]}

        각 row:
            t              [s] 영상 내 시각
            n_keypoints    신뢰도 통과한 경기장 키포인트 수
            n_inliers      RANSAC 인라이어 수 (H 품질)
            n_players      화면 내 선수 수 (GK 제외)
            n_goalkeepers  골키퍼 수
            h_ok           호모그래피가 성립하고 물리적으로 말이 되는가
            camera_speed   직전 유효 프레임 대비 카메라 이동 [m/s]. 컷이면 폭발한다
            color_sim      직전 프레임과의 HSV 히스토그램 상관 (컷 감지 보조 신호)
    """
    import supervision as sv

    info = sv.VideoInfo.from_video_path(video_path)
    meta = VideoMeta(
        path=video_path, fps=float(info.fps), total_frames=int(info.total_frames),
        width=int(info.width), height=int(info.height),
        duration_s=float(info.total_frames) / float(info.fps),
    )

    stride = max(1, int(round(meta.fps / preset.scan_fps)))
    dt = stride / meta.fps
    start_frame = int(start_s * meta.fps)
    end_frame = None
    if max_seconds is not None:
        end_frame = min(meta.total_frames, start_frame + int(max_seconds * meta.fps))

    config = M.pitch_config(preset)
    vertices_cm = np.array(config.vertices, dtype=np.float32)
    probes = geometry.probe_points(meta.width, meta.height)

    frames = sv.get_video_frames_generator(
        source_path=video_path, stride=stride, start=start_frame, end=end_frame
    )
    n_expected = ((end_frame or meta.total_frames) - start_frame) // stride
    if show_progress:
        from tqdm.auto import tqdm
        frames = tqdm(frames, total=n_expected, desc=f"scan @{preset.scan_fps}fps")

    rows: List[Dict[str, Any]] = []
    prev_H: Optional[np.ndarray] = None
    prev_t: Optional[float] = None
    prev_hist: Optional[np.ndarray] = None

    for i, frame in enumerate(frames):
        t = (start_frame + i * stride) / meta.fps

        # --- 경기장 키포인트 → 호모그래피 ---
        kp_result = mdl.field.infer(frame, confidence=preset.det_conf)[0]
        kp_xy, kp_conf = _keypoints_of(kp_result)
        H, n_kp, n_inliers = geometry.solve_homography(
            kp_xy, kp_conf, vertices_cm, preset.kp_conf, preset.min_keypoints
        )
        h_ok = H is not None and geometry.is_sane(H, probes)
        if not h_ok:
            H = None

        # --- 선수 탐지 ---
        det_result = mdl.player.infer(frame, confidence=preset.det_conf)[0]
        det = sv.Detections.from_inference(det_result)
        if len(det):
            det = det[det.class_id != M.BALL_ID]
        if len(det):
            det = det.with_nms(threshold=preset.nms_threshold, class_agnostic=True)
        n_players = int((det.class_id == M.PLAYER_ID).sum()) if len(det) else 0
        n_gk = int((det.class_id == M.GOALKEEPER_ID).sum()) if len(det) else 0

        # --- 카메라 움직임 ---
        speed = float("nan")
        if H is not None and prev_H is not None and prev_t is not None:
            speed = geometry.camera_speed(prev_H, H, probes, t - prev_t)

        # --- 색 히스토그램 (컷 감지 보조) ---
        hist = geometry.color_signature(frame)
        color_sim = float("nan") if prev_hist is None else geometry.color_similarity(prev_hist, hist)
        prev_hist = hist

        rows.append(dict(
            t=round(t, 3), n_keypoints=n_kp, n_inliers=n_inliers,
            n_players=n_players, n_goalkeepers=n_gk,
            h_ok=bool(h_ok), camera_speed=speed, color_sim=color_sim,
        ))

        if H is not None:
            prev_H, prev_t = H, t

    return {
        "video": asdict(meta),
        "preset": preset.name,
        "stride": stride,
        "dt": dt,
        "rows": rows,
    }
