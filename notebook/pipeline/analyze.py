"""Phase 1 — 채택 구간을 정밀 처리해 분석 결과를 만든다.

사전 스캔(2fps)이 고른 구간만 analyze_fps(10fps)로 다시 훑는다.
전체 영상을 정밀 처리하면 대부분 버려질 구간에 비용을 쓰게 된다.

구간 하나당:
    프레임별 호모그래피 → 선수 탐지 → ByteTrack → 경기장 좌표
    → 팀 분류 → 스무딩 → 지표
"""

import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import geometry, metrics, models as M, smooth, teams, tracking

# 팀 분류 학습용 크롭을 모을 때의 프레임 간격 (구간이 짧으면 촘촘히)
CROP_STRIDE = 3
# 트랙당 남길 썸네일 수 (라벨링 UI 에서 사람이 알아볼 최소한)
THUMBS_PER_TRACK = 2
THUMB_HEIGHT = 96


def _encode_thumb(crop: np.ndarray) -> Optional[str]:
    """선수 크롭 → base64 JPEG data URI."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 8 or w < 4:
        return None
    scale = THUMB_HEIGHT / h
    small = cv2.resize(crop, (max(4, int(w * scale)), THUMB_HEIGHT))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _attack_direction(tracks: List[Dict[str, Any]], length_m: float) -> Dict[str, Any]:
    """어느 팀이 +x 로 공격하는지 추정한다.

    골키퍼 위치가 가장 확실한 단서다 — 골키퍼는 자기 골대 앞에 있고,
    그 팀은 반대쪽으로 공격한다. 골키퍼가 안 잡히면 판정하지 않는다
    (억지로 추측하면 히트맵이 통째로 뒤집힌다).
    """
    for tr in tracks:
        if tr.get("role") != "goalkeeper" or tr.get("team") is None or len(tr["xy"]) == 0:
            continue
        gk_x = float(np.mean(tr["xy"][:, 0]))
        # 골키퍼가 x=0 쪽에 있으면 그 팀은 +x 로 공격한다
        attacks_positive = gk_x < length_m / 2
        return {"team0_attacks_positive": attacks_positive if tr["team"] == 0
                                          else not attacks_positive,
                "method": "goalkeeper", "gk_x": round(gk_x, 1)}
    return {"team0_attacks_positive": None, "method": "unknown", "gk_x": None}


def analyze_window(video_path: str, window: Dict[str, Any], mdl: M.Models, preset,
                   show_progress: bool = True) -> Dict[str, Any]:
    """구간 하나를 정밀 처리한다."""
    import supervision as sv

    info = sv.VideoInfo.from_video_path(video_path)
    fps = float(info.fps)
    stride = max(1, int(round(fps / preset.analyze_fps)))
    start_f = int(window["start_t"] * fps)
    end_f = min(int(window["end_t"] * fps), int(info.total_frames))

    config = M.pitch_config(preset)
    vertices_cm = np.array(config.vertices, dtype=np.float32)
    probes = geometry.probe_points(info.width, info.height)
    length_m = preset.pitch_length_cm / 100.0
    width_m = preset.pitch_width_cm / 100.0

    tracker = tracking.make(minimum_consecutive_frames=2)
    tracker.reset()

    raw: Dict[int, Dict[str, Any]] = {}    # track_id → 원시 관측
    crops_for_fit: List[np.ndarray] = []
    frames_ok = frames_failed = 0

    frames = sv.get_video_frames_generator(
        source_path=video_path, stride=stride, start=start_f, end=end_f)
    n_expected = max(1, (end_f - start_f) // stride)
    if show_progress:
        from tqdm.auto import tqdm
        frames = tqdm(frames, total=n_expected, desc=f"analyze {window['window_id']}")

    for i, frame in enumerate(frames):
        t = (start_f + i * stride) / fps

        # --- 호모그래피 ---
        kp = sv.KeyPoints.from_ultralytics(
            mdl.field(frame, conf=preset.det_conf, verbose=False)[0])
        kp_conf = geometry.kp_confidence(kp)
        H, _, _ = geometry.solve_homography(
            kp.xy[0] if kp.xy is not None and len(kp.xy) else None,
            kp_conf[0] if kp_conf is not None and len(kp_conf) else None,
            vertices_cm, preset.kp_conf, preset.min_keypoints)
        if H is None or not geometry.is_sane(H, probes):
            frames_failed += 1
            continue
        frames_ok += 1

        # --- 탐지 + 추적 ---
        det = sv.Detections.from_ultralytics(
            mdl.player(frame, conf=preset.det_conf,
                       imgsz=preset.det_imgsz, verbose=False)[0])
        if len(det):
            det = det[det.class_id != mdl.id_of("ball")]
        if len(det):
            det = det.with_nms(threshold=preset.nms_threshold, class_agnostic=True)
        if not len(det):
            continue
        det = tracker.update(det)
        if not len(det) or det.tracker_id is None:
            continue

        # --- 화면 좌표 → 경기장 좌표 ---
        anchors = det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        xy_m = geometry.to_pitch_m(H, anchors)

        for k in range(len(det)):
            tid = int(det.tracker_id[k])
            cid = int(det.class_id[k])
            role = ("goalkeeper" if cid == mdl.id_of("goalkeeper")
                    else "referee" if cid == mdl.id_of("referee") else "player")
            rec = raw.setdefault(tid, {"track_id": tid, "role": role,
                                       "t": [], "xy": [], "crops": [], "crop_t": []})
            rec["t"].append(t)
            rec["xy"].append([float(xy_m[k, 0]), float(xy_m[k, 1])])

            # 썸네일·팀분류용 크롭은 드물게만 (메모리·속도)
            if i % CROP_STRIDE == 0:
                crop = sv.crop_image(frame, det.xyxy[k])
                if crop is not None and crop.size:
                    if len(rec["crops"]) < THUMBS_PER_TRACK * 4:
                        rec["crops"].append(crop)
                        rec["crop_t"].append(t)
                    if role == "player":
                        crops_for_fit.append(crop)

    # --- 팀 분류 ---
    team_quality = None
    clf = None
    if len(crops_for_fit) >= 10:
        try:
            clf = teams.TeamClassifier().fit(crops_for_fit)
            team_quality = clf.quality(crops_for_fit)
        except Exception:
            clf = None

    for rec in raw.values():
        rec["team"] = None
        if clf is not None and rec["role"] == "player" and rec["crops"]:
            votes = clf.predict(rec["crops"])
            votes = votes[votes >= 0]
            if len(votes):
                rec["team"] = int(np.bincount(votes).argmax())

    # --- 스무딩 ---
    tracks: List[Dict[str, Any]] = []
    for rec in raw.values():
        s = smooth.smooth_track(rec["t"], rec["xy"], preset, length_m, width_m)
        if s["n_kept"] < 2:
            continue
        tracks.append({**rec, **s})

    # 골키퍼는 색으로 못 가른다. 팀 무게중심으로 배정한다.
    _assign_gk(tracks)

    direction = _attack_direction(tracks, length_m)

    return {
        "window": window,
        "tracker_backend": tracker.backend,
        "tracks": tracks,
        "frames_ok": frames_ok,
        "frames_failed": frames_failed,
        "team_quality": team_quality,
        "attack_direction": direction,
        "pitch": {"length_m": length_m, "width_m": width_m},
    }


def _assign_gk(tracks: List[Dict[str, Any]]) -> None:
    """골키퍼 트랙을 가까운 팀 무게중심에 배정한다 (제자리 수정)."""
    players = [tr for tr in tracks if tr["role"] == "player" and tr["team"] is not None]
    gks = [tr for tr in tracks if tr["role"] == "goalkeeper"]
    if not players or not gks:
        return
    p_xy = np.stack([tr["xy"].mean(axis=0) for tr in players])
    p_team = np.array([tr["team"] for tr in players])
    g_xy = np.stack([tr["xy"].mean(axis=0) for tr in gks])
    for tr, team in zip(gks, teams.assign_goalkeepers(g_xy, p_xy, p_team)):
        if team >= 0:
            tr["team"] = int(team)


def normalize_direction(result: Dict[str, Any]) -> bool:
    """팀 0 이 항상 +x 로 공격하도록 좌표를 뒤집는다.

    전·후반이 섞이면 공격 방향이 반대가 되어 구간끼리 히트맵을 겹쳐 볼 수 없다.
    방향을 모르면 뒤집지 않고 그대로 둔다 (잘못 뒤집는 쪽이 더 나쁘다).

    Returns: 실제로 뒤집었는지 여부.
    """
    d = result["attack_direction"]
    if d["team0_attacks_positive"] is not False:
        return False
    L, W = result["pitch"]["length_m"], result["pitch"]["width_m"]
    for tr in result["tracks"]:
        tr["xy"] = np.stack([L - tr["xy"][:, 0], W - tr["xy"][:, 1]], axis=1)
    d["team0_attacks_positive"] = True
    d["flipped"] = True
    return True


def build_metrics(result: Dict[str, Any], preset) -> Dict[str, Any]:
    """정제된 트랙 → 트랙별·팀별 지표 + 썸네일."""
    L, W = result["pitch"]["length_m"], result["pitch"]["width_m"]
    tracks = result["tracks"]

    by_track, thumbs = [], {}
    for tr in tracks:
        by_track.append(metrics.track_metrics(tr, preset, L, W))
        picks = tr["crops"][:1] + tr["crops"][len(tr["crops"]) // 2:][:1]
        enc = [e for e in (_encode_thumb(c) for c in picks) if e]
        thumbs[tr["track_id"]] = enc[:THUMBS_PER_TRACK]

    pos = result["attack_direction"]["team0_attacks_positive"]
    by_team = []
    for team in (0, 1):
        attacks_pos = True if pos is None else (pos if team == 0 else not pos)
        m = metrics.team_metrics(tracks, team, attacks_pos, preset, L, W)
        if m:
            by_team.append(m)

    return {"metrics_by_track": by_track, "metrics_by_team": by_team, "thumbs": thumbs}
