"""정제된 좌표 → 히트맵·궤적·속도·팀 형태 지표.

짧은 구간(10~40초)에 90분용 지표를 쓰면 무의미하다.
총 이동거리 대신 순간 속도·스프린트·침투 벡터를, 정적 포메이션 대신
팀 폭·수비라인 높이의 **시계열**을 낸다.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

MS_TO_KMH = 3.6


def heatmap(xy: np.ndarray, length_m: float, width_m: float,
            cols: int, rows: int) -> List[List[int]]:
    """경기장을 바둑판으로 나눠 셀별 체류 표본 수를 센다."""
    grid = np.zeros((rows, cols), dtype=int)
    if len(xy) == 0:
        return grid.tolist()
    cx = np.clip((xy[:, 0] / length_m * cols).astype(int), 0, cols - 1)
    cy = np.clip((xy[:, 1] / width_m * rows).astype(int), 0, rows - 1)
    for r, c in zip(cy, cx):
        grid[r, c] += 1
    return grid.tolist()


def hull_area(xy: np.ndarray) -> float:
    """점들이 감싸는 볼록껍질 넓이 [m^2]. 활동 범위·팀 컴팩트니스에 쓴다."""
    if len(xy) < 3:
        return 0.0
    pts = np.asarray(xy, dtype=np.float32).reshape(-1, 1, 2)
    return float(cv2.contourArea(cv2.convexHull(pts)))


def find_sprints(t: np.ndarray, speed_ms: np.ndarray, preset) -> List[Dict[str, Any]]:
    """임계 속도를 넘는 연속 구간을 스프린트로 묶는다."""
    if len(t) < 2:
        return []
    thr = preset.sprint_kmh / MS_TO_KMH
    fast = speed_ms >= thr
    out: List[Dict[str, Any]] = []
    i, n = 0, len(fast)
    while i < n:
        if not fast[i]:
            i += 1
            continue
        j = i
        while j < n and fast[j]:
            j += 1
        dur = float(t[j - 1] - t[i])
        if dur >= preset.min_sprint_s:
            out.append({
                "start_t": round(float(t[i]), 2),
                "end_t": round(float(t[j - 1]), 2),
                "duration_s": round(dur, 2),
                "peak_kmh": round(float(speed_ms[i:j].max() * MS_TO_KMH), 1),
            })
        i = j
    return out


def zone_share(xy: np.ndarray, length_m: float) -> Dict[str, float]:
    """수비/중원/공격 3분할 체류 비율. 좌표는 공격 방향이 +x 로 정규화된 상태."""
    if len(xy) == 0:
        return {"def_third": 0.0, "mid_third": 0.0, "att_third": 0.0}
    third = length_m / 3.0
    x = xy[:, 0]
    n = len(x)
    return {
        "def_third": round(float((x < third).sum() / n), 3),
        "mid_third": round(float(((x >= third) & (x < 2 * third)).sum() / n), 3),
        "att_third": round(float((x >= 2 * third).sum() / n), 3),
    }


def track_metrics(track: Dict[str, Any], preset,
                  length_m: float, width_m: float,
                  traj_stride: int = 2) -> Dict[str, Any]:
    """트랙 하나의 지표.

    선수 단위 지표는 웹에서 track_ids 를 합산해 만든다. 여기서는 트랙 단위까지만.
    """
    t, xy, v = track["t"], track["xy"], track["speed_ms"]
    if len(t) == 0:
        return {"track_id": track["track_id"], "n_samples": 0}

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([0.0])

    return {
        "track_id": track["track_id"],
        "team": track.get("team"),
        "role": track.get("role"),
        "first_t": round(float(t[0]), 2),
        "last_t": round(float(t[-1]), 2),
        "n_samples": int(len(t)),
        "heatmap": heatmap(xy, length_m, width_m, preset.grid_cols, preset.grid_rows),
        # 궤적은 표시용이라 솎아낸다. SVG path 가 과하게 무거워지는 것을 막는다.
        "trajectory": [[round(float(a), 2), round(float(b), 2), round(float(c), 2)]
                       for a, b, c in zip(t[::traj_stride],
                                          xy[::traj_stride, 0], xy[::traj_stride, 1])],
        "speed_series": [[round(float(a), 2), round(float(b) * MS_TO_KMH, 1)]
                         for a, b in zip(t[::traj_stride], v[::traj_stride])],
        "top_speed_kmh": round(float(v.max() * MS_TO_KMH), 1),
        "avg_speed_kmh": round(float(v.mean() * MS_TO_KMH), 1),
        "sprints": find_sprints(t, v, preset),
        "distance_m": round(float(steps.sum()), 1),
        "displacement": {
            "from": [round(float(xy[0, 0]), 1), round(float(xy[0, 1]), 1)],
            "to": [round(float(xy[-1, 0]), 1), round(float(xy[-1, 1]), 1)],
            "net_m": round(float(np.linalg.norm(xy[-1] - xy[0])), 1),
        },
        "zone_share": zone_share(xy, length_m),
        "coverage_area_m2": round(hull_area(xy), 1),
    }


def team_metrics(tracks: List[Dict[str, Any]], team: int, attacks_positive: bool,
                 preset, length_m: float, width_m: float,
                 series_stride: int = 2) -> Optional[Dict[str, Any]]:
    """한 팀의 형태 시계열.

    Args:
        attacks_positive: 이 팀이 +x 방향으로 공격하는가.
            수비라인 '높이'는 자기 골대로부터의 거리이므로 방향에 따라 계산이 뒤집힌다.
    """
    members = [tr for tr in tracks if tr.get("team") == team and len(tr["t"]) > 0]
    if not members:
        return None

    # 같은 프레임에서 뽑았으므로 t 값이 정렬돼 있다. 시각 → 좌표 조회표를 만든다.
    by_time: Dict[float, List[np.ndarray]] = {}
    for tr in members:
        for tt, pt in zip(tr["t"], tr["xy"]):
            by_time.setdefault(round(float(tt), 2), []).append(pt)

    times = sorted(by_time)[::series_stride]
    centroid, widths, depths, def_line, compact, counts = [], [], [], [], [], []

    for tt in times:
        pts = np.stack(by_time[tt])
        if len(pts) < 2:
            continue
        c = pts.mean(axis=0)
        centroid.append([tt, round(float(c[0]), 1), round(float(c[1]), 1)])
        widths.append([tt, round(float(pts[:, 1].max() - pts[:, 1].min()), 1)])
        depths.append([tt, round(float(pts[:, 0].max() - pts[:, 0].min()), 1)])
        compact.append([tt, round(hull_area(pts), 0)])
        counts.append(len(pts))

        # 최후방 4명(없으면 있는 만큼)의 평균 x → 자기 골대로부터의 거리
        k = min(4, len(pts))
        xs = np.sort(pts[:, 0])
        back = xs[:k].mean() if attacks_positive else xs[-k:].mean()
        height = back if attacks_positive else (length_m - back)
        def_line.append([tt, round(float(height), 1)])

    all_xy = np.concatenate([tr["xy"] for tr in members])

    return {
        "team": team,
        "attacks_positive": attacks_positive,
        "n_tracks": len(members),
        # 한 프레임에 동시에 잡힌 최대 인원. 11명이 아니라는 사실을 숨기지 않는다.
        "max_players_in_frame": int(max(counts)) if counts else 0,
        "avg_players_in_frame": round(float(np.mean(counts)), 1) if counts else 0.0,
        "heatmap": heatmap(all_xy, length_m, width_m, preset.grid_cols, preset.grid_rows),
        "centroid_series": centroid,
        "width_series": widths,
        "depth_series": depths,
        "def_line_series": def_line,
        "compactness_series": compact,
    }
