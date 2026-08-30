"""좌표 스무딩과 이상치 제거.

프레임별 호모그래피는 카메라 팬·손떨림 탓에 미세하게 흔들린다.
그대로 두면 정지한 선수도 2D 에서 떨리고, 속도 지표가 통째로 오염된다.

호모그래피 행렬 자체를 평균내지 않는다 — 사영변환은 선형 평균이 성립하지 않는다.
대신 변환된 **경기장 좌표**를 트랙별로 스무딩한다.
"""

from typing import List, Tuple

import numpy as np

# 사람이 낼 수 있는 최고 속도의 상한 [m/s]. 이를 넘는 이동은 좌표 튐으로 본다.
# 우사인 볼트가 12.4 m/s 이므로 축구 선수에게 이 값은 확실한 이상치다.
MAX_HUMAN_SPEED = 13.0


def reject_jumps(t: np.ndarray, xy: np.ndarray,
                 max_speed: float = MAX_HUMAN_SPEED) -> np.ndarray:
    """물리적으로 불가능한 점프를 제거한다. 살아남을 점의 불리언 마스크를 반환.

    앞 점 대비 속도가 상한을 넘으면 버린다. 버린 점은 기준이 되지 않으므로
    한 번 튄 뒤 제자리로 돌아오는 경우에도 원래 궤적이 살아남는다.
    """
    n = len(t)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep

    anchor = 0
    for i in range(1, n):
        dt = t[i] - t[anchor]
        if dt <= 0:
            keep[i] = False
            continue
        if np.linalg.norm(xy[i] - xy[anchor]) / dt > max_speed:
            keep[i] = False
        else:
            anchor = i
    return keep


def moving_average(xy: np.ndarray, window: int) -> np.ndarray:
    """가장자리를 잃지 않는 이동평균 (양끝은 있는 만큼만 평균)."""
    n = len(xy)
    if n == 0 or window <= 1:
        return xy.copy()
    half = window // 2
    out = np.empty_like(xy, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = xy[lo:hi].mean(axis=0)
    return out


def speeds(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """각 표본의 순간 속도 [m/s]. 중앙차분, 양끝은 전진/후진 차분."""
    n = len(t)
    if n < 2:
        return np.zeros(n)
    v = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        dt = t[hi] - t[lo]
        v[i] = 0.0 if dt <= 0 else float(np.linalg.norm(xy[hi] - xy[lo]) / dt)
    return v


def clip_to_pitch(xy: np.ndarray, length_m: float, width_m: float,
                  margin_m: float = 5.0) -> np.ndarray:
    """경기장 밖으로 조금 벗어난 좌표를 경계로 당긴다.

    터치라인 밖의 선수(스로인 대기 등)는 실재하므로 margin 만큼은 허용한다.
    """
    out = xy.copy()
    out[:, 0] = np.clip(out[:, 0], -margin_m, length_m + margin_m)
    out[:, 1] = np.clip(out[:, 1], -margin_m, width_m + margin_m)
    return out


def smooth_track(t: List[float], xy: List[Tuple[float, float]], preset,
                 length_m: float, width_m: float) -> dict:
    """한 트랙의 원시 좌표 시계열 → 정제된 좌표·속도.

    Returns:
        {"t":…, "xy":…, "speed_ms":…, "n_raw":…, "n_kept":…}
        전부 numpy 배열. 남은 점이 2개 미만이면 빈 배열.
    """
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float)
    n_raw = len(t)
    if n_raw == 0:
        empty = np.empty(0)
        return {"t": empty, "xy": np.empty((0, 2)), "speed_ms": empty,
                "n_raw": 0, "n_kept": 0}

    order = np.argsort(t)
    t, xy = t[order], xy[order]

    keep = reject_jumps(t, xy)
    t, xy = t[keep], xy[keep]
    if len(t) < 2:
        return {"t": t, "xy": xy, "speed_ms": np.zeros(len(t)),
                "n_raw": n_raw, "n_kept": len(t)}

    xy = clip_to_pitch(xy, length_m, width_m)
    xy = moving_average(xy, preset.h_smooth_window)
    return {"t": t, "xy": xy, "speed_ms": speeds(t, xy),
            "n_raw": n_raw, "n_kept": len(t)}
