"""호모그래피: 화면 픽셀 좌표 ↔ 경기장 미터 좌표.

이 프로젝트의 심장. 여기가 실패하면 히트맵도 궤적도 나오지 않는다.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

# 호모그래피가 물리적으로 말이 되는지 검사하는 한계값 [m].
# 경기장은 105x68 이므로, 화면 하단 기준점이 이 범위를 벗어나면 H가 망가진 것이다.
SANITY_LIMIT_M = 300.0

# RANSAC 재투영 허용 오차 [m]. 목적지 좌표계가 미터이므로 단위도 미터다.
RANSAC_REPROJ_M = 3.0


def solve_homography(
    kp_xy: np.ndarray,
    kp_conf: np.ndarray,
    vertices_cm: np.ndarray,
    kp_conf_thresh: float,
    min_keypoints: int,
) -> Tuple[Optional[np.ndarray], int, int]:
    """키포인트 대응에서 H를 구한다.

    Args:
        kp_xy: (N, 2) 화면 픽셀 좌표
        kp_conf: (N,) 키포인트별 신뢰도
        vertices_cm: (N, 2) 경기장 기준점 좌표 [cm] — kp_xy와 인덱스가 1:1 대응
        kp_conf_thresh: 이 값을 넘는 키포인트만 사용
        min_keypoints: 최소 대응점 개수 (4 미만이면 호모그래피 불가)

    Returns:
        (H, 사용한 키포인트 수, RANSAC 인라이어 수).
        실패 시 H는 None.

    Note:
        목적지 좌표는 **미터**다 (vertices_cm / 100).
    """
    if kp_xy is None or kp_conf is None or len(kp_xy) == 0:
        return None, 0, 0

    mask = kp_conf > kp_conf_thresh
    n_used = int(mask.sum())
    if n_used < max(4, min_keypoints):
        return None, n_used, 0

    src = kp_xy[mask].astype(np.float32)
    dst = (vertices_cm[mask].astype(np.float32) / 100.0)

    if n_used == 4:
        # 점이 딱 4개면 RANSAC이 걸러낼 여지가 없다. 최소해를 그대로 쓴다.
        H, inlier_mask = cv2.findHomography(src, dst, 0)
        n_inliers = 4 if H is not None else 0
    else:
        H, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_M
        )
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if H is None:
        return None, n_used, 0
    return H, n_used, n_inliers


def to_pitch_m(H: np.ndarray, xy_px: np.ndarray) -> np.ndarray:
    """화면 픽셀 좌표 (N,2) → 경기장 미터 좌표 (N,2)."""
    if xy_px is None or len(xy_px) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(xy_px, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float32))
    return out.reshape(-1, 2)


def probe_points(width: int, height: int) -> np.ndarray:
    """카메라 움직임 측정용 화면 기준점.

    화면 하단부만 쓴다. 상단은 지평선 너머로 매핑돼 값이 발산하기 때문이다.
    """
    xs = np.array([0.25, 0.5, 0.75]) * width
    ys = np.array([0.70, 0.85]) * height
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32)


def is_sane(H: np.ndarray, probes: np.ndarray) -> bool:
    """H가 물리적으로 말이 되는지 검사.

    키포인트가 잘못 잡히면 수학적으로는 풀리지만 좌표가 수 km 밖으로 날아간다.
    이런 H를 걸러내지 않으면 히트맵 전체가 오염된다.
    """
    try:
        m = to_pitch_m(H, probes)
    except cv2.error:
        return False
    if not np.all(np.isfinite(m)):
        return False
    return bool(np.all(np.abs(m) < SANITY_LIMIT_M))


def camera_speed(
    H_prev: np.ndarray, H_cur: np.ndarray, probes: np.ndarray, dt: float
) -> float:
    """두 프레임 사이 카메라가 훑고 지나간 속도 [m/s].

    화면의 같은 위치가 경기장 어디를 가리키는지 비교한다.
    컷이 나면 값이 폭발하고, 부드러운 팬이면 완만하다.
    """
    if dt <= 0:
        return 0.0
    a = to_pitch_m(H_prev, probes)
    b = to_pitch_m(H_cur, probes)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    return float(np.linalg.norm(b - a, axis=1).mean() / dt)


def color_signature(frame_bgr: np.ndarray) -> np.ndarray:
    """컷 감지 보조 신호용 HSV 히스토그램."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def color_similarity(h_prev: np.ndarray, h_cur: np.ndarray) -> float:
    """히스토그램 상관계수. 1에 가까울수록 같은 장면, 낮으면 컷 의심."""
    return float(cv2.compareHist(h_prev, h_cur, cv2.HISTCMP_CORREL))
