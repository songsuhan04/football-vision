"""유니폼 색으로 팀 분류.

**잔디를 마스킹하지 않는다.** 직관과 반대라 이유를 남긴다:

크롭이 작다(중계 영상에서 평균 59x27px). 잔디 픽셀을 빼면 상체에 30~200픽셀만
남아 특징이 노이즈에 묻힌다. 실측으로 커버리지 51%, 정확도 57% — 동전던지기였다.

마스킹을 안 하면 잔디는 **모든 크롭에 공통으로 실리는 상수 오프셋**이라
KMeans 가 알아서 무시한다. 크롭 사이의 차이인 유니폼 색만 군집에 반영된다.
같은 데이터에서 커버리지 100%, 정확도 99.3% (SigLIP+UMAP 의 99.6% 와 동급).

roboflow/sports 의 SigLIP+UMAP 방식이 더 정확하지만 transformers 를 끌고 오고
CPU 에서 느리다. 정확도 차이가 0.3%p 라 가벼운 쪽을 기본으로 쓴다.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# 상체 크롭 비율 — 머리·다리를 피해 유니폼 몸통만 본다
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55
TORSO_LEFT, TORSO_RIGHT = 0.25, 0.75

MIN_CROP_H, MIN_CROP_W = 10, 5


def torso_feature(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """선수 크롭 → 유니폼 색 특징 5차원. 못 뽑으면 None.

    정규화 BGR(조명 불변 색도) 3차원 + 채도·명도 2차원.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < MIN_CROP_H or w < MIN_CROP_W:
        return None

    torso = crop_bgr[int(TORSO_TOP * h):int(TORSO_BOTTOM * h),
                     int(TORSO_LEFT * w):int(TORSO_RIGHT * w)]
    if torso.size == 0:
        torso = crop_bgr        # 크롭이 너무 작으면 전체를 쓴다

    px = torso.astype(np.float32).reshape(-1, 3)
    chroma = (px / (px.sum(axis=1, keepdims=True) + 1e-6)).mean(axis=0)

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    return np.concatenate([chroma,
                           [hsv[..., 1].mean() / 255.0, hsv[..., 2].mean() / 255.0]])


class TeamClassifier:
    """KMeans(k=2) 로 두 팀을 가른다."""

    def __init__(self):
        self.km = None
        self.mu = None
        self.sd = None
        self.fitted = False

    def _matrix(self, crops: List[np.ndarray]):
        feats, idx = [], []
        for i, c in enumerate(crops):
            f = torso_feature(c)
            if f is not None:
                feats.append(f)
                idx.append(i)
        return (np.stack(feats) if feats else np.empty((0, 5))), idx

    def fit(self, crops: List[np.ndarray]) -> "TeamClassifier":
        from sklearn.cluster import KMeans

        X, _ = self._matrix(crops)
        if len(X) < 10:
            raise RuntimeError(
                f"팀 분류에 쓸 크롭이 {len(X)}개뿐이다. 선수 탐지가 거의 안 됐다.")
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        self.km = KMeans(n_clusters=2, n_init=10, random_state=0).fit((X - self.mu) / self.sd)
        self.fitted = True
        return self

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """각 크롭의 팀 번호(0/1). 판정 불가는 -1."""
        if not self.fitted:
            raise RuntimeError("fit() 을 먼저 호출할 것")
        out = np.full(len(crops), -1, dtype=int)
        X, idx = self._matrix(crops)
        if len(X):
            out[idx] = self.km.predict((X - self.mu) / self.sd)
        return out

    def quality(self, crops: List[np.ndarray]) -> Dict[str, Any]:
        """분류를 믿어도 되는지 판단할 지표.

        실루엣만 보면 안 된다 — 한 군집에 1명, 나머지 전부인 퇴화 해가
        실루엣 0.87 을 받는 것을 실제로 관찰했다. **균형**을 함께 본다.
        """
        from sklearn.metrics import silhouette_score

        X, idx = self._matrix(crops)
        if len(X) < 10:
            return {"coverage": 0.0, "balance": 0.0, "silhouette": 0.0, "ok": False}
        labels = self.km.predict((X - self.mu) / self.sd)
        counts = np.bincount(labels, minlength=2)
        balance = float(counts.min() / counts.sum())
        sil = float(silhouette_score((X - self.mu) / self.sd, labels)) \
            if len(set(labels)) > 1 else 0.0
        coverage = len(idx) / max(1, len(crops))
        return {"coverage": round(coverage, 3), "balance": round(balance, 3),
                "silhouette": round(sil, 3),
                # 두 팀이 비슷한 수로 잡혀야 정상이다. 한쪽이 15% 미만이면 의심.
                "ok": bool(balance >= 0.15 and coverage >= 0.8)}


def assign_goalkeepers(gk_xy: np.ndarray, players_xy: np.ndarray,
                       players_team: np.ndarray) -> np.ndarray:
    """골키퍼를 가까운 팀 무게중심에 배정한다.

    골키퍼는 유니폼이 양 팀 모두와 달라 색으로는 못 가른다.
    roboflow/sports 예제와 같은 방식.
    """
    if len(gk_xy) == 0:
        return np.empty(0, dtype=int)
    out = []
    for xy in gk_xy:
        best, best_d = -1, float("inf")
        for t in (0, 1):
            m = players_team == t
            if not m.any():
                continue
            d = float(np.linalg.norm(players_xy[m].mean(axis=0) - xy))
            if d < best_d:
                best, best_d = t, d
        out.append(best)
    return np.array(out, dtype=int)
