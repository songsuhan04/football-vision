"""프레임 프로파일 → 분석 가능 구간(analyzable window).

컷 전환과 화각 부족은 겉보기엔 다른 문제지만 결과는 같다 —
**좌표 변환을 믿을 수 없는 구간이 생긴다.** 그래서 하나의 판정으로 통합한다.

탈락 구간과 그 이유를 전부 남긴다. 32개 중 6개만 분석됐다는 사실을 숨기면
결과 전체의 신뢰가 무너지기 때문이다.
"""

from typing import Any, Dict, List, Optional, Tuple

import math

# 판정 실패 사유 (사람이 읽을 라벨)
R_NO_HOMOGRAPHY = "호모그래피 실패"
R_FEW_KEYPOINTS = "키포인트 부족"
R_FEW_PLAYERS = "선수 부족 (클로즈업·리플레이 추정)"
R_FAST_CAMERA = "카메라 급변 (컷·급속 팬 추정)"
R_TOO_SHORT = "구간이 짧음"


def _row_verdict(row: Dict[str, Any], preset) -> Optional[str]:
    """이 프레임이 쓸 만한가. 쓸 만하면 None, 아니면 실패 사유."""
    if row["n_keypoints"] < preset.min_keypoints:
        return R_FEW_KEYPOINTS
    if not row["h_ok"]:
        return R_NO_HOMOGRAPHY
    if row["n_players"] < preset.min_players:
        return R_FEW_PLAYERS
    speed = row["camera_speed"]
    # 첫 유효 프레임은 비교 대상이 없어 NaN이다. 이건 실패가 아니다.
    if speed is not None and not math.isnan(speed) and speed > preset.max_camera_speed:
        return R_FAST_CAMERA
    return None


def _bridge(verdicts: List[Optional[str]], max_gap: int) -> List[Optional[str]]:
    """앞뒤가 정상인 짧은 결손을 메워 구간이 쪼개지지 않게 한다."""
    out = list(verdicts)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is None:
            i += 1
            continue
        j = i
        while j < n and out[j] is not None:
            j += 1
        gap = j - i
        has_left = i > 0 and out[i - 1] is None
        has_right = j < n and out[j] is None
        if gap <= max_gap and has_left and has_right:
            for k in range(i, j):
                out[k] = None
        i = j
    return out


def _runs(verdicts: List[Optional[str]]) -> List[Tuple[int, int, bool]]:
    """연속 구간을 (시작 idx, 끝 idx(포함), 정상 여부) 목록으로."""
    runs = []
    if not verdicts:
        return runs
    start = 0
    cur_ok = verdicts[0] is None
    for i in range(1, len(verdicts)):
        ok = verdicts[i] is None
        if ok != cur_ok:
            runs.append((start, i - 1, cur_ok))
            start, cur_ok = i, ok
    runs.append((start, len(verdicts) - 1, cur_ok))
    return runs


def _dominant_reason(verdicts: List[Optional[str]], lo: int, hi: int) -> str:
    """탈락 구간에서 가장 많이 나온 실패 사유."""
    counts: Dict[str, int] = {}
    for v in verdicts[lo:hi + 1]:
        if v:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return R_NO_HOMOGRAPHY
    return max(counts.items(), key=lambda kv: kv[1])[0]


def find_windows(scan_result: Dict[str, Any], preset) -> Dict[str, Any]:
    """스캔 결과에서 분석 가능 구간을 뽑고, 탈락 구간은 이유와 함께 기록한다."""
    rows: List[Dict[str, Any]] = scan_result["rows"]
    dt: float = scan_result["dt"]

    raw = [_row_verdict(r, preset) for r in rows]
    bridged = _bridge(raw, preset.bridge_bad_frames)
    n_bridged = sum(1 for a, b in zip(raw, bridged) if a is not None and b is None)

    index: List[Dict[str, Any]] = []
    accepted, rejected = 0, 0

    for lo, hi, ok in _runs(bridged):
        start_t = rows[lo]["t"]
        end_t = rows[hi]["t"] + dt      # 마지막 샘플이 대표하는 구간까지 포함
        dur = end_t - start_t

        if ok and dur >= preset.min_window_s:
            accepted += 1
            seg = rows[lo:hi + 1]
            speeds = [r["camera_speed"] for r in seg
                      if r["camera_speed"] is not None and not math.isnan(r["camera_speed"])]
            index.append({
                "window_id": f"w{accepted:02d}",
                "start_t": round(start_t, 2), "end_t": round(end_t, 2),
                "duration_s": round(dur, 2),
                "status": "analyzed",
                "avg_keypoints": round(sum(r["n_keypoints"] for r in seg) / len(seg), 1),
                "avg_players": round(sum(r["n_players"] for r in seg) / len(seg), 1),
                "avg_camera_speed": round(sum(speeds) / len(speeds), 1) if speeds else None,
                "n_scan_frames": len(seg),
            })
        else:
            rejected += 1
            if ok:
                detail = f"{R_TOO_SHORT} — {dur:.1f}초 < {preset.min_window_s:.0f}초"
            else:
                reason = _dominant_reason(bridged, lo, hi)
                seg = rows[lo:hi + 1]
                avg_kp = sum(r["n_keypoints"] for r in seg) / len(seg)
                avg_pl = sum(r["n_players"] for r in seg) / len(seg)
                detail = f"{reason} — 키포인트 평균 {avg_kp:.1f}개, 선수 평균 {avg_pl:.1f}명"
            index.append({
                "window_id": f"x{rejected:02d}",
                "start_t": round(start_t, 2), "end_t": round(end_t, 2),
                "duration_s": round(dur, 2),
                "status": "rejected",
                "reason": detail,
            })

    index.sort(key=lambda w: w["start_t"])

    scanned_s = len(rows) * dt
    analyzed_s = sum(w["duration_s"] for w in index if w["status"] == "analyzed")
    n_kp_ok = sum(1 for r in rows if r["n_keypoints"] >= preset.min_keypoints)
    n_h_ok = sum(1 for r in rows if r["h_ok"])

    summary = {
        "preset": preset.name,
        "scanned_s": round(scanned_s, 1),
        "scan_frames": len(rows),
        "keypoint_ok_rate": round(n_kp_ok / len(rows), 3) if rows else 0.0,
        "homography_ok_rate": round(n_h_ok / len(rows), 3) if rows else 0.0,
        "avg_players": round(sum(r["n_players"] for r in rows) / len(rows), 1) if rows else 0.0,
        "bridged_frames": n_bridged,
        "n_analyzed": accepted,
        "n_rejected": rejected,
        "analyzed_s": round(analyzed_s, 1),
        "coverage": round(analyzed_s / scanned_s, 3) if scanned_s else 0.0,
    }
    return {"summary": summary, "window_index": index}
