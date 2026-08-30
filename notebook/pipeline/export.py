"""analysis-bundle.json 조립.

웹(web/lib/analysis.ts)과 이 파일이 스키마를 공유한다. 한쪽만 바꾸면 깨진다.

지표는 **트랙 단위**로 내보낸다. 선수 라벨링은 웹에서 하므로,
웹이 track_ids 를 합산해 선수 지표를 만든다.
→ 라벨링을 바꿔도 Colab 을 다시 돌릴 필요가 없다.
"""

import json
import os
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def build_window(result: Dict[str, Any], built: Dict[str, Any], preset) -> Dict[str, Any]:
    """구간 하나의 결과 객체."""
    w = result["window"]
    tracks_out = []
    for tr in result["tracks"]:
        tracks_out.append({
            "track_id": tr["track_id"],
            "role": tr["role"],
            "team": tr["team"],
            "first_t": round(float(tr["t"][0]), 2),
            "last_t": round(float(tr["t"][-1]), 2),
            "n_samples": int(len(tr["t"])),
            "thumbs": built["thumbs"].get(tr["track_id"], []),
        })

    total = result["frames_ok"] + result["frames_failed"]
    warnings: List[str] = []
    if result["frames_failed"]:
        pct = result["frames_failed"] / total if total else 0
        warnings.append(
            f"{result['frames_failed']}/{total} 프레임({pct:.0%})에서 호모그래피 실패 — 해당 시각 제외됨")
    if result["attack_direction"]["method"] == "unknown":
        warnings.append("공격 방향 미확정(골키퍼 미탐지) — 다른 구간과 히트맵 비교 불가")
    q = result["team_quality"]
    if q is None:
        warnings.append("팀 분류 실패 — 모든 트랙의 team 이 null")
    elif not q["ok"]:
        warnings.append(
            f"팀 분류 신뢰도 낮음 (커버리지 {q['coverage']:.0%}, 균형 {q['balance']:.0%}) "
            "— 두 팀 유니폼 색이 비슷하거나 한쪽만 잡혔을 수 있다")

    return {
        "meta": {
            "window_id": w["window_id"],
            "label": w.get("label", ""),
            "start_t": w["start_t"], "end_t": w["end_t"],
            "duration_s": w["duration_s"],
            "fps_sampled": preset.analyze_fps,
            "pitch": {"length_m": result["pitch"]["length_m"],
                      "width_m": result["pitch"]["width_m"]},
            "grid": {"cols": preset.grid_cols, "rows": preset.grid_rows},
            "attack_direction": result["attack_direction"],
            "team_quality": q,
            "homography": {
                "frames_ok": result["frames_ok"],
                "frames_failed": result["frames_failed"],
            },
        },
        "tracks": tracks_out,
        "metrics_by_track": built["metrics_by_track"],
        "metrics_by_team": built["metrics_by_team"],
        "warnings": warnings,
    }


def build_bundle(video_meta: Dict[str, Any], window_index: List[Dict[str, Any]],
                 windows: List[Dict[str, Any]], preset,
                 source_profile: Optional[str] = None) -> Dict[str, Any]:
    """최상위 번들. 탈락 구간도 이유와 함께 전부 담는다."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_profile": source_profile or preset.name,
        "video": {
            "path": os.path.basename(video_meta["path"]),
            "duration_s": round(video_meta["duration_s"], 1),
            "fps": video_meta["fps"],
            "width": video_meta["width"], "height": video_meta["height"],
        },
        "window_index": window_index,   # 채택/탈락 전부 — 채택률을 숨기지 않는다
        "windows": windows,
    }


def save(bundle: Dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    return path


def summarize(bundle: Dict[str, Any]) -> None:
    """번들 내용을 사람이 읽게 출력한다."""
    idx = bundle["window_index"]
    n_ok = sum(1 for w in idx if w["status"] == "analyzed")
    print("=" * 62)
    print(f"  analysis-bundle  |  {bundle['video']['path']}  "
          f"({bundle['video']['duration_s']:.0f}초)")
    print("=" * 62)
    print(f"  구간: 채택 {n_ok} / 탈락 {len(idx) - n_ok}")
    print()
    for w in bundle["windows"]:
        m = w["meta"]
        n_tr = len(w["tracks"])
        teamed = sum(1 for t in w["tracks"] if t["team"] is not None)
        hom = m["homography"]
        print(f"  [{m['window_id']}] {m['start_t']:.1f}~{m['end_t']:.1f}s "
              f"({m['duration_s']:.1f}초)")
        print(f"      트랙 {n_tr}개 (팀 배정 {teamed}개) · "
              f"호모그래피 {hom['frames_ok']}/{hom['frames_ok']+hom['frames_failed']} 프레임")
        print(f"      팀 색 분리도 {m['team_separation']} · "
              f"공격방향 {m['attack_direction']['method']}")
        for tm in w["metrics_by_team"]:
            print(f"      팀 {tm['team']}: 트랙 {tm['n_tracks']}개 · "
                  f"동시 최대 {tm['max_players_in_frame']}명")
        for warn in w["warnings"]:
            print(f"      ⚠️  {warn}")
        print()
