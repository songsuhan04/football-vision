"""Phase 0 진단 리포트 — 이 영상으로 프로젝트가 가능한지를 숫자로 판정한다.

여기서 나오는 판정이 다음 단계 진행 여부를 결정한다. 낙관적으로 포장하지 않는다.
"""

from typing import Any, Dict


def _bar(ratio: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(1.0, ratio)) * width))
    return "█" * filled + "·" * (width - filled)


def verdict(summary: Dict[str, Any], preset) -> Dict[str, Any]:
    """채택 구간 수와 키포인트율로 다음 행동을 정한다."""
    n = summary["n_analyzed"]
    kp_rate = summary["keypoint_ok_rate"]
    broadcast = preset.name == "broadcast"

    if broadcast and kp_rate < 0.30:
        return {
            "level": "STOP",
            "headline": "키포인트 탐지율이 비정상적으로 낮다",
            "detail": (
                f"중계 영상은 이 모델의 학습 도메인이라 {kp_rate:.0%}는 나올 수 없는 값이다.\n"
                "영상 문제가 아니라 코드 문제일 가능성이 높다.\n"
                "→ roboflow/sports 공개 샘플 클립(0-A)으로 먼저 돌려 파이프라인을 검증할 것."
            ),
        }
    if not broadcast and kp_rate < 0.20:
        return {
            "level": "STOP",
            "headline": "화각 부족 — 자동 좌표 변환 불가",
            "detail": (
                f"키포인트 통과 프레임이 {kp_rate:.0%}뿐이다. 경기장 랜드마크가 화면에 거의 없다.\n"
                "선택지: (a) 넓은 구간만 소수 채택  (b) 키포인트 모델 파인튜닝  (c) 재촬영\n"
                "→ 셋 중 무엇을 할지 결정이 필요하다."
            ),
        }
    if n == 0:
        return {
            "level": "STOP",
            "headline": "채택 구간이 하나도 없다",
            "detail": "필터를 완화하거나(min_window_s, min_players) 다른 영상이 필요하다.",
        }
    if n <= 2:
        return {
            "level": "WARN",
            "headline": f"채택 구간이 {n}개뿐 — 데모가 빈약해진다",
            "detail": (
                "필터 완화 후 재측정: preset.with_(min_window_s=6.0, min_players=5)\n"
                "그래도 적으면 더 긴 영상(풀경기 롱샷 다수)으로 교체할 것.\n"
                "하이라이트는 짧은 컷과 리플레이 비중이 높아 채택률이 낮게 나온다."
            ),
        }
    if not broadcast and kp_rate < 0.60:
        return {
            "level": "WARN",
            "headline": f"키포인트율 {kp_rate:.0%} — 광학 흐름 보강이 필요하다",
            "detail": "Phase 0.5의 H 전파(propagate)를 켜서 결손 구간을 메울 것.",
        }
    return {
        "level": "GO",
        "headline": f"채택 구간 {n}개 / {summary['analyzed_s']:.0f}초 — 진행 가능",
        "detail": "Phase 1(구간 루프 + 데이터 계약)으로 넘어간다.",
    }


def print_summary(scan_result: Dict[str, Any], window_result: Dict[str, Any], preset) -> None:
    s = window_result["summary"]
    v = scan_result["video"]

    print("=" * 62)
    print(f"  Phase 0 진단  |  프리셋: {s['preset']}")
    print("=" * 62)
    print(f"  영상      {v['width']}x{v['height']}  {v['fps']:.1f}fps  {v['duration_s']:.0f}초")
    print(f"  스캔      {s['scan_frames']}프레임 / {s['scanned_s']:.0f}초 "
          f"(@{preset.scan_fps}fps)")
    print()
    print("  ── 프레임 품질 " + "─" * 44)
    print(f"  키포인트 {preset.min_keypoints}개 이상   {_bar(s['keypoint_ok_rate'])} {s['keypoint_ok_rate']:6.1%}")
    print(f"  호모그래피 성립     {_bar(s['homography_ok_rate'])} {s['homography_ok_rate']:6.1%}")
    print(f"  화면 내 평균 선수   {s['avg_players']:.1f}명  (필터 기준 {preset.min_players}명)")
    if s["bridged_frames"]:
        print(f"  이어붙인 결손 프레임 {s['bridged_frames']}개")
    print()
    print("  ── 분석 가능 구간 " + "─" * 41)
    print(f"  채택 {s['n_analyzed']}개 / 탈락 {s['n_rejected']}개")
    print(f"  분석 가능 시간      {_bar(s['coverage'])} {s['analyzed_s']:.0f}초 ({s['coverage']:.1%})")
    print()

    for w in window_result["window_index"]:
        mark = "✅" if w["status"] == "analyzed" else "  "
        span = f"{w['start_t']:7.1f}~{w['end_t']:7.1f}s ({w['duration_s']:5.1f}s)"
        if w["status"] == "analyzed":
            print(f"  {mark} {w['window_id']:>4}  {span}  "
                  f"kp {w['avg_keypoints']:.1f} · 선수 {w['avg_players']:.1f}명")
        else:
            print(f"  {mark} {w['window_id']:>4}  {span}  {w['reason']}")

    ver = verdict(s, preset)
    icon = {"GO": "✅", "WARN": "⚠️ ", "STOP": "🛑"}[ver["level"]]
    print()
    print("=" * 62)
    print(f"  {icon} {ver['level']} — {ver['headline']}")
    print("-" * 62)
    for line in ver["detail"].split("\n"):
        print(f"  {line}")
    print("=" * 62)


def plot_profile(scan_result: Dict[str, Any], window_result: Dict[str, Any], preset):
    """시간축 프로파일. 채택 구간은 초록, 탈락 구간은 붉게 칠한다.

    Colab 기본 환경에 한글 폰트가 없어 그래프 라벨은 영문으로 둔다.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = scan_result["rows"]
    t = np.array([r["t"] for r in rows])
    kp = np.array([r["n_keypoints"] for r in rows])
    pl = np.array([r["n_players"] for r in rows])
    sp = np.array([r["camera_speed"] if r["camera_speed"] is not None else np.nan
                   for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    for ax in axes:
        for w in window_result["window_index"]:
            ax.axvspan(w["start_t"], w["end_t"],
                       color="#2ecc71" if w["status"] == "analyzed" else "#e74c3c",
                       alpha=0.13, lw=0)

    axes[0].plot(t, kp, lw=1, color="#2c3e50")
    axes[0].axhline(preset.min_keypoints, ls="--", color="#e74c3c", lw=1)
    axes[0].set_ylabel("keypoints")

    axes[1].plot(t, pl, lw=1, color="#2c3e50")
    axes[1].axhline(preset.min_players, ls="--", color="#e74c3c", lw=1)
    axes[1].set_ylabel("players")

    axes[2].plot(t, sp, lw=1, color="#2c3e50")
    axes[2].axhline(preset.max_camera_speed, ls="--", color="#e74c3c", lw=1)
    axes[2].set_yscale("symlog", linthresh=10)
    axes[2].set_ylabel("camera [m/s]")
    axes[2].set_xlabel("time [s]")

    s = window_result["summary"]
    fig.suptitle(
        f"Phase 0 profile — {s['preset']} | "
        f"{s['n_analyzed']} windows / {s['analyzed_s']:.0f}s accepted ({s['coverage']:.0%})",
        fontsize=12)
    fig.tight_layout()
    return fig
