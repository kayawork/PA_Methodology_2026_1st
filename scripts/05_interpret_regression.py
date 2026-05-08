"""
05_interpret_regression.py
회귀결과 CSV를 읽어 다중회귀 해석 문장을 자동 생성한다.

실행 예시:
python scripts/05_interpret_regression.py
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ModelMeta:
    name: str
    dv: str
    focal_vars: tuple[str, ...] = ()
    control_vars: tuple[str, ...] = ()
    r2: Optional[float] = None
    adj_r2: Optional[float] = None
    f_p: Optional[float] = None


def p_to_text(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return "p < .01"
    if p < 0.05:
        return "p < .05"
    return f"p = {p:.3f}"


def beta_strength(beta_abs: float) -> str:
    if beta_abs >= 0.50:
        return "매우 큰 영향"
    if beta_abs >= 0.30:
        return "큰 영향"
    if beta_abs >= 0.10:
        return "중간 영향"
    return "작은 영향"


def explain_model(file_path: Path, meta: ModelMeta) -> str:
    report = StringIO()

    def write(line: str = "") -> None:
        print(line)
        report.write(line + "\n")

    if not file_path.exists():
        write(f"[{meta.name}] 파일 없음: {file_path}")
        return report.getvalue()

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df = df[df["변수"] != "(상수)"].copy()

    # 숫자형 강제 변환
    for col in ["B", "β", "p"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sig_df = df[df["p"] < 0.05].copy()
    nonsig_df = df[df["p"] >= 0.05].copy()
    focal_df = df[df["변수"].isin(meta.focal_vars)].copy()
    focal_sig_df = focal_df[focal_df["p"] < 0.05].copy()
    control_df = df[df["변수"].isin(meta.control_vars)].copy()

    write("=" * 72)
    write(f"[{meta.name}] 종속변수: {meta.dv}")
    write("=" * 72)

    if meta.r2 is not None:
        line = f"- 설명력: R² = {meta.r2:.3f}"
        if meta.adj_r2 is not None:
            line += f", Adj.R² = {meta.adj_r2:.3f}"
        write(line)
    if meta.f_p is not None:
        model_sig = "유의함" if meta.f_p < 0.05 else "유의하지 않음"
        write(f"- 모형 유의성: {p_to_text(meta.f_p)} ({model_sig})")

    if sig_df.empty:
        write("- 유의한 독립변수가 없습니다 (p < .05 기준).")
        write("")
        return report.getvalue()

    sig_sorted = sig_df.reindex(sig_df["β"].abs().sort_values(ascending=False).index)
    top_row = sig_sorted.iloc[0]

    if not focal_sig_df.empty:
        focal_sorted = focal_sig_df.reindex(
            focal_sig_df["β"].abs().sort_values(ascending=False).index
        )
        top_focal = focal_sorted.iloc[0]
        write(
            f"- 핵심 독립변수 중 영향력이 가장 큰 변수: "
            f"{top_focal['변수']} (β = {top_focal['β']:.3f})"
        )
    else:
        top_focal = None
        write("- 핵심 독립변수(전문가지향성, 학습민첩성)는 유의하지 않았습니다.")

    write(f"- 전체 변수 기준 최대 영향 변수: {top_row['변수']} (β = {top_row['β']:.3f})")
    write("")
    write("[핵심 독립변수 해석]")
    if focal_df.empty:
        write("  - 핵심 독립변수 정보가 설정되지 않았습니다.")
    else:
        for _, row in focal_df.iterrows():
            direction = "정(+)적" if row["B"] > 0 else "부(-)적"
            sig_text = "유의함" if row["p"] < 0.05 else "유의하지 않음"
            impact = beta_strength(abs(row["β"])) if pd.notna(row["β"]) else "판단 불가"
            write(
                f"  - {row['변수']}: {direction} 영향 "
                f"(B={row['B']:.3f}, β={row['β']:.3f}, {p_to_text(row['p'])}, {sig_text}, {impact})"
            )

    write("")
    write("[통제변수 결과]")
    if control_df.empty:
        write("  - 통제변수 정보가 설정되지 않았습니다.")
    else:
        for _, row in control_df.iterrows():
            direction = "정(+)적" if row["B"] > 0 else "부(-)적"
            sig_text = "유의함" if row["p"] < 0.05 else "유의하지 않음"
            write(
                f"  - {row['변수']}: {direction} 영향 "
                f"(B={row['B']:.3f}, β={row['β']:.3f}, {p_to_text(row['p'])}, {sig_text})"
            )

    write("")
    write("[전체 유의변수 참고]")
    for _, row in sig_sorted.iterrows():
        direction = "정(+)적" if row["B"] > 0 else "부(-)적"
        impact = beta_strength(abs(row["β"]))
        write(
            f"  - {row['변수']}: {direction} 영향 "
            f"(B={row['B']:.3f}, β={row['β']:.3f}, {p_to_text(row['p'])}, {impact})"
        )

    if not nonsig_df.empty:
        names = ", ".join(nonsig_df["변수"].astype(str).tolist())
        write("")
        write(f"[비유의 변수] {names} (p >= .05)")

    write("")
    write("[한 줄 결론]")
    if top_focal is not None:
        write(
            f"  {meta.dv}에 대해 핵심 독립변수 중에서는 "
            f"{top_focal['변수']}의 영향이 가장 크게 나타났습니다."
        )
    else:
        write(
            f"  {meta.dv}에 대해 핵심 독립변수의 유의한 효과는 확인되지 않았으며, "
            f"전체 변수 기준으로는 {top_row['변수']}의 영향이 가장 컸습니다."
        )
    write("")
    return report.getvalue()


def main() -> None:
    base = Path("processed")
    models = [
        (
            base / "regression_model1_innovation.csv",
            ModelMeta(
                name="모형 1",
                dv="혁신행동",
                focal_vars=("전문가지향성", "학습민첩성"),
                control_vars=("성별", "연령", "고용형태"),
                r2=0.358,
                adj_r2=0.356,
                f_p=0.0,
            ),
        ),
        (
            base / "regression_model2_turnover.csv",
            ModelMeta(
                name="모형 2",
                dv="이직의도",
                focal_vars=("전문가지향성", "학습민첩성"),
                control_vars=("성별", "연령", "고용형태"),
                r2=0.084,
                adj_r2=0.082,
                f_p=0.0,
            ),
        ),
    ]

    print("다중회귀 결과 자동 해석")
    print("")
    report_chunks = ["다중회귀분석 결과 해석 보고서", ""]
    for file_path, meta in models:
        report_chunks.append(explain_model(file_path, meta))

    report_text = "\n".join(report_chunks).rstrip() + "\n"
    report_path = base / "regression_interpretation_report.txt"
    report_path.write_text(report_text, encoding="utf-8-sig")
    print(f"보고서 저장 완료: {report_path}")


if __name__ == "__main__":
    main()
