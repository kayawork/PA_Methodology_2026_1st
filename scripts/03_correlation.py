# ============================================================
# 03_correlation.py
# 상관분석 (Pearson r + 유의확률)
# 실행: python scripts/03_correlation.py
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

# ── 전처리 데이터 로드 ───────────────────────────────────────
data = pd.read_csv('processed/kird_preprocessed.csv', encoding='utf-8-sig')
print(f"데이터 로드 완료: {len(data)}명\n")

corr_cols = ['전문가지향성', '학습민첩성', '혁신행동', '이직의도']

# ── 상관계수 행렬 ────────────────────────────────────────────
print("=" * 50)
print("상관분석 (Pearson r)")
print("=" * 50)

corr_matrix = data[corr_cols].corr().round(3)
print("\n[상관계수 행렬]")
print(corr_matrix.to_string())

# ── 유의확률 ─────────────────────────────────────────────────
print("\n[유의확률]")
print(f"  {'변수 쌍':<30} {'r':>7}  {'p':>8}  sig")
print(f"  {'-' * 55}")
for i, c1 in enumerate(corr_cols):
    for j, c2 in enumerate(corr_cols):
        if i < j:
            r, p = stats.pearsonr(data[c1], data[c2])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            pair = f"{c1} × {c2}"
            print(f"  {pair:<30} {r:>7.3f}  {p:>8.4f}  {sig}")

print("\n  * p < .05  ** p < .01  *** p < .001")

# ── 저장 ────────────────────────────────────────────────────
corr_matrix.to_csv('processed/correlation_matrix.csv', encoding='utf-8-sig')
print("\n저장 완료 → processed/correlation_matrix.csv")
