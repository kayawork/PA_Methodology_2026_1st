# ============================================================
# 02_reliability_and_descriptive.py
# 신뢰도 분석 (Cronbach's α) + 기술통계 분석
# 실행: python scripts/02_reliability_and_descriptive.py
# ============================================================

import pandas as pd
import numpy as np

# ── 전처리 데이터 로드 ───────────────────────────────────────
data = pd.read_csv('processed/kird_preprocessed.csv', encoding='utf-8-sig')
print(f"데이터 로드 완료: {len(data)}명\n")

# ── 신뢰도 분석 함수 (Cronbach's α) ─────────────────────────
def cronbach_alpha(df_items):
    df_items = df_items.dropna()
    n = df_items.shape[1]
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    return round((n / (n - 1)) * (1 - item_vars.sum() / total_var), 3)

# 원본 데이터에서 문항별 컬럼 필요 → 재로드
df_raw = pd.read_csv('KIRD_employee_data.csv', header=1, encoding='cp949')
expert_cols   = [c for c in df_raw.columns if '전문가 지향성' in c]
agility_cols  = [c for c in df_raw.columns if '학습민첩성' in c]
innov_cols    = [c for c in df_raw.columns if '혁신 행동' in c]
turnover_cols = [c for c in df_raw.columns if '이직의도' in c]

for cols in [expert_cols, agility_cols, innov_cols, turnover_cols]:
    for c in cols:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

# ── 신뢰도 출력 ──────────────────────────────────────────────
print("=" * 50)
print("신뢰도 분석 (Cronbach's α)")
print("=" * 50)
variables = {
    '전문가지향성': expert_cols,
    '학습민첩성':   agility_cols,
    '혁신행동':     innov_cols,
    '이직의도':     turnover_cols,
}
for name, cols in variables.items():
    alpha = cronbach_alpha(df_raw[cols])
    status = "우수" if alpha >= 0.8 else "양호" if alpha >= 0.7 else "주의"
    print(f"  {name:<12} ({len(cols)}문항)  α = {alpha}  [{status}]")

print("\n  기준: α ≥ .70 (Nunnally, 1978)")

# ── 기술통계 분석 ────────────────────────────────────────────
print("\n" + "=" * 50)
print("기술통계 분석")
print("=" * 50)

desc_cols = ['전문가지향성', '학습민첩성', '혁신행동', '이직의도']
desc = data[desc_cols].agg(['count', 'mean', 'std', 'min', 'max'])
desc.loc['skew'] = data[desc_cols].skew()
desc.loc['kurt'] = data[desc_cols].kurt()
desc.index = ['N', '평균', '표준편차', '최솟값', '최댓값', '왜도', '첨도']

print(desc.round(3).to_string())
print("\n  정규성 기준: |왜도| < 2, |첨도| < 7 (West et al., 1995)")

# ── 저장 ────────────────────────────────────────────────────
desc.round(3).to_csv('processed/descriptive_stats.csv', encoding='utf-8-sig')
print("\n저장 완료 → processed/descriptive_stats.csv")
