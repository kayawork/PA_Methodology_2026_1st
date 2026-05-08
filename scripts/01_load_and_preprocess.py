# ============================================================
# 01_load_and_preprocess.py
# 데이터 로드 → 변수 합산평균 → 전처리 → 저장
# 실행: python scripts/01_load_and_preprocess.py
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ── 데이터 로드 ──────────────────────────────────────────────
df = pd.read_csv('KIRD_employee_data.csv', header=1, encoding='cp949')
print(f"원본 데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")

# ── 변수 컬럼 정의 ───────────────────────────────────────────
expert_cols   = [c for c in df.columns if '전문가 지향성' in c]   # Q6_1~4  (4문항)
agility_cols  = [c for c in df.columns if '학습민첩성' in c]      # Q10_1~5 (5문항)
innov_cols    = [c for c in df.columns if '혁신 행동' in c]       # Q28_4~7 (4문항)
turnover_cols = [c for c in df.columns if '이직의도' in c]        # Q30_4~6 (3문항)

print(f"\n문항 수 확인")
print(f"  전문가지향성 : {len(expert_cols)}문항")
print(f"  학습민첩성   : {len(agility_cols)}문항")
print(f"  혁신행동     : {len(innov_cols)}문항")
print(f"  이직의도     : {len(turnover_cols)}문항")

# ── 숫자 변환 ────────────────────────────────────────────────
for cols in [expert_cols, agility_cols, innov_cols, turnover_cols]:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df['성별']    = pd.to_numeric(df['성별'],    errors='coerce')
df['연령']    = pd.to_numeric(df['연령'],    errors='coerce')
df['고용형태'] = pd.to_numeric(df['고용형태'], errors='coerce')

# ── 합산평균 변수 생성 ───────────────────────────────────────
df['전문가지향성'] = df[expert_cols].mean(axis=1)
df['학습민첩성']   = df[agility_cols].mean(axis=1)
df['혁신행동']     = df[innov_cols].mean(axis=1)
df['이직의도']     = df[turnover_cols].mean(axis=1)

# ── 분석용 데이터셋 추출 및 결측 제거 ───────────────────────
ana_cols = ['성별', '연령', '고용형태',
            '전문가지향성', '학습민첩성',
            '혁신행동', '이직의도']
data = df[ana_cols].dropna()

print(f"\n결측 제거 후 유효 표본 수: {len(data)}명")

# ── 저장 ────────────────────────────────────────────────────
Path('processed').mkdir(parents=True, exist_ok=True)
data.to_csv('processed/kird_preprocessed.csv', index=False, encoding='utf-8-sig')
print("저장 완료 → processed/kird_preprocessed.csv")
