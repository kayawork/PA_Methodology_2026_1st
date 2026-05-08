# ============================================================
# 04_multiple_regression.py
# 다중회귀분석
#   모형 1: 통제변수 + 전문가지향성 + 학습민첩성 → 혁신행동
#   모형 2: 통제변수 + 전문가지향성 + 학습민첩성 → 이직의도
# 실행: python scripts/04_multiple_regression.py
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

# ── 전처리 데이터 로드 ───────────────────────────────────────
data = pd.read_csv('processed/kird_preprocessed.csv', encoding='utf-8-sig')
print(f"데이터 로드 완료: {len(data)}명\n")

# ── 다중회귀분석 함수 ────────────────────────────────────────
def multiple_regression(data, X_cols, y_col):
    X = data[X_cols]
    y = data[y_col]
    X_np = np.column_stack([np.ones(len(X))] + [X[c].values for c in X_cols])
    y_np = y.values
    n, k = X_np.shape

    # OLS 계수 추정
    XtX_inv = np.linalg.inv(X_np.T @ X_np)
    betas   = XtX_inv @ X_np.T @ y_np
    y_hat   = X_np @ betas
    resid   = y_np - y_hat

    # R², Adj.R², F통계량
    ss_res  = np.sum(resid ** 2)
    ss_tot  = np.sum((y_np - y_np.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot
    r2_adj  = 1 - (1 - r2) * (n - 1) / (n - k)
    ss_reg  = ss_tot - ss_res
    f_stat  = (ss_reg / (k - 1)) / (ss_res / (n - k))
    f_p     = 1 - stats.f.cdf(f_stat, k - 1, n - k)

    # 표준오차, t값, p값
    mse    = ss_res / (n - k)
    se     = np.sqrt(np.diag(XtX_inv) * mse)
    t_vals = betas / se
    p_vals = [2 * (1 - stats.t.cdf(abs(t), df=n - k)) for t in t_vals]

    # 표준화 계수 (β)
    x_std     = np.std(X_np[:, 1:], axis=0, ddof=1)
    y_std     = np.std(y_np, ddof=1)
    std_betas = [None] + list(betas[1:] * x_std / y_std)

    # 출력
    print(f"\n  종속변수: {y_col}")
    print(f"  R² = {r2:.3f}   Adj.R² = {r2_adj:.3f}")
    print(f"  F({k-1}, {n-k}) = {f_stat:.3f}   p = {f_p:.4f}")
    print(f"\n  {'변수':<14} {'B':>7} {'SE':>7} {'β':>7} {'t':>8} {'p':>8}  sig")
    print(f"  {'-' * 62}")

    labels  = ['(상수)'] + X_cols
    results = []
    for label, b, s, t, p, sb in zip(labels, betas, se, t_vals, p_vals, std_betas):
        sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        sb_str = f"{sb:.3f}" if sb is not None else "  -  "
        print(f"  {label:<14} {b:>7.3f} {s:>7.3f} {sb_str:>7} {t:>8.3f} {p:>8.4f}  {sig}")
        results.append({
            '변수': label, 'B': round(b, 3), 'SE': round(s, 3),
            'β': round(sb, 3) if sb else None,
            't': round(t, 3), 'p': round(p, 4), 'sig': sig
        })

    return pd.DataFrame(results), {'R2': round(r2, 3), 'Adj_R2': round(r2_adj, 3),
                                    'F': round(f_stat, 3), 'F_p': round(f_p, 4)}


# ── 변수 설정 ────────────────────────────────────────────────
ctrl_vars = ['성별', '연령', '고용형태']
main_vars = ['전문가지향성', '학습민첩성']
all_vars  = ctrl_vars + main_vars

# ── 분석 실행 ────────────────────────────────────────────────
print("=" * 60)
print("다중회귀분석")
print("=" * 60)

print("\n[모형 1] 혁신행동")
coef1, fit1 = multiple_regression(data, all_vars, '혁신행동')

print("\n[모형 2] 이직의도")
coef2, fit2 = multiple_regression(data, all_vars, '이직의도')

print("\n  * p < .05  ** p < .01  *** p < .001")

# ── 모형 비교 요약 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("모형 비교 요약")
print("=" * 60)
print(f"  {'':15} {'모형1(혁신행동)':>16} {'모형2(이직의도)':>16}")
print(f"  {'-' * 50}")
print(f"  {'R²':15} {fit1['R2']:>16.3f} {fit2['R2']:>16.3f}")
print(f"  {'Adj. R²':15} {fit1['Adj_R2']:>16.3f} {fit2['Adj_R2']:>16.3f}")
print(f"  {'F':15} {fit1['F']:>16.3f} {fit2['F']:>16.3f}")
print(f"  {'p':15} {fit1['F_p']:>16.4f} {fit2['F_p']:>16.4f}")

# ── 저장 ────────────────────────────────────────────────────
coef1.to_csv('processed/regression_model1_innovation.csv', index=False, encoding='utf-8-sig')
coef2.to_csv('processed/regression_model2_turnover.csv',   index=False, encoding='utf-8-sig')
print("\n저장 완료")
print("  → processed/regression_model1_innovation.csv")
print("  → processed/regression_model2_turnover.csv")
