# ============================================================
# 06_visualization.py
# 분석 단계별 시각화
#   Fig 1. 기술통계 — 변수별 히스토그램
#   Fig 2. 기술통계 — 변수별 박스플롯
#   Fig 3. 상관분석 — 히트맵
#   Fig 4. 다중회귀 — 표준화 계수(β) 비교
#   Fig 5. 산점도   — 주요 변수 쌍 + 회귀선
# 실행: python scripts/05_visualization.py
# ============================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path

Path('processed/.mplconfig').mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(Path('processed/.mplconfig').resolve())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── 한글 폰트 설정 ───────────────────────────────────────────
font_candidates = [
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',            # macOS
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',    # macOS
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',# Linux
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',       # Linux
]

selected_font = None
for font_path in font_candidates:
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        selected_font = fm.FontProperties(fname=font_path).get_name()
        break

if selected_font:
    plt.rcParams['font.family'] = selected_font
else:
    # 시스템 한글 폰트를 못 찾는 경우 기본 sans-serif로 진행
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ── 출력 폴더 생성 ───────────────────────────────────────────
Path('processed/figures').mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ──────────────────────────────────────────────
data = pd.read_csv('processed/kird_preprocessed.csv', encoding='utf-8-sig')
print(f"데이터 로드 완료: {len(data)}명\n")

desc_cols  = ['전문가지향성', '학습민첩성', '혁신행동', '이직의도']
all_vars   = ['성별', '연령', '고용형태', '전문가지향성', '학습민첩성']
var_labels = ['성별', '연령', '고용형태', '전문가\n지향성', '학습\n민첩성']
colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']


# ── OLS 회귀 함수 (표준화 계수·p값 반환) ─────────────────────
def get_betas(data, X_cols, y_col):
    X_np = np.column_stack([np.ones(len(data))] + [data[c].values for c in X_cols])
    y_np = data[y_col].values
    n, k = X_np.shape
    XtX_inv = np.linalg.inv(X_np.T @ X_np)
    betas   = XtX_inv @ X_np.T @ y_np
    resid   = y_np - X_np @ betas
    mse     = np.sum(resid**2) / (n - k)
    se      = np.sqrt(np.diag(XtX_inv) * mse)
    t_vals  = betas / se
    p_vals  = [2 * (1 - stats.t.cdf(abs(t), df=n-k)) for t in t_vals]
    x_std   = np.std(X_np[:, 1:], axis=0, ddof=1)
    y_std   = np.std(y_np, ddof=1)
    std_betas = list(betas[1:] * x_std / y_std)
    return std_betas, p_vals[1:]


# ════════════════════════════════════════════════════════════
# Fig 1. 기술통계 — 히스토그램 (2×2)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('기술통계 분석 — 변수별 분포', fontsize=14, fontweight='bold', y=1.01)

for ax, col, color in zip(axes.flatten(), desc_cols, colors):
    ax.hist(data[col], bins=20, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(data[col].mean(), color='black', linestyle='--',
               linewidth=1.2, label=f'M = {data[col].mean():.2f}')
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('응답값 (1~5점)', fontsize=9)
    ax.set_ylabel('빈도', fontsize=9)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('processed/figures/fig1_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 저장 완료 → processed/figures/fig1_histogram.png")


# ════════════════════════════════════════════════════════════
# Fig 2. 기술통계 — 박스플롯
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(
    [data[c] for c in desc_cols],
    tick_labels=desc_cols,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker='o', markersize=3, alpha=0.4)
)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('기술통계 분석 — 변수별 박스플롯', fontsize=14, fontweight='bold')
ax.set_ylabel('응답값 (1~5점)', fontsize=11)
ax.set_ylim(0.5, 5.5)
ax.axhline(3, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='척도 중간값 (3점)')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('processed/figures/fig2_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 저장 완료 → processed/figures/fig2_boxplot.png")


# ════════════════════════════════════════════════════════════
# Fig 3. 상관분석 — 히트맵
# ════════════════════════════════════════════════════════════
corr = data[desc_cols].corr()

annot = pd.DataFrame(index=desc_cols, columns=desc_cols, dtype=str)
for i, c1 in enumerate(desc_cols):
    for j, c2 in enumerate(desc_cols):
        r = corr.loc[c1, c2]
        if i == j:
            annot.loc[c1, c2] = '1.000'
        else:
            _, p = stats.pearsonr(data[c1], data[c2])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            annot.loc[c1, c2] = f'{r:.3f}{sig}'

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=annot, fmt='', cmap='RdYlBu_r',
            vmin=-1, vmax=1, center=0,
            linewidths=0.5, linecolor='white',
            square=True, ax=ax,
            annot_kws={'size': 11},
            xticklabels=desc_cols,
            yticklabels=desc_cols)

ax.set_title('상관분석 — 상관계수 히트맵', fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(desc_cols, rotation=0, fontsize=10)
ax.set_yticklabels(desc_cols, rotation=0, fontsize=10)
fig.text(0.5, -0.04, '* p < .05   ** p < .01   *** p < .001',
         ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('processed/figures/fig3_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 저장 완료 → processed/figures/fig3_heatmap.png")


# ════════════════════════════════════════════════════════════
# Fig 4. 다중회귀 — 표준화 계수(β) 비교
# ════════════════════════════════════════════════════════════
betas1, pvals1 = get_betas(data, all_vars, '혁신행동')
betas2, pvals2 = get_betas(data, all_vars, '이직의도')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('다중회귀분석 — 표준화 계수(β) 비교', fontsize=14, fontweight='bold')

for ax, betas, pvals, title, color in zip(
    axes,
    [betas1, betas2],
    [pvals1, pvals2],
    ['모형 1: 혁신행동  (R²=.358)', '모형 2: 이직의도  (R²=.084)'],
    ['#4C72B0', '#C44E52']
):
    bar_colors = [color if p < 0.05 else '#CCCCCC' for p in pvals]
    bars = ax.bar(var_labels, betas, color=bar_colors, edgecolor='white', linewidth=0.8)

    for bar, beta, p in zip(bars, betas, pvals):
        sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ypos = beta + 0.01 if beta >= 0 else beta - 0.025
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{beta:.3f}{sig}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('표준화 계수 (β)', fontsize=10)
    ax.set_ylim(min(betas) - 0.08, max(betas) + 0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor=color,     label='유의 (p < .05)'),
        mpatches.Patch(facecolor='#CCCCCC', label='비유의'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('processed/figures/fig4_regression_beta.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 저장 완료 → processed/figures/fig4_regression_beta.png")


# ════════════════════════════════════════════════════════════
# Fig 5. 산점도 + 회귀선 (주요 변수 쌍)
# ════════════════════════════════════════════════════════════
pairs = [
    ('전문가지향성', '혁신행동',  '#4C72B0'),
    ('학습민첩성',   '혁신행동',  '#55A868'),
    ('전문가지향성', '이직의도',  '#DD8452'),
    ('학습민첩성',   '이직의도',  '#C44E52'),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle('주요 변수 간 산점도 및 회귀선', fontsize=14, fontweight='bold')

for ax, (xvar, yvar, color) in zip(axes.flatten(), pairs):
    ax.scatter(data[xvar], data[yvar], alpha=0.08, s=10, color=color)

    slope, intercept, r, p, _ = stats.linregress(data[xvar], data[yvar])
    x_line = np.linspace(data[xvar].min(), data[xvar].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax.set_title(f'{xvar} → {yvar}  (r = {r:.3f}{sig})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(xvar, fontsize=10)
    ax.set_ylabel(yvar, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('processed/figures/fig5_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 저장 완료 → processed/figures/fig5_scatter.png")

print("\n✅ 전체 시각화 완료 → processed/figures/")
