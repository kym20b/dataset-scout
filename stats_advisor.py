"""
검정 방법 추천 — 규칙 기반. LLM을 쓰지 않는다.

"어떤 검정을 써야 하는가"는 변수 타입의 조합으로 결정된다.
사람이 결과 변수와 설명 변수를 고르면, 나머지는 계산이다.

  결과 타입 × 설명 타입 → 검정 방법
  + 전제 점검(기대빈도·표본 수) → 대안 검정으로 자동 전환
  + 효과 크기 계산 (p값만으로는 크기를 알 수 없다)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

MIN_SAMPLE = 30          # 위키 공통 규칙: 표본 30 미만은 참고용
MIN_EXPECTED = 5         # 카이제곱 전제: 모든 칸의 기대빈도 5 이상
MAX_CATEGORIES = 20      # 이보다 많으면 범주형으로 보지 않는다


# ==================================================================
# 1. 변수 타입 판정
# ==================================================================
def classify_variable(df: pd.DataFrame, col: str, date_cols: dict | None = None) -> dict:
    """검정에 쓸 수 있는 형태인지 판정한다."""
    date_cols = date_cols or {}
    s = df[col]
    nunique = int(s.nunique(dropna=True))
    n_valid = int(s.notna().sum())

    if col in date_cols:
        kind, usable, note = "날짜", False, "날짜는 직접 검정하지 않습니다. 기간 구분으로 바꿔 쓰세요."
    elif nunique <= 1:
        kind, usable, note = "상수", False, "모든 행이 같은 값이라 아무것도 구분하지 못합니다."
    elif nunique == 2:
        kind, usable, note = "이진", True, ""
    elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if nunique <= 10:
            kind, usable, note = "범주형(숫자)", True, "숫자지만 값이 적어 범주로 다룹니다."
        else:
            kind, usable, note = "연속형", True, ""
    elif nunique <= MAX_CATEGORIES:
        kind, usable, note = "범주형", True, ""
    else:
        kind, usable, note = "식별자/텍스트", False, (
            f"고유값이 {nunique:,}개입니다. 집단으로 묶을 수 없어 검정에 쓸 수 없습니다."
        )

    return {
        "column": col, "kind": kind, "usable": usable,
        "nunique": nunique, "n_valid": n_valid, "note": note,
        "is_categorical": kind in ("이진", "범주형", "범주형(숫자)"),
        "is_continuous": kind == "연속형",
    }


# ==================================================================
# 2. 검정 방법 추천
# ==================================================================
@dataclass
class Recommendation:
    test: str                      # 실제로 쓸 검정
    reason: str                    # 왜 이 검정인가
    first_choice: str = ""         # 원래 후보 (전제 위반으로 바뀐 경우)
    switched_why: str = ""         # 바뀐 이유
    warnings: list[str] = field(default_factory=list)
    blocked: str = ""              # 검정 자체가 불가능한 경우 사유


def recommend_test(df: pd.DataFrame, outcome: str, explain: str,
                   date_cols: dict | None = None) -> tuple[Recommendation, dict, dict]:
    """결과 변수와 설명 변수의 타입 조합으로 검정 방법을 정한다."""
    o = classify_variable(df, outcome, date_cols)
    e = classify_variable(df, explain, date_cols)

    if not o["usable"]:
        return Recommendation("", "", blocked=f"결과 변수 `{outcome}`: {o['note']}"), o, e
    if not e["usable"]:
        return Recommendation("", "", blocked=f"설명 변수 `{explain}`: {e['note']}"), o, e
    if outcome == explain:
        return Recommendation("", "", blocked="같은 컬럼을 두 번 선택했습니다."), o, e

    sub = df[[outcome, explain]].dropna()
    if len(sub) < 10:
        return Recommendation("", "", blocked=f"결측을 빼면 {len(sub)}행뿐입니다."), o, e

    warns = []
    if len(sub) < MIN_SAMPLE:
        warns.append(
            f"전체 표본이 {len(sub)}행으로 최소표본 {MIN_SAMPLE}에 미달합니다. "
            "결론의 근거로 쓰지 마세요."
        )

    # ---- 범주형 × 범주형 → 카이제곱 (전제 위반 시 Fisher)
    if o["is_categorical"] and e["is_categorical"]:
        table = pd.crosstab(sub[explain], sub[outcome])
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        n_small = int((expected < MIN_EXPECTED).sum())

        small_groups = table.sum(axis=1)
        thin = small_groups[small_groups < MIN_SAMPLE]
        if len(thin):
            warns.append(
                f"표본 {MIN_SAMPLE} 미만인 집단: "
                + ", ".join(f"{i}({v}건)" for i, v in thin.items())
                + " — 참고용으로만 쓰세요."
            )

        if n_small > 0:
            if table.shape == (2, 2):
                return Recommendation(
                    test="Fisher 정확검정",
                    reason="두 범주형 변수의 관계를 봅니다. 2×2 표입니다.",
                    first_choice="카이제곱 검정",
                    switched_why=(
                        f"기대빈도가 {MIN_EXPECTED} 미만인 칸이 {n_small}개 있어 "
                        "카이제곱의 전제를 충족하지 못합니다."
                    ),
                    warnings=warns,
                ), o, e
            warns.append(
                f"기대빈도 {MIN_EXPECTED} 미만인 칸이 {n_small}개입니다. "
                "2×2가 아니라 Fisher로 자동 전환할 수 없으니, 범주를 합치는 것을 검토하세요."
            )

        return Recommendation(
            test="카이제곱 검정",
            reason=f"두 범주형 변수의 관계를 봅니다. 교차표 {table.shape[0]}×{table.shape[1]}.",
            warnings=warns,
        ), o, e

    # ---- 연속형 × 범주형 → 집단 수에 따라 t검정 / ANOVA
    if o["is_continuous"] and e["is_categorical"]:
        groups = sub.groupby(explain)[outcome]
        k = groups.ngroups
        sizes = groups.size()
        thin = sizes[sizes < MIN_SAMPLE]
        if len(thin):
            warns.append(
                f"표본 {MIN_SAMPLE} 미만인 집단: "
                + ", ".join(f"{i}({v}건)" for i, v in thin.items())
            )

        if k == 2:
            return Recommendation(
                test="t검정 (두 집단 평균 비교)",
                reason=f"연속형 결과를 두 집단으로 나눠 비교합니다.",
                warnings=warns + [
                    "표본이 작거나 분포가 심하게 치우쳐 있으면 Mann-Whitney U 검정을 함께 보세요."
                ],
            ), o, e

        return Recommendation(
            test="분산분석 (ANOVA)",
            reason=f"연속형 결과를 {k}개 집단으로 나눠 비교합니다.",
            warnings=warns + [
                "ANOVA는 '어딘가 차이가 있다'까지만 말합니다. "
                "어느 집단끼리 다른지는 사후검정이 필요합니다."
            ],
        ), o, e

    # ---- 범주형 결과 × 연속형 설명 → 방향을 뒤집어 처리
    if o["is_categorical"] and e["is_continuous"]:
        k = sub[outcome].nunique()
        test = "t검정 (두 집단 평균 비교)" if k == 2 else "분산분석 (ANOVA)"
        return Recommendation(
            test=test,
            reason=(
                f"결과가 범주형이고 설명이 연속형입니다. "
                f"`{outcome}` 집단별로 `{explain}`의 평균을 비교하는 형태로 바꿔서 봅니다."
            ),
            warnings=warns + [
                "이 방향은 '집단에 따라 값이 다른가'를 봅니다. "
                "'값에 따라 집단이 갈리는가'를 보려면 구간으로 나눠 범주형으로 만드세요."
            ],
        ), o, e

    # ---- 연속형 × 연속형 → 상관분석
    return Recommendation(
        test="상관분석 (피어슨 · 스피어만)",
        reason="두 연속형 변수가 함께 움직이는지 봅니다.",
        warnings=warns + [
            "상관은 인과가 아닙니다. 직선 관계만 잡아내므로 산점도를 함께 보세요."
        ],
    ), o, e


# ==================================================================
# 3. 검정 실행
# ==================================================================
@dataclass
class TestResult:
    test: str
    statistic: float | None
    p_value: float | None
    effect: dict                       # 효과 크기 (이름 → 값)
    table: pd.DataFrame | None = None  # 교차표·집단별 요약
    note: str = ""


def _iga(word) -> str:
    """한글 조사 이/가를 받침에 맞춰 고른다."""
    s = str(word)
    if not s:
        return "가"
    ch = s[-1]
    if "가" <= ch <= "힣":
        return "이" if (ord(ch) - 0xAC00) % 28 else "가"
    return "가"


def _cramers_v(chi2: float, n: int, table_shape) -> float:
    k = min(table_shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def run_test(df: pd.DataFrame, outcome: str, explain: str, rec: Recommendation) -> TestResult:
    sub = df[[outcome, explain]].dropna()

    # ---- 카이제곱 / Fisher
    if rec.test.startswith("카이제곱") or rec.test.startswith("Fisher"):
        table = pd.crosstab(sub[explain], sub[outcome])

        if rec.test.startswith("Fisher"):
            odds, p = stats.fisher_exact(table.values)
            stat, effect = odds, {"오즈비": odds}
        else:
            stat, p, dof, expected = stats.chi2_contingency(table.values, correction=False)
            effect = {
                "Cramér's V": _cramers_v(stat, len(sub), table.shape),
                "자유도": dof,
            }

        # 2×2면 실무에서 바로 쓰는 지표를 추가로 계산.
        # 어느 쪽을 '발생'으로 볼지 명시해야 한다. 마지막 열을 발생으로 본다
        # (churn_yn의 N/Y → Y, resolved의 False/True → True).
        if table.shape == (2, 2):
            event = table.columns[-1]
            rates = (table[event] / table.sum(axis=1) * 100)
            g0, g1 = table.index[0], table.index[1]
            r0, r1 = rates.iloc[0], rates.iloc[1]

            effect["발생 기준"] = f"{outcome} = {event}"
            effect[f"{g0} 발생률"] = f"{r0:.1f}%"
            effect[f"{g1} 발생률"] = f"{r1:.1f}%"
            effect["차이"] = f"{r0 - r1:+.1f}%p"
            if min(r0, r1) > 0:
                hi, lo = (g0, g1) if r0 >= r1 else (g1, g0)
                effect["상대 비율"] = (
                    f"{hi}{_iga(hi)} {lo}의 {max(r0, r1) / min(r0, r1):.2f}배"
                )

        pct = (table.div(table.sum(axis=1), axis=0) * 100).round(1)
        show = table.copy().astype(str)
        for col in table.columns:
            show[col] = table[col].astype(str) + " (" + pct[col].astype(str) + "%)"
        show["합계"] = table.sum(axis=1)

        return TestResult(rec.test, float(stat), float(p), effect, show)

    # ---- t검정
    if rec.test.startswith("t검정"):
        cat, num = (explain, outcome) if sub[outcome].nunique() > 2 else (outcome, explain)
        if sub[explain].nunique() == 2:
            cat, num = explain, outcome
        groups = [g[num].values for _, g in sub.groupby(cat)]
        labels = list(sub.groupby(cat).groups.keys())

        stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)  # Welch
        u_stat, u_p = stats.mannwhitneyu(groups[0], groups[1])

        m1, m2 = np.mean(groups[0]), np.mean(groups[1])
        pooled = np.sqrt((np.var(groups[0], ddof=1) + np.var(groups[1], ddof=1)) / 2)
        effect = {
            "평균 차이": f"{m1 - m2:+,.2f}",
            "Cohen's d": round(float((m1 - m2) / pooled), 3) if pooled else 0.0,
            "Mann-Whitney p": f"{u_p:.4g}",
        }
        summary = sub.groupby(cat)[num].agg(["count", "mean", "std", "median"]).round(2)
        return TestResult("t검정 (Welch)", float(stat), float(p), effect, summary,
                          note="분산이 다를 수 있으므로 Welch 방식을 씁니다.")

    # ---- ANOVA
    if rec.test.startswith("분산분석"):
        cat, num = (explain, outcome) if sub[outcome].nunique() > sub[explain].nunique() else (outcome, explain)
        if classify_variable(sub, explain)["is_categorical"]:
            cat, num = explain, outcome
        groups = [g[num].values for _, g in sub.groupby(cat)]
        stat, p = stats.f_oneway(*groups)
        k_stat, k_p = stats.kruskal(*groups)

        grand = sub[num].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        ss_total = ((sub[num] - grand) ** 2).sum()
        effect = {
            "eta²": round(float(ss_between / ss_total), 4) if ss_total else 0.0,
            "집단 수": len(groups),
            "Kruskal-Wallis p": f"{k_p:.4g}",
        }
        summary = sub.groupby(cat)[num].agg(["count", "mean", "std"]).round(2)
        return TestResult("분산분석 (ANOVA)", float(stat), float(p), effect, summary)

    # ---- 상관분석
    r, rp = stats.pearsonr(sub[outcome], sub[explain])
    rho, sp = stats.spearmanr(sub[outcome], sub[explain])
    effect = {
        "피어슨 r": round(float(r), 4),
        "스피어만 ρ": round(float(rho), 4),
        "결정계수 R²": round(float(r ** 2), 4),
        "스피어만 p": f"{sp:.4g}",
    }
    return TestResult("상관분석", float(r), float(rp), effect, None,
                      note="피어슨은 직선 관계, 스피어만은 순위 관계를 봅니다.")


# ==================================================================
# 4. 층화 비교 — 제3의 변수를 통제한다
# ==================================================================
@dataclass
class StratifiedResult:
    strata: pd.DataFrame          # 층별 결과
    overall_direction: str
    consistent: bool              # 모든 층에서 방향이 같은가
    reversed_strata: list         # 방향이 뒤집힌 층
    note: str


def stratified_compare(df: pd.DataFrame, outcome: str, explain: str,
                       stratifier: str) -> StratifiedResult | None:
    """
    층으로 나눠도 관계가 유지되는지 본다.
    모든 층에서 방향이 뒤집히면 심슨의 역설이다.

    현재는 결과가 이진, 설명이 범주형(2종)인 경우만 지원한다.
    """
    sub = df[[outcome, explain, stratifier]].dropna()
    if sub[outcome].nunique() != 2 or sub[explain].nunique() != 2:
        return None

    o_vals = sorted(sub[outcome].unique(), key=str)
    e_vals = sorted(sub[explain].unique(), key=str)
    positive = o_vals[-1]      # 뒤쪽 값을 '발생'으로 본다

    def rate(frame, ev):
        part = frame[frame[explain] == ev]
        return (part[outcome] == positive).mean() * 100 if len(part) else np.nan, len(part)

    # 전체
    r0_all, n0_all = rate(sub, e_vals[0])
    r1_all, n1_all = rate(sub, e_vals[1])
    overall_diff = r0_all - r1_all

    rows = []
    reversed_ = []
    for key, g in sub.groupby(stratifier):
        r0, n0 = rate(g, e_vals[0])
        r1, n1 = rate(g, e_vals[1])
        diff = r0 - r1
        if not np.isnan(diff) and np.sign(diff) != np.sign(overall_diff) and diff != 0:
            reversed_.append(str(key))
        rows.append({
            stratifier: key,
            f"{e_vals[0]} n": n0, f"{e_vals[0]} 비율": round(r0, 1) if not np.isnan(r0) else None,
            f"{e_vals[1]} n": n1, f"{e_vals[1]} 비율": round(r1, 1) if not np.isnan(r1) else None,
            "차이(%p)": round(diff, 1) if not np.isnan(diff) else None,
            "표본충족": "○" if min(n0, n1) >= MIN_SAMPLE else "×",
        })

    strata = pd.DataFrame(rows)
    direction = f"{e_vals[0]}가 {e_vals[1]}보다 {'높음' if overall_diff > 0 else '낮음'}"

    if reversed_:
        note = (
            f"**{len(reversed_)}개 층에서 방향이 뒤집힙니다** ({', '.join(reversed_)}). "
            f"전체 평균({overall_diff:+.1f}%p)은 집단 구성 차이 때문에 생긴 것일 수 있습니다. "
            "심슨의 역설을 의심하세요."
        )
    else:
        note = (
            f"모든 층에서 방향이 같습니다. `{stratifier}`로는 설명되지 않는 관계입니다."
        )

    return StratifiedResult(strata, direction, not reversed_, reversed_, note)


# ==================================================================
# 5. 다중비교 — 몇 번 검정했는지 세어둔다
# ==================================================================
def bonferroni(alpha: float, k: int) -> float:
    return alpha / k if k else alpha


def family_error_rate(alpha: float, k: int) -> float:
    """검정을 k번 했을 때 최소 한 번 잘못 기각할 확률."""
    return 1 - (1 - alpha) ** k
