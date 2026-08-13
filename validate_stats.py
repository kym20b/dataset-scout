"""
검정 추천·실행 로직 검증 — 수업에서 확인한 결과를 재현하는지 본다.

기대값
  자동이체 × 이탈   : chi2=14.880, p=0.000115, 11.9%p, 2.5배
  채널로 층화        : 6개 채널 전부 같은 방향 (역전 없음)
  챗봇 데이터        : 난이도로 층화하면 두 층 모두 역전 (심슨의 역설)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

from stats_advisor import (  # noqa: E402
    recommend_test, run_test, stratified_compare, classify_variable,
    bonferroni, family_error_rate,
)

RAW = Path(r"c:\Users\kym\Desktop\모두의 연구소_데이터\my-wiki-02\raw")
HERE = Path(__file__).parent


def line(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ------------------------------------------------------------------
line("1. 자동이체 × 이탈 — 카이제곱")

cust = pd.read_csv(RAW / "data_customers.csv")
ev = pd.read_csv(RAW / "data_product_events.csv")
acq = pd.read_csv(RAW / "data_customer_acquisition.csv")

autopay = set(ev.loc[ev["feature"] == "자동이체등록", "customer_id"])
cust["autopay"] = cust["customer_id"].isin(autopay).map({True: "등록", False: "미등록"})
cust = cust.merge(acq[["customer_id", "acquisition_channel"]], on="customer_id")

print("변수 판정:")
for c in ["churn_yn", "autopay", "acquisition_channel", "age", "customer_id"]:
    v = classify_variable(cust, c)
    print(f"  {c:<22} {v['kind']:<14} 고유값 {v['nunique']:<5} "
          f"{'사용가능' if v['usable'] else '사용불가 — ' + v['note']}")

rec, o, e = recommend_test(cust, "churn_yn", "autopay")
print(f"\n추천: {rec.test}")
print(f"근거: {rec.reason}")
for w in rec.warnings:
    print(f"  경고: {w}")

res = run_test(cust, "churn_yn", "autopay", rec)
print(f"\n통계량 = {res.statistic:.3f}")
print(f"p값     = {res.p_value:.6f}")
print("효과 크기:")
for k, v in res.effect.items():
    print(f"  {k}: {v}")
print("\n교차표:")
print(res.table.to_string())

ok = abs(res.statistic - 14.880) < 0.01 and abs(res.p_value - 0.000115) < 1e-5
print(f"\n→ 기대값(14.880 / 0.000115) 일치: {'OK' if ok else '불일치'}")

# ------------------------------------------------------------------
line("2. 채널로 층화 — 역전이 없어야 한다")

st = stratified_compare(cust, "churn_yn", "autopay", "acquisition_channel")
print(st.strata.to_string(index=False))
print(f"\n방향 일치: {st.consistent}")
print(f"판정: {st.note}")

# ------------------------------------------------------------------
line("3. 챗봇 데이터 — 심슨의 역설을 잡아내는가")

bot = pd.read_csv(HERE / ".." / "Day3_실습데이터" / "chatbot_resolution.csv")
overall = bot.groupby("channel")["resolved"].mean() * 100
print("전체 해결률:")
for k, v in overall.items():
    print(f"  {k}: {v:.1f}%")

st2 = stratified_compare(bot, "resolved", "channel", "difficulty")
print("\n난이도별:")
print(st2.strata.to_string(index=False))
print(f"\n방향 일치: {st2.consistent}   역전된 층: {st2.reversed_strata}")
print(f"판정: {st2.note}")
print(f"\n→ 심슨의 역설 탐지: {'OK' if not st2.consistent else '실패 — 잡아내지 못함'}")

# ------------------------------------------------------------------
line("4. 다른 타입 조합 — 추천이 바뀌는가")

for outcome, explain in [
    ("age", "autopay"),                    # 연속형 × 이진 → t검정
    ("age", "acquisition_channel"),        # 연속형 × 범주형(6) → ANOVA
    ("churn_yn", "age"),                   # 이진 × 연속형 → t검정(방향 전환)
    ("churn_yn", "customer_id"),           # 이진 × 식별자 → 차단
]:
    r, _, _ = recommend_test(cust, outcome, explain)
    if r.blocked:
        print(f"  {outcome:<10} × {explain:<20} → 차단: {r.blocked}")
    else:
        sw = f"  (원래 {r.first_choice} → {r.switched_why})" if r.first_choice else ""
        print(f"  {outcome:<10} × {explain:<20} → {r.test}{sw}")

usage = pd.read_csv(RAW / "data_usage_history.csv")
r, _, _ = recommend_test(usage, "billing_amount", "data_gb")
print(f"  {'billing_amount':<10} × {'data_gb':<20} → {r.test}")
res = run_test(usage, "billing_amount", "data_gb", r)
print(f"     피어슨 r = {res.effect['피어슨 r']}, R² = {res.effect['결정계수 R²']}")

# ------------------------------------------------------------------
line("5. 기대빈도 부족 시 Fisher 자동 전환")

small = cust.sample(45, random_state=1)
r, _, _ = recommend_test(small, "churn_yn", "autopay")
print(f"45행 표본 → {r.test}")
if r.first_choice:
    print(f"  원래 후보: {r.first_choice}")
    print(f"  전환 사유: {r.switched_why}")
for w in r.warnings:
    print(f"  경고: {w}")

# ------------------------------------------------------------------
line("6. 다중비교 기준")

for k in (1, 6, 7, 20):
    print(f"  검정 {k:>2}회 → 보정 α = {bonferroni(0.05, k):.5f}, "
          f"최소 1회 오류 확률 = {family_error_rate(0.05, k) * 100:.1f}%")
