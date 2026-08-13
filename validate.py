"""
진단기 검증 — 이미 알고 있는 사실을 잡아내는지 확인한다.

기준값은 5주차 수업 중 BigQuery로 직접 확인한 값이다.
실행: python validate.py [데이터폴더]
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from profiler import (  # noqa: E402
    diagnose_all, predict_fanout, load_table, intersect_coverage,
)

RAW = Path(sys.argv[1] if len(sys.argv) > 1
           else r"c:\Users\kym\Desktop\모두의 연구소_데이터\my-wiki-02\raw")

# 수업 중 확인한 기준값 (행 수는 raw CSV 기준)
EXPECTED = {
    "data_customers":            {"rows": 500,  "type": "마스터"},
    "data_usage_history":        {"rows": 6000, "type": "패널/스냅샷"},
    "data_customer_acquisition": {"rows": 500,  "type": "마스터"},
    "data_consultations":        {"rows": 1320, "type": "이벤트 로그"},
    "data_voc":                  {"rows": 1307, "type": "이벤트 로그"},
    "data_product_events":       {"rows": 4312, "type": "이벤트 로그"},
    "data_subscription_events":  {"rows": 690,  "type": "이벤트 로그"},
    "data_marketing_spend":      {"rows": 396,  "type": "집계 테이블"},
    "data_monthly_subscription_status": {"rows": 19691, "type": "패널/스냅샷"},
}

print("=" * 78)
print("1. 테이블별 구조 진단")
print("=" * 78)

tables, metas = {}, {}
for f in sorted(RAW.glob("*.csv")):
    try:
        df, meta = load_table(f, f.stem)
        tables[f.stem], metas[f.stem] = df, meta
    except Exception as e:
        print(f"\n[{f.name}] 로딩 실패: {e}")

diags, cands = diagnose_all(tables)

for name in sorted(tables):
    d, meta = diags[name], metas[name]
    f = type("F", (), {"stem": name, "name": name + ".csv"})
    g = d["grain"]
    grain_txt = " + ".join(g.columns) if g.columns else "(찾지 못함)"
    if g.surrogate_keys and g.natural:
        grain_txt += f"   [대리키 제외: {', '.join(g.surrogate_keys)}]"

    exp = EXPECTED.get(f.stem)
    mark = ""
    if exp:
        ok_rows = d["rows"] == exp["rows"]
        ok_type = d["type"] == exp["type"]
        mark = "  [OK]" if (ok_rows and ok_type) else f"  [불일치 기대={exp}]"

    print(f"\n▸ {f.stem}  ({d['rows']:,}행 × {d['cols']}열, {meta['encoding']}){mark}")
    print(f"    유형   : {d['type']}  — {d['type_reason']}")
    print(f"    그레인 : {grain_txt}   (조합 {g.checked_combos}회 검사)")
    if d["date_cols"]:
        dc = ", ".join(f"{c}({i['granularity']})" for c, i in d["date_cols"].items())
        print(f"    날짜   : {dc}")
    flagged = d["profile"][d["profile"]["플래그"] != ""]
    for _, r in flagged.iterrows():
        print(f"    플래그 : {r['컬럼']} → {r['플래그']}")
    for w in meta["warnings"]:
        print(f"    경고   : {w}")

print()
print("=" * 78)
print("2. 팬아웃 예측 — BigQuery 실측값과 대조")
print("=" * 78)

# raw CSV 기준 기대값 (usage_history는 고객당 12행이므로 12 × 1,320 = 15,840)
CHECKS = [
    ("data_usage_history", "customer_id", "data_consultations", "customer_id", 15840),
    ("data_customers",     "customer_id", "data_usage_history", "customer_id", 6000),
    ("data_customers",     "customer_id", "data_consultations", "customer_id", 1320),
    ("data_consultations", "consult_id",  "data_satisfaction",  "consult_id",  1320),
    ("data_consultations", "customer_id", "data_voc",           "customer_id", None),
]

for lt, lc, rt, rc, expected in CHECKS:
    if lt not in tables or rt not in tables:
        continue
    r = predict_fanout(tables[lt], lc, tables[rt], rc)
    mark = ""
    if expected is not None:
        mark = "  [OK]" if r.joined_rows == expected else f"  [불일치 기대={expected:,}]"
    print(f"\n▸ {lt} ⋈ {rt}  on {lc}{mark}")
    print(f"    {r.left_rows:,}행 ⋈ {r.right_rows:,}행 → {r.joined_rows:,}행 "
          f"({r.fanout_factor:.2f}배, {r.relation})")
    print(f"    [{r.risk}] {r.note}")
    if r.left_dropped:
        print(f"    왼쪽에서 사라지는 행: {r.left_dropped}")

print()
print("=" * 78)
print("3. 조인키 후보 자동 탐색 (컬럼명을 보지 않고 값 겹침으로)")
print("=" * 78)

print(f"\n총 {len(cands)}건 발견. 상위 12건:\n")
print(f"{'왼쪽':<38} {'오른쪽':<38} {'겹침':>6} {'관계':>6}")
print("-" * 92)
for c in cands[:12]:
    rel = f"{'1' if c.left_unique else 'N'}:{'1' if c.right_unique else 'N'}"
    print(f"{c.left_table + '.' + c.left_col:<38} "
          f"{c.right_table + '.' + c.right_col:<38} "
          f"{c.overlap:>5.0%} {rel:>6}")

print()
print("=" * 78)
print("4. 유효구간과 교집합")
print("=" * 78)

all_cov = []
for name, d in diags.items():
    for cov in d["coverage"]:
        all_cov.append(cov)
        gap_txt = ""
        if cov.gaps:
            gap_txt = f"  빈 구간 {len(cov.gaps)}개 (예: {', '.join(cov.gaps[:3])})"
        print(f"\n▸ {cov.table}.{cov.column} ({cov.granularity})")
        print(f"    {cov.start} ~ {cov.end}   관측 {cov.periods_present}/{cov.periods_expected}{gap_txt}")
        if cov.note:
            print(f"    주의: {cov.note}")

inter = intersect_coverage(all_cov)
if inter:
    print(f"\n▸ 전체 교집합: {inter['start']} ~ {inter['end']}  "
          f"({'사용 가능' if inter['valid'] else '겹치는 구간 없음'})")
