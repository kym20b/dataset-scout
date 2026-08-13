"""
데이터셋 구조 진단기 — Streamlit UI.

실행:  streamlit run app.py

CSV를 올리면 컬럼의 의미를 모르는 상태에서도
그레인·조인 위험·유효구간을 진단한다. 데이터는 밖으로 나가지 않는다.
"""

import pandas as pd
import streamlit as st

from profiler import (
    diagnose_all, load_table, predict_fanout, intersect_coverage,
)
from stats_advisor import (
    MIN_SAMPLE, bonferroni, classify_variable, family_error_rate,
    recommend_test, run_test, stratified_compare,
)

st.set_page_config(page_title="데이터셋 구조 진단기", page_icon="🔍", layout="wide")

RISK_COLOR = {"위험": "🔴", "주의": "🟡", "안전": "🟢"}
TYPE_HELP = {
    "마스터": "1행 = 개체 1개. 비율 지표의 **분모** 후보입니다.",
    "이벤트 로그": "1행 = 사건 1건. **건수 지표**의 원천이며, 조인 시 행이 불어납니다.",
    "패널/스냅샷": "1행 = 개체 × 기간. **시계열 추이**와 상대 시점 정렬이 가능합니다.",
    "집계 테이블": "이미 group by된 표. 개별 개체로 쪼갤 수 없습니다.",
    "부속 테이블(날짜 없음)": "다른 표에 딸린 속성 정보입니다.",
    "판별 보류": "이 표만으로는 유형을 확정할 수 없습니다. "
                "이 키를 참조하는 다른 표를 함께 올리면 판별됩니다.",
}


# ------------------------------------------------------------------
# 사이드바 — 업로드
# ------------------------------------------------------------------
st.sidebar.title("🔍 데이터셋 구조 진단기")
st.sidebar.caption("컬럼 의미를 모르는 상태에서 구조만으로 진단합니다.")

files = st.sidebar.file_uploader(
    "CSV 파일 (여러 개 가능)", type=["csv"], accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**데이터는 외부로 전송되지 않습니다.** "
    "모든 계산이 이 컴퓨터에서 이뤄집니다."
)

if not files:
    st.title("데이터셋 구조 진단기")
    st.markdown(
        """
        분석을 시작하기 전에 **이 데이터로 무엇을 할 수 있고 어디가 위험한지**를 먼저 봅니다.

        | 진단 | 내용 |
        |---|---|
        | **구조** | 1행이 무엇인지(그레인), 테이블 유형, 쓸모없는 컬럼 |
        | **관계** | 조인키 후보, 조인하면 행이 몇 배로 불어나는지 |
        | **유효구간** | 각 표가 커버하는 기간, 여러 표를 함께 쓸 때의 공통 구간 |

        컬럼명이 `customer_id`든 `VAR001`이든 동일하게 동작합니다.
        의미가 아니라 구조만 보기 때문입니다.

        👈 왼쪽에서 CSV를 올려 시작하세요.
        """
    )
    st.stop()


# ------------------------------------------------------------------
# 로딩
# ------------------------------------------------------------------
@st.cache_data(show_spinner="파일 읽는 중…")
def _load(raw_files):
    tables, metas, errors = {}, {}, []
    for name, content in raw_files:
        import io
        try:
            df, meta = load_table(io.BytesIO(content), name)
            tables[name], metas[name] = df, meta
        except Exception as e:
            errors.append((name, str(e)))
    return tables, metas, errors


@st.cache_data(show_spinner="구조 진단 중…")
def _diagnose(_tables, cache_key):
    # _tables는 해시 대상에서 제외되므로(언더스코어 규칙),
    # 업로드 파일이 바뀐 것을 알리는 cache_key를 반드시 함께 넘긴다.
    return diagnose_all(_tables)


raw = [(f.name.rsplit(".", 1)[0], f.getvalue()) for f in files]
tables, metas, errors = _load(raw)
cache_key = tuple(sorted((n, len(b)) for n, b in raw))

for name, err in errors:
    st.error(f"**{name}** 읽기 실패 — {err}")

if not tables:
    st.stop()

diags, cands = _diagnose(tables, cache_key)

tab1, tab2, tab3, tab5, tab4 = st.tabs(
    ["📋 구조 진단", "🔗 관계 진단", "📅 유효구간", "🧪 검정 추천", "📝 요약"]
)


# ==================================================================
# TAB 1 — 구조
# ==================================================================
with tab1:
    st.subheader("테이블별 구조")

    summary = pd.DataFrame([
        {
            "테이블": d["name"],
            "행": f"{d['rows']:,}",
            "열": d["cols"],
            "유형": d["type"],
            "그레인": " + ".join(d["grain"].columns) if d["grain"].columns else "—",
        }
        for d in diags.values()
    ])
    st.dataframe(summary, hide_index=True, width="stretch")

    st.markdown("---")

    for name, d in diags.items():
        g = d["grain"]
        with st.expander(f"**{name}**  ·  {d['type']}  ·  {d['rows']:,}행 × {d['cols']}열"):
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown(f"**유형: {d['type']}**")
                st.caption(d["type_reason"])
                if d["type"] in TYPE_HELP:
                    st.info(TYPE_HELP[d["type"]], icon="💡")

            with c2:
                st.markdown("**그레인 (1행 = 무엇인가)**")
                if g.columns:
                    st.code(" + ".join(g.columns), language=None)
                    if g.density:
                        st.caption(f"밀도 {g.density:.2f} · 조합 {g.checked_combos}회 검사")
                else:
                    st.warning("유일한 컬럼 조합을 찾지 못했습니다.")
                if g.surrogate_keys:
                    st.caption(f"대리키(일련번호) 후보: {', '.join(g.surrogate_keys)}")
                if g.duplicate_rows:
                    st.error(f"완전히 동일한 중복 행 {g.duplicate_rows}개")
                if g.note:
                    st.caption(g.note)

            if g.rejected:
                with st.popover("우연히 유일해진 조합 (그레인 아님)"):
                    for cols, dens in g.rejected[:5]:
                        st.caption(f"`{' + '.join(cols)}` — 밀도 {dens:.4f}")
                    st.caption(
                        "고유값 곱이 행 수보다 훨씬 커서 우연히 유일해진 조합입니다. "
                        "구조적 의미가 없습니다."
                    )

            for w in metas.get(name, {}).get("warnings", []):
                st.warning(w)

            st.markdown("**컬럼**")
            prof = d["profile"]
            st.dataframe(prof, hide_index=True, width="stretch")

            bad = prof[prof["플래그"].str.contains("분산0|전량결측", na=False)]
            if len(bad):
                st.error(
                    "**지표로 쓸 수 없는 컬럼**: "
                    + ", ".join(f"`{c}`" for c in bad["컬럼"])
                    + " — 모든 행이 같은 값이거나 전부 비어 있어 어떤 것도 구분하지 못합니다."
                )


# ==================================================================
# TAB 2 — 관계
# ==================================================================
with tab2:
    st.subheader("조인키 후보")
    st.caption("컬럼명이 아니라 **값이 얼마나 겹치는지**로 찾습니다. 이름이 달라도 찾아냅니다.")

    if not cands:
        st.info("테이블이 하나뿐이거나, 값이 겹치는 컬럼을 찾지 못했습니다.")
    else:
        rows = []
        for c in cands:
            rows.append({
                "왼쪽": f"{c.left_table}.{c.left_col}",
                "오른쪽": f"{c.right_table}.{c.right_col}",
                "값 겹침": f"{c.overlap:.0%}",
                "관계": f"{'1' if c.left_unique else 'N'}:{'1' if c.right_unique else 'N'}",
                "이름 일치": "○" if c.same_name else "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)

    st.markdown("---")
    st.subheader("조인 시뮬레이터 — 조인하기 전에 결과를 봅니다")

    names = list(tables)
    c1, c2, c3, c4 = st.columns(4)
    lt = c1.selectbox("왼쪽 테이블", names, key="lt")
    lc = c2.selectbox("왼쪽 컬럼", list(tables[lt].columns), key="lc")
    rt = c3.selectbox("오른쪽 테이블", names,
                      index=min(1, len(names) - 1), key="rt")
    rc = c4.selectbox("오른쪽 컬럼", list(tables[rt].columns), key="rc")

    if lt == rt and lc == rc:
        st.info("서로 다른 테이블을 선택하세요.")
    else:
        r = predict_fanout(tables[lt], lc, tables[rt], rc)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("왼쪽 행 수", f"{r.left_rows:,}")
        m2.metric("오른쪽 행 수", f"{r.right_rows:,}")
        m3.metric("조인 후 행 수", f"{r.joined_rows:,}",
                  delta=f"{r.fanout_factor:.2f}배",
                  delta_color="inverse" if r.fanout_factor > 1.05 else "normal")
        m4.metric("관계", r.relation)

        st.markdown(f"### {RISK_COLOR.get(r.risk, '')} {r.risk} — {r.note}")

        if r.left_dropped or r.right_dropped:
            st.warning(
                f"**INNER JOIN 시 사라지는 행** — "
                f"왼쪽 {r.left_dropped:,}행 · 오른쪽 {r.right_dropped:,}행. "
                "누락이 문제라면 LEFT JOIN을 쓰고, 사라지는 행이 어떤 집단인지 먼저 확인하세요."
            )

        if r.fanout_factor > 1.05:
            st.error(
                f"**SUM·AVG·COUNT를 그대로 쓰면 안 됩니다.** "
                f"조인 후 `{lt}`의 각 행이 평균 {r.fanout_factor:.2f}번 복제됩니다. "
                "합계는 그만큼 부풀려지고, 평균은 행이 많은 개체 쪽으로 쏠립니다.\n\n"
                "**해결**: 조인 전에 각 테이블을 먼저 집계한 뒤 붙이세요."
            )


# ==================================================================
# TAB 3 — 유효구간
# ==================================================================
with tab3:
    st.subheader("테이블별 커버 기간")

    all_cov = [c for d in diags.values() for c in d["coverage"]]

    if not all_cov:
        st.info("날짜로 해석되는 컬럼을 찾지 못했습니다. 기간 분석은 불가능합니다.")
    else:
        cov_rows = [{
            "테이블": c.table, "컬럼": c.column, "단위": c.granularity,
            "시작": c.start, "종료": c.end,
            "관측 구간": f"{c.periods_present}/{c.periods_expected}",
            "빈 구간": len(c.gaps),
        } for c in all_cov]
        st.dataframe(pd.DataFrame(cov_rows), hide_index=True, width="stretch")

        for c in all_cov:
            if c.gaps or c.note:
                with st.expander(f"⚠️ {c.table}.{c.column} — 확인 필요"):
                    if c.note:
                        st.warning(c.note)
                    if c.gaps:
                        st.markdown(
                            f"**기록이 없는 구간 {len(c.gaps)}개** "
                            f"(최대 24개까지 표시)"
                        )
                        st.code(", ".join(c.gaps), language=None)
                        st.caption(
                            "이 구간이 **'값이 0'인지 '기록이 없는 것'인지는 데이터만으로 알 수 없습니다.** "
                            "업무 담당자에게 확인해야 합니다. 기록 공백을 0으로 보고하면 "
                            "'그 달에는 아무 일도 없었다'는 거짓 문장이 만들어집니다."
                        )

        st.markdown("---")
        inter = intersect_coverage(all_cov)
        if inter:
            st.subheader("여러 표를 함께 쓸 때의 공통 구간")
            if inter["valid"]:
                st.success(
                    f"**{inter['start']} ~ {inter['end']}** — "
                    "이 구간 밖은 일부 표에 데이터가 없어 계산해도 값이 유효하지 않습니다."
                )
            else:
                st.error("공통 구간이 없습니다. 이 표들을 같은 기간으로 묶을 수 없습니다.")
            st.dataframe(pd.DataFrame(inter["detail"]), hide_index=True, width="stretch")


# ==================================================================
# TAB 5 — 검정 추천
# ==================================================================
with tab5:
    st.subheader("검정 방법 추천")
    st.caption(
        "**어떤 검정을 쓸지는 변수 타입의 조합으로 정해집니다.** "
        "결과 변수와 설명 변수만 고르면 나머지는 계산입니다. 모두 로컬에서 실행됩니다."
    )

    if "test_log" not in st.session_state:
        st.session_state.test_log = []

    tname = st.selectbox("테이블", list(tables), key="stat_table")
    tdf = tables[tname]
    tdates = diags[tname]["date_cols"]

    # 검정에 쓸 수 있는 컬럼만 고르게 한다
    usable, blocked = [], []
    for c in tdf.columns:
        v = classify_variable(tdf, c, tdates)
        (usable if v["usable"] else blocked).append((c, v))

    if blocked:
        with st.expander(f"검정에 쓸 수 없는 컬럼 {len(blocked)}개"):
            st.dataframe(
                pd.DataFrame([
                    {"컬럼": c, "타입": v["kind"], "고유값": v["nunique"], "이유": v["note"]}
                    for c, v in blocked
                ]), hide_index=True, width="stretch",
            )

    if len(usable) < 2:
        st.warning("검정에 쓸 수 있는 컬럼이 2개 미만입니다.")
    else:
        names_u = [c for c, _ in usable]
        kind_of = {c: v["kind"] for c, v in usable}

        c1, c2 = st.columns(2)
        outcome = c1.selectbox(
            "결과 변수 (알고 싶은 것)", names_u,
            format_func=lambda c: f"{c}  ·  {kind_of[c]}", key="stat_out",
        )
        explain = c2.selectbox(
            "설명 변수 (원인 후보)", names_u,
            index=min(1, len(names_u) - 1),
            format_func=lambda c: f"{c}  ·  {kind_of[c]}", key="stat_exp",
        )

        rec, ov, evv = recommend_test(tdf, outcome, explain, tdates)

        if rec.blocked:
            st.error(rec.blocked)
        else:
            st.markdown(f"### 추천: **{rec.test}**")
            st.caption(f"{ov['kind']} × {evv['kind']} → {rec.reason}")

            if rec.first_choice:
                st.warning(
                    f"**{rec.first_choice}에서 자동 전환했습니다.** {rec.switched_why}",
                    icon="🔄",
                )
            for w in rec.warnings:
                st.warning(w)

            if st.button("검정 실행", type="primary"):
                res = run_test(tdf, outcome, explain, rec)
                st.session_state.test_log.append(f"{tname}: {outcome} × {explain}")

                m1, m2 = st.columns(2)
                m1.metric("p값", f"{res.p_value:.6f}" if res.p_value >= 1e-6 else f"{res.p_value:.2e}")
                m2.metric("통계량", f"{res.statistic:.4f}")

                if res.p_value < 0.05:
                    st.success(
                        f"우연히 이 정도 차이가 나올 확률이 {res.p_value * 100:.4f}%입니다. "
                        "우연으로 보기 어렵습니다.",
                        icon="✅",
                    )
                else:
                    st.info(
                        f"우연히 이 정도가 나올 확률이 {res.p_value * 100:.1f}%입니다. "
                        "**'차이가 없다'가 아니라 '증거를 찾지 못했다'** 입니다. "
                        "표본이 작아서 못 잡았을 수도 있습니다.",
                        icon="ℹ️",
                    )

                st.markdown("**효과 크기** — p값은 크기를 말해주지 않습니다")
                st.dataframe(
                    pd.DataFrame(
                        [{"지표": k, "값": str(v)} for k, v in res.effect.items()]
                    ), hide_index=True, width="stretch",
                )

                if res.table is not None:
                    st.markdown("**집단별 요약**")
                    st.dataframe(res.table, width="stretch")
                if res.note:
                    st.caption(res.note)

            # ---- 층화
            st.markdown("---")
            st.markdown("#### 층화 비교 — 제3의 변수를 통제합니다")
            st.caption(
                "다른 변수로 나눠도 관계가 유지되는지 봅니다. "
                "방향이 뒤집히면 전체 평균은 집단 구성 때문에 생긴 착시입니다."
            )

            strat_opts = [
                c for c, v in usable
                if c not in (outcome, explain) and v["is_categorical"]
            ]
            if not strat_opts:
                st.info("층으로 나눌 범주형 컬럼이 없습니다.")
            else:
                strat = st.selectbox("층으로 나눌 변수", strat_opts, key="stat_strat")
                sres = stratified_compare(tdf, outcome, explain, strat)

                if sres is None:
                    st.info(
                        "현재 층화 비교는 결과·설명이 모두 2종인 경우만 지원합니다."
                    )
                else:
                    st.dataframe(sres.strata, hide_index=True, width="stretch")
                    if sres.consistent:
                        st.success(sres.note, icon="✅")
                    else:
                        st.error(sres.note, icon="⚠️")

                    thin = (sres.strata["표본충족"] == "×").sum()
                    if thin:
                        st.warning(
                            f"{thin}개 층이 최소표본 {MIN_SAMPLE}에 미달합니다. "
                            "그 층은 결론의 근거로 쓰지 마세요."
                        )

    # ---- 다중비교 카운터
    st.markdown("---")
    k = len(st.session_state.test_log)
    if k:
        c1, c2, c3 = st.columns(3)
        c1.metric("이번 세션 검정 횟수", k)
        c2.metric("본페로니 보정 α", f"{bonferroni(0.05, k):.5f}")
        c3.metric("최소 1회 오류 확률", f"{family_error_rate(0.05, k) * 100:.1f}%")
        if k >= 3:
            st.warning(
                f"**여러 번 검정했습니다.** 관계가 전혀 없어도 {k}번 중 하나가 "
                f"유의하게 나올 확률이 {family_error_rate(0.05, k) * 100:.1f}%입니다. "
                f"결론을 낼 때는 p < {bonferroni(0.05, k):.5f} 기준을 쓰고, "
                "**몇 개를 검정했는지 반드시 기록하세요.**",
                icon="⚠️",
            )
        with st.expander("검정 기록"):
            for i, t in enumerate(st.session_state.test_log, 1):
                st.caption(f"{i}. {t}")
        if st.button("기록 초기화"):
            st.session_state.test_log = []
            st.rerun()
    else:
        st.caption("아직 검정을 실행하지 않았습니다.")


# ==================================================================
# TAB 4 — 요약
# ==================================================================
with tab4:
    st.subheader("진단 요약")

    issues = []
    for name, d in diags.items():
        g = d["grain"]
        if not g.columns:
            issues.append(("🔴", name, "그레인을 특정하지 못했습니다. 중복 행을 먼저 확인하세요."))
        if g.duplicate_rows:
            issues.append(("🔴", name, f"완전 중복 행 {g.duplicate_rows}개"))
        dead = d["profile"][d["profile"]["플래그"].str.contains("분산0|전량결측", na=False)]
        for c in dead["컬럼"]:
            issues.append(("🟡", name, f"`{c}` — 모든 행이 같은 값이라 지표로 쓸 수 없습니다."))
        for c in d["coverage"]:
            if c.gaps:
                issues.append(("🟡", name, f"`{c.column}` 기간에 빈 구간 {len(c.gaps)}개"))
        if d["rows"] < 30:
            issues.append(("🟡", name, f"{d['rows']}행 — 최소표본 30 미달"))

    risky = []
    for c in cands:
        if c.overlap >= 0.5 and not c.left_unique and not c.right_unique:
            r = predict_fanout(tables[c.left_table], c.left_col,
                               tables[c.right_table], c.right_col)
            if r.risk == "위험":
                risky.append(
                    f"`{c.left_table}` ⋈ `{c.right_table}` on `{c.left_col}` → "
                    f"{r.fanout_factor:.2f}배 ({r.joined_rows:,}행)"
                )

    if risky:
        st.error("**N:N 조인 위험 — 합계가 부풀려집니다**")
        for r in risky[:10]:
            st.markdown(f"- {r}")

    if issues:
        st.markdown("**발견된 문제**")
        st.dataframe(
            pd.DataFrame(issues, columns=["", "테이블", "내용"]),
            hide_index=True, width="stretch",
        )
    else:
        st.success("구조적 문제를 찾지 못했습니다.")

    st.markdown("---")
    st.markdown("**이 진단으로 알 수 없는 것**")
    st.info(
        "- 어떤 컬럼이 **무엇을 뜻하는지** — 구조만 봤습니다\n"
        "- 빈 구간이 **'0'인지 '기록 누락'인지** — 업무 지식이 필요합니다\n"
        "- 이 데이터로 **무엇을 물어야 하는지** — 도메인 담당자만 압니다\n\n"
        "여기까지는 기계가 할 수 있는 부분이고, 정의를 확정하는 것은 사람의 몫입니다.",
        icon="ℹ️",
    )
