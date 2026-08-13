"""
데이터셋 구조 진단기 — 핵심 로직.

컬럼의 "의미"는 보지 않는다. 오직 구조만 본다.
따라서 컬럼명이 customer_id든 VAR001이든 동일하게 동작한다.

제공 기능
  1. 인코딩 자동 감지 로딩
  2. 컬럼 프로파일 (결측·고유값·분산0)
  3. 날짜 컬럼 탐지 (여러 포맷)
  4. 그레인 판정 (최소 유일 컬럼 조합)
  5. 테이블 유형 분류 (마스터/이벤트/패널/집계)
  6. 조인키 후보 탐색 (값 겹침 기반)
  7. 팬아웃 예측 (조인 후 행 수를 실제로 계산)
  8. 유효구간 계산 (커버 기간 + 빈 구간)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 설정값 — 조합 폭발과 메모리 초과를 막는 상한
# ------------------------------------------------------------------
GRAIN_MAX_COMBO = 3       # 그레인 탐색 시 최대 컬럼 조합 크기
GRAIN_MAX_CANDIDATES = 10  # 그레인 후보로 볼 최대 컬럼 수
GRAIN_SAMPLE_ROWS = 200_000  # 이보다 크면 표본으로 좁힌 뒤 전체 검증

# 밀도 = 실제 행 수 / (조합 컬럼들의 고유값 곱)
# 진짜 그레인(500명 × 12개월 = 6,000행)은 1.0에 가깝고,
# 우연히 유일해진 조합(이름 344종 × 가입일 452종 = 155,488 >> 500행)은 0에 가깝다.
MIN_GRAIN_DENSITY = 0.20
JOIN_MIN_OVERLAP = 0.30    # 조인키 후보로 볼 최소 값 겹침 비율
JOIN_MAX_PAIRS = 4_000     # 컬럼 쌍 비교 상한
JOIN_MIN_DISTINCT = 20     # 키로 인정할 최소 고유값 수
# duration_min(1~60), csat(1~5) 같은 측정값은 다른 숫자 컬럼과 우연히 겹친다.
# 고유값이 적은 숫자 컬럼은 키 후보에서 제외한다.

ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]

# 날짜로 볼 문자열 패턴 (정규식으로 먼저 거른 뒤 파싱 — 오탐 방지)
DATE_PATTERNS = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}"), "day", None),
    (re.compile(r"^\d{4}/\d{2}/\d{2}"), "day", None),
    (re.compile(r"^\d{4}\.\d{2}\.\d{2}"), "day", None),
    (re.compile(r"^\d{4}-\d{2}$"), "month", "%Y-%m"),
    (re.compile(r"^\d{4}/\d{2}$"), "month", "%Y/%m"),
    (re.compile(r"^\d{8}$"), "day", "%Y%m%d"),
    (re.compile(r"^\d{6}$"), "month", "%Y%m"),
]


# ==================================================================
# 1. 로딩
# ==================================================================
def load_table(source, name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """인코딩을 순차 시도해 CSV를 읽는다. 실패 이유도 함께 돌려준다."""
    meta = {"name": name, "encoding": None, "warnings": []}
    last_err = None

    for enc in ENCODINGS:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc, low_memory=False)
            meta["encoding"] = enc
            if enc == "latin1":
                meta["warnings"].append(
                    "latin1로 읽혔습니다. 한글이 깨질 수 있으니 원본 인코딩을 확인하세요."
                )
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:  # 파싱 자체 실패
            raise ValueError(f"CSV로 읽을 수 없습니다: {e}") from e
    else:
        raise ValueError(f"인코딩을 판별하지 못했습니다: {last_err}")

    if len(df) == 0:
        meta["warnings"].append("행이 없습니다.")
    if len(df) < 30:
        meta["warnings"].append(
            f"행이 {len(df)}개뿐입니다. 최소표본 30 기준에 미달해 통계 판단에 쓸 수 없습니다."
        )
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        meta["warnings"].append(f"중복된 컬럼명이 있습니다: {dups}")

    return df, meta


# ==================================================================
# 2. 날짜 컬럼 탐지
# ==================================================================
def detect_datetime_columns(df: pd.DataFrame, sample: int = 500) -> dict[str, dict]:
    """날짜/연월로 해석 가능한 컬럼을 찾는다. 컬럼명은 보지 않는다."""
    found = {}

    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        # 이미 datetime 타입
        if pd.api.types.is_datetime64_any_dtype(s):
            found[col] = {"granularity": "day", "format": None, "match_rate": 1.0}
            continue

        # 순수 숫자형인데 날짜 범위를 벗어나면 제외 (금액·수량 오탐 방지)
        probe = s.sample(min(sample, len(s)), random_state=0).astype(str).str.strip()

        for pattern, gran, fmt in DATE_PATTERNS:
            rate = probe.str.match(pattern).mean()
            if rate < 0.95:
                continue

            # 8자리/6자리 숫자는 실제 날짜 범위인지 한 번 더 확인
            if fmt in ("%Y%m%d", "%Y%m"):
                yrs = pd.to_numeric(probe.str[:4], errors="coerce")
                if not ((yrs >= 1900) & (yrs <= 2100)).mean() > 0.95:
                    continue

            parsed = pd.to_datetime(probe, format=fmt, errors="coerce")
            if parsed.notna().mean() >= 0.95:
                found[col] = {
                    "granularity": gran,
                    "format": fmt,
                    "match_rate": float(parsed.notna().mean()),
                }
                break

    return found


def parse_dates(series: pd.Series, info: dict) -> pd.Series:
    """탐지 결과에 맞춰 실제 파싱한다."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    s = series.astype(str).str.strip()
    return pd.to_datetime(s, format=info["format"], errors="coerce")


# ==================================================================
# 3. 컬럼 프로파일
# ==================================================================
def profile_columns(df: pd.DataFrame, date_cols: dict) -> pd.DataFrame:
    """컬럼별 기본 통계. 여기서 '분산 0'과 '전량 결측'이 걸러진다."""
    n = len(df)
    rows = []

    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        n_null = int(s.isna().sum())

        if col in date_cols:
            kind = f"날짜({date_cols[col]['granularity']})"
        elif pd.api.types.is_numeric_dtype(s):
            # 숫자인데 고유값이 적으면 사실상 범주형 (등급·플래그 등)
            kind = "범주형(숫자코드)" if 0 < nunique <= 10 else "연속형"
        elif pd.api.types.is_bool_dtype(s):
            kind = "이진"
        else:
            kind = "범주형" if nunique <= max(50, n * 0.05) else "텍스트/식별자"

        flags = []
        if n and nunique == 1:
            flags.append("분산0")
        if n and n_null == n:
            flags.append("전량결측")
        if n and nunique == n and n_null == 0:
            flags.append("고유키후보")
        if n and 0 < n_null / n >= 0.5:
            flags.append("결측50%↑")

        rows.append(
            {
                "컬럼": col,
                "타입": kind,
                "고유값": nunique,
                "결측": n_null,
                "결측률": round(n_null / n * 100, 1) if n else 0.0,
                "예시": _sample_value(s),
                "플래그": " · ".join(flags),
            }
        )

    return pd.DataFrame(rows)


def _sample_value(s: pd.Series) -> str:
    nz = s.dropna()
    if len(nz) == 0:
        return ""
    v = str(nz.iloc[0])
    return v[:30] + "…" if len(v) > 30 else v


# ==================================================================
# 4. 그레인 판정
# ==================================================================
@dataclass
class GrainResult:
    columns: list[str] | None          # 최종 채택 그레인 (자연 그레인 우선)
    surrogate_keys: list[str]          # 단독으로 유일한 컬럼 = 대리키 후보
    natural: list[str] | None          # 대리키를 뺀 뒤 찾은 실질 그레인
    duplicate_rows: int                # 완전 중복 행 수
    checked_combos: int
    density: float = 0.0               # 자연 그레인의 밀도 (1.0에 가까울수록 구조적)
    rejected: list[tuple[list[str], float]] = field(default_factory=list)
    note: str = ""

    @property
    def is_exact(self) -> bool:
        return self.columns is not None


def _density(df: pd.DataFrame, cols: list[str]) -> float:
    """행 수 / 고유값 곱. 조합이 '빽빽하게' 채워져 있는지를 본다."""
    prod = 1.0
    for c in cols:
        prod *= max(df[c].nunique(), 1)
    return len(df) / prod if prod else 0.0


def _search_unique_combo(df, cands, max_combo, probe):
    """
    유일성을 만드는 최소 조합을 찾되, 밀도가 낮은 조합은 '우연'으로 보고 버린다.
    버린 조합도 함께 돌려줘서 사용자가 판단할 수 있게 한다.
    """
    checked = 0
    rejected = []
    for size in range(1, min(max_combo, len(cands)) + 1):
        for combo in combinations(cands, size):
            checked += 1
            cols = list(combo)
            if probe.duplicated(subset=cols).any():
                continue
            if df.duplicated(subset=cols).any():
                continue

            d = _density(df, cols)
            if len(cols) > 1 and d < MIN_GRAIN_DENSITY:
                rejected.append((cols, d))   # 우연히 유일해진 조합
                continue
            return cols, checked, d, rejected
    return None, checked, 0.0, rejected


def find_grain(df: pd.DataFrame, date_cols: dict | None = None,
               max_combo: int = GRAIN_MAX_COMBO) -> GrainResult:
    """
    1행이 무엇인지를 찾는다 = 어떤 컬럼 조합이 유일한가.

    주의: usage_id·consult_id 같은 일련번호(대리키)가 있으면 그 하나로 항상
    유일해져서 "1행 = 개체 1개"라는 잘못된 결론이 나온다. 그래서 대리키를
    따로 걸러내고, 그것을 제외한 '자연 그레인'을 함께 찾는다.
    """
    n = len(df)
    if n == 0:
        return GrainResult(None, [], None, 0, 0, "행이 없습니다.")

    date_cols = date_cols or {}
    dup_rows = int(df.duplicated().sum())

    def eligible(c) -> bool:
        s = df[c]
        if s.isna().sum() > 0:          # 키는 결측일 수 없다
            return False
        nu = s.nunique(dropna=True)
        if not (1 < nu <= n):
            return False
        if c in date_cols:
            return True
        # 숫자 컬럼은 고유값이 충분히 많을 때만 키로 본다.
        # reward_amount(3종)·age·csat 같은 측정값이 그레인에 끼는 것을 막는다.
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return nu >= max(0.3 * n, 20)
        return True

    cands = [c for c in df.columns if eligible(c)]
    cands.sort(key=lambda c: df[c].nunique(), reverse=True)
    cands = cands[:GRAIN_MAX_CANDIDATES]

    if not cands:
        return GrainResult(None, [], None, dup_rows, 0,
                           note="키가 될 만한 컬럼이 없습니다.")

    probe = df if n <= GRAIN_SAMPLE_ROWS else df.sample(GRAIN_SAMPLE_ROWS, random_state=0)

    # 단독으로 유일한 컬럼 = 대리키 후보
    surrogates = [c for c in cands if df[c].nunique() == n]

    # 대리키를 뺀 나머지로 자연 그레인을 찾는다
    rest = [c for c in cands if c not in surrogates]
    natural, checked, dens, rejected = _search_unique_combo(df, rest, max_combo, probe)

    # 대리키가 따로 있는데 자연 그레인에 날짜가 없다면 우연일 가능성이 높다.
    # (subscription_events의 customer_id + event_type 같은 조합)
    if natural and surrogates and len(natural) > 1 and date_cols:
        if not any(c in date_cols for c in natural):
            rejected.append((natural, dens))
            natural = None

    if natural:
        note = f"밀도 {dens:.2f} — 구조적 그레인으로 판정."
        if surrogates:
            note += f" 대리키 {surrogates} 제외."
        return GrainResult(natural, surrogates, natural, dup_rows, checked,
                           dens, rejected, note)

    # 자연 그레인이 없다 → 대리키가 유일한 키
    if surrogates:
        note = (
            f"`{surrogates[0]}`는 행마다 다른 일련번호(대리키)로 보입니다. "
            "이를 제외하면 구조적으로 유일한 조합이 없어, 같은 개체가 여러 행에 나타납니다."
        )
        if rejected:
            cols, d = rejected[0]
            note += (
                f" ({' + '.join(cols)} 조합이 유일하긴 하나 밀도가 {d:.4f}로 낮아 "
                "우연으로 판단했습니다.)"
            )
        return GrainResult([surrogates[0]], surrogates, None, dup_rows, checked,
                           0.0, rejected, note)

    return GrainResult(
        None, [], None, dup_rows, checked, 0.0, rejected,
        f"{max_combo}개 이하 조합으로는 유일성을 만들지 못했습니다.",
    )


# ==================================================================
# 5. 테이블 유형 분류
# ==================================================================
def classify_table(df: pd.DataFrame, grain: GrainResult, date_cols: dict) -> tuple[str, str]:
    """
    마스터 / 이벤트로그 / 패널 / 집계 중 하나로 분류하고 근거를 돌려준다.
    이 분류가 이후 지표·검정 추천의 출발점이 된다.
    """
    n = len(df)
    if n == 0:
        return "판정불가", "행이 없습니다."

    has_date = len(date_cols) > 0
    natural = grain.natural          # 대리키를 뺀 실질 그레인
    surro = grain.surrogate_keys

    # ── 1. 자연 그레인이 (개체 + 기간) 2개 조합 → 패널 또는 집계
    if natural and len(natural) == 2:
        date_part = [c for c in natural if c in date_cols]
        id_part = [c for c in natural if c not in date_cols]
        if date_part and id_part:
            gran = date_cols[date_part[0]]["granularity"]
            n_entity = df[id_part[0]].nunique()
            # 개체 수가 적으면 개체가 아니라 '구분값'이다 → 이미 집계된 표
            if n_entity <= 20:
                return "집계 테이블", (
                    f"`{id_part[0]}`({n_entity}종) × `{date_part[0]}`({gran}) 조합으로 유일합니다. "
                    "개체 식별자 없이 구분·기간별로 이미 group by된 표입니다."
                )
            return "패널/스냅샷", (
                f"`{id_part[0]}`({n_entity:,}개) × `{date_part[0]}`({gran}) 조합으로 유일합니다. "
                "개체를 기간마다 반복 관측한 구조입니다."
            )

    # ── 2. 자연 그레인이 단일 비날짜 컬럼 → 마스터
    if natural and len(natural) == 1 and natural[0] not in date_cols:
        return "마스터", (
            f"`{natural[0]}` 하나로 유일합니다. 1행 = 개체 1개."
            + (f" (대리키 {surro} 별도 존재)" if surro else "")
        )

    # ── 3. 대리키만 유일하고 자연 그레인이 없음 → 같은 개체가 여러 행
    if surro and not natural:
        # 반복되는 개체 식별자 후보를 찾는다
        repeat_ids = [
            c for c in df.columns
            if c not in surro and c not in date_cols
            and 1 < df[c].nunique() < n * 0.9
            and not pd.api.types.is_float_dtype(df[c])
        ]
        repeat_ids.sort(key=lambda c: df[c].nunique(), reverse=True)

        if has_date and repeat_ids:
            top = repeat_ids[0]
            return "이벤트 로그", (
                f"대리키 `{surro[0]}` 외에는 유일한 조합이 없고, "
                f"`{top}`가 반복되며 날짜 컬럼이 있습니다. 1행 = 사건 1건."
            )
        if repeat_ids:
            # 날짜가 없으면 '마스터 + 범주 속성'과 '이벤트 로그'를 구분할 수 없다.
            # 다른 표와의 참조 관계가 있어야 판별된다 (_refine_type에서 처리).
            return "판별 보류", (
                f"`{surro[0]}`가 유일하고 `{repeat_ids[0]}` 등이 반복됩니다. "
                "날짜 컬럼이 없어 **마스터인지 이벤트 로그인지 이 표만으로는 알 수 없습니다.** "
                "이 키를 참조하는 다른 표를 함께 올리면 판별됩니다."
            )
        return "기타", f"`{surro[0]}` 외에 구조를 특정할 단서가 없습니다."

    # ── 4. 유일 조합 자체가 없음
    if not natural and not surro:
        if has_date:
            return "집계 테이블(중복 존재)", (
                "유일한 컬럼 조합이 없습니다. 이미 집계된 표이거나 중복 행이 있습니다."
            )
        return "기타", grain.note or "표준 유형에 들어맞지 않습니다."

    return "기타", grain.note or "표준 유형에 들어맞지 않습니다."


# ==================================================================
# 6. 조인키 후보 탐색
# ==================================================================
@dataclass
class JoinCandidate:
    left_table: str
    left_col: str
    right_table: str
    right_col: str
    overlap: float          # 겹치는 값 / 작은 쪽 고유값
    left_unique: bool
    right_unique: bool
    same_name: bool
    left_n: int = 0
    right_n: int = 0


def _norm_keys(s: pd.Series) -> set:
    """타입 불일치(123 vs '00123')를 흡수하기 위해 문자열로 정규화한다."""
    v = s.dropna()
    if pd.api.types.is_float_dtype(v):
        # 1.0 → '1' (정수로 떨어지는 실수만)
        if (v % 1 == 0).all():
            v = v.astype("int64")
    return set(v.astype(str).str.strip())


def find_join_candidates(tables: dict[str, pd.DataFrame]) -> list[JoinCandidate]:
    """테이블 쌍마다 값이 겹치는 컬럼을 찾는다. 컬럼명이 달라도 찾아낸다."""
    # 미리 정규화한 값 집합을 캐시 (반복 계산 방지)
    cache: dict[tuple[str, str], set] = {}

    def keys(t, c):
        if (t, c) not in cache:
            cache[(t, c)] = _norm_keys(tables[t][c])
        return cache[(t, c)]

    def key_like(t, c) -> bool:
        """측정값 컬럼을 키 후보에서 제외한다."""
        s = tables[t][c]
        if pd.api.types.is_float_dtype(s):
            return False          # 금액·비율 등 연속값은 키가 아니다
        if pd.api.types.is_bool_dtype(s):
            return False
        return len(keys(t, c)) >= JOIN_MIN_DISTINCT

    out: list[JoinCandidate] = []
    pairs_checked = 0

    for t1, t2 in combinations(tables.keys(), 2):
        df1, df2 = tables[t1], tables[t2]
        for c1 in df1.columns:
            if not key_like(t1, c1):
                continue
            k1 = keys(t1, c1)
            for c2 in df2.columns:
                if not key_like(t2, c2):
                    continue
                if pairs_checked >= JOIN_MAX_PAIRS:
                    return sorted(out, key=lambda x: (-x.same_name, -x.overlap))
                pairs_checked += 1

                k2 = keys(t2, c2)
                inter = len(k1 & k2)
                if inter == 0:
                    continue
                overlap = inter / min(len(k1), len(k2))
                same_name = c1 == c2

                if overlap < JOIN_MIN_OVERLAP and not same_name:
                    continue

                out.append(
                    JoinCandidate(
                        left_table=t1, left_col=c1,
                        right_table=t2, right_col=c2,
                        overlap=overlap,
                        left_unique=len(k1) == len(df1.dropna(subset=[c1])),
                        right_unique=len(k2) == len(df2.dropna(subset=[c2])),
                        same_name=same_name,
                        left_n=len(k1), right_n=len(k2),
                    )
                )

    return sorted(out, key=lambda x: (-x.same_name, -x.overlap))


# ==================================================================
# 7. 팬아웃 예측 — 추정이 아니라 실제 계산
# ==================================================================
@dataclass
class FanoutResult:
    left_rows: int
    right_rows: int
    joined_rows: int          # inner join 후 행 수 (정확값)
    fanout_factor: float      # joined / left
    relation: str             # 1:1 / 1:N / N:1 / N:N
    left_dropped: int         # 오른쪽에 없어서 사라지는 왼쪽 행
    right_dropped: int
    risk: str
    note: str = ""


def predict_fanout(
    left: pd.DataFrame, left_col: str,
    right: pd.DataFrame, right_col: str,
) -> FanoutResult:
    """
    조인 전에 조인 후 행 수를 계산한다.
    실제로 조인하지 않고 키별 개수의 곱으로 구하므로 메모리가 터지지 않는다.
    """
    lc = left[left_col].dropna()
    rc = right[right_col].dropna()

    if pd.api.types.is_float_dtype(lc) and (lc % 1 == 0).all():
        lc = lc.astype("int64")
    if pd.api.types.is_float_dtype(rc) and (rc % 1 == 0).all():
        rc = rc.astype("int64")

    l_counts = lc.astype(str).str.strip().value_counts()
    r_counts = rc.astype(str).str.strip().value_counts()

    shared = l_counts.index.intersection(r_counts.index)
    joined = int((l_counts[shared] * r_counts[shared]).sum())

    l_max = int(l_counts.max()) if len(l_counts) else 0
    r_max = int(r_counts.max()) if len(r_counts) else 0
    relation = f"{'1' if l_max == 1 else 'N'}:{'1' if r_max == 1 else 'N'}"

    left_dropped = int(len(left) - l_counts[shared].sum())
    right_dropped = int(len(right) - r_counts[shared].sum())
    factor = joined / len(left) if len(left) else 0.0

    if relation == "N:N":
        risk = "위험"
        note = "양쪽 모두 키가 중복됩니다. 카티션 곱이 발생해 합계·평균이 부풀려집니다."
    elif relation in ("1:N", "N:1") and factor > 1.5:
        risk = "주의"
        note = f"행이 {factor:.2f}배로 늘어납니다. SUM/AVG를 그대로 쓰면 왜곡됩니다."
    elif left_dropped > 0:
        risk = "주의"
        note = f"INNER JOIN 시 왼쪽 {left_dropped}행이 사라집니다."
    else:
        risk = "안전"
        note = "행 수가 보존됩니다."

    return FanoutResult(
        left_rows=len(left), right_rows=len(right), joined_rows=joined,
        fanout_factor=factor, relation=relation,
        left_dropped=left_dropped, right_dropped=right_dropped,
        risk=risk, note=note,
    )


# ==================================================================
# 8. 유효구간
# ==================================================================
@dataclass
class Coverage:
    table: str
    column: str
    granularity: str
    start: str
    end: str
    periods_present: int
    periods_expected: int
    gaps: list[str] = field(default_factory=list)
    note: str = ""


def compute_coverage(name: str, df: pd.DataFrame, date_cols: dict) -> list[Coverage]:
    """날짜 컬럼별 커버 기간과 빈 구간을 찾는다."""
    out = []
    for col, info in date_cols.items():
        parsed = parse_dates(df[col], info).dropna()
        if len(parsed) == 0:
            continue

        freq = "MS" if info["granularity"] == "month" else "D"
        periods = parsed.dt.to_period("M" if freq == "MS" else "D")
        present = set(periods.unique())
        full = pd.period_range(periods.min(), periods.max(),
                               freq="M" if freq == "MS" else "D")
        missing = [str(p) for p in full if p not in present]

        note = ""
        n_null = int(df[col].isna().sum())
        if n_null:
            note = f"결측 {n_null}행({n_null / len(df) * 100:.1f}%). 기록 공백인지 확인 필요."

        out.append(
            Coverage(
                table=name, column=col, granularity=info["granularity"],
                start=str(periods.min()), end=str(periods.max()),
                periods_present=len(present), periods_expected=len(full),
                gaps=missing[:24], note=note,
            )
        )
    return out


def intersect_coverage(covs: list[Coverage]) -> dict:
    """여러 테이블을 함께 쓸 때의 공통 사용 가능 구간(교집합)."""
    monthly = [c for c in covs if c.granularity in ("month", "day")]
    if len(monthly) < 2:
        return {}

    starts, ends = [], []
    for c in monthly:
        starts.append(pd.Period(c.start[:7], freq="M"))
        ends.append(pd.Period(c.end[:7], freq="M"))

    lo, hi = max(starts), min(ends)
    return {
        "start": str(lo),
        "end": str(hi),
        "valid": lo <= hi,
        "binding_start": monthly[int(np.argmax([str(s) for s in starts].index(str(lo))))].table
        if starts else None,
        "detail": [
            {"테이블": c.table, "컬럼": c.column, "시작": c.start[:7], "종료": c.end[:7]}
            for c in monthly
        ],
    }


# ==================================================================
# 통합 진단
# ==================================================================
def diagnose(name: str, df: pd.DataFrame) -> dict:
    """테이블 하나에 대한 전체 구조 진단 (단일 테이블 기준)."""
    date_cols = detect_datetime_columns(df)
    grain = find_grain(df, date_cols)
    ttype, reason = classify_table(df, grain, date_cols)

    return {
        "name": name,
        "rows": len(df),
        "cols": len(df.columns),
        "date_cols": date_cols,
        "grain": grain,
        "type": ttype,
        "type_reason": reason,
        "profile": profile_columns(df, date_cols),
        "coverage": compute_coverage(name, df, date_cols),
        "foreign_keys": [],
        "referenced_by": [],
    }


def diagnose_all(tables: dict[str, pd.DataFrame]) -> tuple[dict, list[JoinCandidate]]:
    """
    여러 테이블을 함께 진단한다.

    단일 테이블만 보면 마스터와 이벤트 로그를 구분할 수 없다.
    (customers도 consultations도 "유일한 ID 하나 + 반복되는 컬럼들"로 보인다)
    구분의 열쇠는 테이블 간 참조 관계다 — 다른 표의 유일키를 가리키는
    컬럼이 있으면 그 표는 이벤트 로그다.
    """
    diags = {name: diagnose(name, df) for name, df in tables.items()}
    cands = find_join_candidates(tables)

    def primary_keys(t: str) -> set[str]:
        """그 표를 대표하는 키. 참조 대상이 될 수 있는 유일한 컬럼."""
        g = diags[t]["grain"]
        pk = set(g.surrogate_keys[:1])
        if g.columns:
            pk |= set(g.columns)
        return pk

    # 각 테이블에서 "다른 표의 기본키를 가리키는 컬럼"(외래키)을 찾는다.
    # 상대 쪽에서 유일하기만 하면 안 되고, 그 표의 기본키여야 한다.
    # (referrals.referred_customer_id는 유일하지만 그 표의 기본키가 아니다)
    for c in cands:
        if c.overlap < 0.5:
            continue
        if c.right_unique and not c.left_unique and c.right_col in primary_keys(c.right_table):
            diags[c.left_table]["foreign_keys"].append((c.left_col, c.right_table))
            diags[c.right_table]["referenced_by"].append((c.right_col, c.left_table))
        if c.left_unique and not c.right_unique and c.left_col in primary_keys(c.left_table):
            diags[c.right_table]["foreign_keys"].append((c.right_col, c.left_table))
            diags[c.left_table]["referenced_by"].append((c.left_col, c.right_table))

    # 참조 관계를 반영해 유형을 다시 판정한다
    for name, d in diags.items():
        d["type"], d["type_reason"] = _refine_type(d, tables[name])

    return diags, cands


def _refine_type(d: dict, df: pd.DataFrame) -> tuple[str, str]:
    """테이블 간 참조 관계를 반영한 최종 유형 판정."""
    grain, ttype = d["grain"], d["type"]
    fks = d["foreign_keys"]
    refs = d["referenced_by"]

    # 패널·집계는 그레인만으로 확정된다 — 그대로 둔다
    if ttype in ("패널/스냅샷", "집계 테이블", "판정불가", "기타"):
        return ttype, d["type_reason"]

    # 대리키만 유일한 경우가 갈림길이다
    if grain.surrogate_keys and not grain.natural:
        key = grain.surrogate_keys[0]

        if fks:
            fk_txt = ", ".join(f"`{c}`→{t}" for c, t in dict(fks).items())
            kind = "이벤트 로그" if d["date_cols"] else "부속 테이블(날짜 없음)"
            return kind, (
                f"`{key}`는 이 표에만 있는 일련번호이고, {fk_txt} 참조가 있습니다. "
                f"1행 = {'사건 1건' if d['date_cols'] else '참조 대상의 부속 정보 1건'}."
            )

        if refs:
            ref_txt = ", ".join(sorted({t for _, t in refs}))
            return "마스터", (
                f"`{key}`가 유일하고 다른 표({ref_txt})가 이 키를 참조합니다. "
                "1행 = 개체 1개."
            )

        return ttype, d["type_reason"]

    return ttype, d["type_reason"]
