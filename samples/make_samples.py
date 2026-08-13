"""
연습용 샘플 데이터셋 생성기.

우리 수업 데이터(통신사 CS)와 무관한 도메인 3종을 만든다.
각 세트는 진단기의 서로 다른 기능을 드러내도록 설계했다.

  A. 온라인 서점 (3개 파일)  → 조인·팬아웃·분산0 컬럼·기간 공백
  B. 헬스장 회원 (2개 파일)  → 마스터 + 패널, 결측 많은 컬럼
  C. 채용 전형 (1개 파일)    → 심슨의 역설, 각종 검정

실행: python make_samples.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(20260813)
HERE = Path(__file__).resolve().parent


def save(df: pd.DataFrame, name: str):
    path = HERE / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  {name:<34} {len(df):>6,}행 × {len(df.columns)}열")


# ==================================================================
# A. 온라인 서점 — 조인 3단, 팬아웃, 분산0, 기간 공백
# ==================================================================
print("\n[A] 온라인 서점")

CATS = ["소설", "경제경영", "인문", "과학", "자기계발", "여행"]
PUBS = [f"{p}출판사" for p in "가나다라마바사아자차카타파하거"]

n_books = 200
books = pd.DataFrame({
    "book_id": [f"B{i:04d}" for i in range(1, n_books + 1)],
    "title": [f"도서 {i:03d}" for i in range(1, n_books + 1)],
    "category": RNG.choice(CATS, n_books),
    "publisher": RNG.choice(PUBS, n_books),
    "price": (RNG.integers(9, 45, n_books) * 1000).astype(int),
    "page_count": RNG.integers(120, 700, n_books),
    "currency": "KRW",          # 분산 0 — 진단기가 "지표로 쓸 수 없음"으로 잡아야 한다
})
save(books, "bookstore_books.csv")

# 주문: 2023-01 ~ 2023-12, 단 2023-07은 통째로 비운다 (시스템 교체 가정)
n_orders = 1_500
months = [m for m in range(1, 13) if m != 7]
order_month = RNG.choice(months, n_orders)
order_day = RNG.integers(1, 29, n_orders)

orders = pd.DataFrame({
    "order_id": [f"O{i:05d}" for i in range(1, n_orders + 1)],
    "customer_id": [f"C{i:04d}" for i in RNG.integers(1, 401, n_orders)],
    "order_date": [f"2023-{m:02d}-{d:02d}" for m, d in zip(order_month, order_day)],
    "channel": RNG.choice(["웹", "앱", "제휴몰"], n_orders, p=[0.45, 0.45, 0.10]),
    "payment_method": RNG.choice(["카드", "계좌이체", "간편결제"], n_orders),
    "coupon_code": RNG.choice([None, "WELCOME", "SPRING", "VIP"], n_orders,
                              p=[0.72, 0.12, 0.10, 0.06]),   # 결측 72%
})
save(orders, "bookstore_orders.csv")

# 주문 상세: 주문당 1~4권 → orders와 1:N, books와 N:1
rows = []
item_no = 1
for oid in orders["order_id"]:
    for bid in RNG.choice(books["book_id"], RNG.integers(1, 5), replace=False):
        price = int(books.loc[books.book_id == bid, "price"].iloc[0])
        qty = int(RNG.choice([1, 1, 1, 2, 3]))
        rows.append({
            "item_id": f"I{item_no:06d}", "order_id": oid, "book_id": bid,
            "quantity": qty, "unit_price": price, "line_total": price * qty,
        })
        item_no += 1
save(pd.DataFrame(rows), "bookstore_order_items.csv")


# ==================================================================
# B. 헬스장 회원 — 마스터 + 패널, 중도 가입/해지
# ==================================================================
print("\n[B] 헬스장 회원")

n_mem = 300
join_month = RNG.choice(range(1, 11), n_mem, p=np.array(
    [18, 14, 12, 10, 9, 8, 7, 8, 7, 7]) / 100)
cancelled = RNG.random(n_mem) < 0.28
cancel_month = np.where(
    cancelled, np.minimum(join_month + RNG.integers(2, 10, n_mem), 12), 0)

members = pd.DataFrame({
    "member_id": [f"M{i:04d}" for i in range(1, n_mem + 1)],
    "join_date": [f"2024-{m:02d}-{RNG.integers(1, 28):02d}" for m in join_month],
    "cancel_date": [
        f"2024-{c:02d}-{RNG.integers(1, 28):02d}" if c else None
        for c in cancel_month
    ],
    "plan": RNG.choice(["1개월", "6개월", "12개월"], n_mem, p=[0.3, 0.4, 0.3]),
    "age": RNG.integers(19, 65, n_mem),
    "gender": RNG.choice(["남", "여"], n_mem),
    "referred_by": RNG.choice([None] + [f"M{i:04d}" for i in range(1, 60)],
                              n_mem, p=[0.8] + [0.2 / 59] * 59),
})
save(members, "gym_members.csv")

# 월별 출석: 가입~해지 구간만 존재 → 완전 균형 패널이 아니다
rows = []
for mid, jm, cm in zip(members.member_id, join_month, cancel_month):
    end = cm if cm else 12
    for m in range(jm, end + 1):
        base = RNG.poisson(9)
        # 해지 직전 2개월은 출석이 급감한다 (검정 실습용 신호)
        if cm and m >= cm - 1:
            base = max(0, int(base * 0.35))
        rows.append({
            "member_id": mid,
            "year_month": f"2024-{m:02d}",
            "visit_count": base,
            "pt_session_count": int(RNG.poisson(1.2)) if RNG.random() < 0.35 else 0,
            "avg_stay_min": round(float(RNG.normal(62, 14)), 1) if base else 0.0,
        })
save(pd.DataFrame(rows), "gym_monthly_visits.csv")


# ==================================================================
# C. 채용 전형 — 심슨의 역설
# ==================================================================
print("\n[C] 채용 전형")

# 설계: 전체로는 남성 합격률이 훨씬 높지만,
# 부서별로 나누면 모든 부서에서 여성이 더 높다.
# 원인은 여성이 경쟁이 심한 부서에 몰려 지원했기 때문이다.
SPEC = [
    # 부서,     남 지원, 남 합격률, 여 지원, 여 합격률
    ("개발",      300, 0.70,  50, 0.74),
    ("영업",      200, 0.65,  50, 0.68),
    ("디자인",     50, 0.25, 250, 0.28),
    ("마케팅",     50, 0.20, 250, 0.23),
]

rows, aid = [], 1
for dept, n_m, r_m, n_f, r_f in SPEC:
    for gender, n, rate in (("남", n_m, r_m), ("여", n_f, r_f)):
        hired = np.zeros(n, dtype=int)
        hired[: int(round(n * rate))] = 1
        RNG.shuffle(hired)
        exp = np.clip(RNG.normal(4.5, 2.4, n), 0, 20).round(1)
        # 점수는 경력과 약한 양의 상관 + 합격자가 조금 높다
        score = np.clip(58 + exp * 1.6 + hired * 6 + RNG.normal(0, 9, n), 0, 100)
        for i in range(n):
            rows.append({
                "applicant_id": f"A{aid:04d}",
                "department": dept,
                "gender": gender,
                "experience_years": exp[i],
                "test_score": round(float(score[i]), 1),
                "interview_round": int(RNG.integers(1, 4)),
                "hired": int(hired[i]),
            })
            aid += 1

hiring = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
save(hiring, "hiring_applications.csv")

# 설계 의도대로 나왔는지 확인
print("\n  전체 합격률")
tot = hiring.groupby("gender")["hired"].agg(["sum", "count"])
for g, r in tot.iterrows():
    print(f"    {g}: {r['sum'] / r['count'] * 100:5.1f}%  ({r['sum']}/{r['count']})")

print("\n  부서별 합격률")
by = hiring.groupby(["department", "gender"])["hired"].agg(["sum", "count"])
for (d, g), r in by.iterrows():
    print(f"    {d:<5} {g}: {r['sum'] / r['count'] * 100:5.1f}%  ({r['sum']}/{r['count']})")

print("\n완료.")
