import ast
import re
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from itertools import chain
import seaborn as sns
import random
import matplotlib.gridspec as gridspec
import platform
from pathlib import Path

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

# 운영체제별 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = r"C:\WINDOWS\FONTS\MALGUNSL.TTF"
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
else:  # Linux
    plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

DATA_DIR = os.path.join("data", "processed_data")
PARQUET_PATH = os.path.join(
    "data", "processed_data", "integrated_reviews_detail.parquet"
)


# 재귀적으로 모든 JSON 파일 읽기
json_files = [str(p) for p in Path(DATA_DIR).rglob("*.json")]

print(f"총 JSON 파일 개수: {len(json_files)}")
for jf in json_files[:5]:  # 처음 5개 파일 경로 출력
    print(f"  - {jf}")


# 상품 데이터
product_rows = []
total_rating_rows = []

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    total_rating_dist = raw.get("total_rating_distribution")
    if not total_rating_dist:
        continue

    total_rating_rows.append(
        {
            "search_name": raw.get("search_name"),
            "total_product": raw.get("total_product"),
            "total_collected_reviews": raw.get("total_collected_reviews"),
            "rating_5": total_rating_dist.get("5", 0),
            "rating_4": total_rating_dist.get("4", 0),
            "rating_3": total_rating_dist.get("3", 0),
            "rating_2": total_rating_dist.get("2", 0),
            "rating_1": total_rating_dist.get("1", 0),
        }
    )

    for item in raw.get("data", []):
        if "product_info" in item:
            p_info = item.get("product_info", {})
        else:
            p_info = item

        product_rows.append(
            {
                "product_id": p_info.get("product_id"),
                "product_name": p_info.get("product_name_clean")
                or p_info.get("product_name"),
                "brand": p_info.get("brand"),
                "category": p_info.get("category_normal") or p_info.get("category"),
                "total_reviews": p_info.get("total_reviews", 0),
                # 상품별 평점 분포
                "rating_5": p_info.get("rating_distribution", {}).get("5", 0),
                "rating_4": p_info.get("rating_distribution", {}).get("4", 0),
                "rating_3": p_info.get("rating_distribution", {}).get("3", 0),
                "rating_2": p_info.get("rating_distribution", {}).get("2", 0),
                "rating_1": p_info.get("rating_distribution", {}).get("1", 0),
            }
        )

df_product = pd.DataFrame(product_rows).drop_duplicates(subset="product_id")
df_total_rating = pd.DataFrame(total_rating_rows)

print("\n===== json 데이터 컬럼 =====")
print(df_product.columns)
print("상품 수:", len(df_product))

print("\n===== 전체 평점 분포 데이터프레임 =====")
print(df_total_rating)


# parquet 텍스트 리뷰
df_review = pd.read_parquet(PARQUET_PATH)

df_review["has_image"] = df_review["has_image"].fillna(0).astype(int)
df_review["helpful_count"] = df_review["helpful_count"].fillna(0).astype(int)
df_review["review_date"] = pd.to_datetime(df_review["date"], errors="coerce")
df_review["label"] = df_review["label"].astype("Int64")
df_review["tokens"] = df_review["tokens"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
df_review["source"] = "parquet"

print("\n===== 텍스트 리뷰 데이터프레임 =====")
print(df_review)
print((df_review).info())


# JSON 텍스트 없는 리뷰 추출
json_review_rows = []

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for item in raw.get("data", []):
        if "product_info" not in item:
            continue

        product_id = item["product_info"].get("product_id")

        for r in item.get("reviews", {}).get("data", []):
            json_review_rows.append(
                {
                    "product_id": product_id,
                    "review_id": r.get("id"),
                    "score": r.get("score"),
                    "review_date": pd.to_datetime(r.get("date"), errors="coerce"),
                    "label": pd.NA,
                    "has_image": 0,
                    "helpful_count": 0,
                    "source": "json",
                }
            )

df_review_json = pd.DataFrame(json_review_rows)


# 리뷰 단위 통합 데이터프레임
df_review_parquet = df_review[
    [
        "product_id",
        "review_id",
        "score",
        "review_date",
        "label",
        "has_image",
        "helpful_count",
        "source",
    ]
]

df_review_all = pd.concat([df_review_parquet, df_review_json], ignore_index=True)

print("\n===== 통합 리뷰 데이터 =====")
print(df_review_all.info())
print(df_review_all["source"].value_counts())


# 기본 통계/분석
product_score = (
    df_review_all.groupby("product_id")
    .agg(
        mean_score=("score", "mean"),
        mean_helpful=("helpful_count", "mean"),
        review_count=("score", "count"),
    )
    .reset_index()
)

print(product_score)

# 리뷰 많은 상품 TOP 6
top_6_products = (
    df_review_all.groupby("product_id")
    .size()
    .reset_index(name="total_reviews")
    .merge(df_product[["product_id", "product_name"]], on="product_id", how="left")
    .sort_values("total_reviews", ascending=False)
    .head(6)
)

print("\n===== 리뷰 많은 상품 TOP 6 =====")
print(top_6_products)


# 리뷰 많은 상품 TOP 6
# 상품별 월별 평점 분포 + 월별 리뷰 수 + 평균 평점
top_6_product_ids = top_6_products["product_id"].tolist()

df_top6 = df_review_all[df_review_all["product_id"].isin(top_6_product_ids)].dropna(
    subset=["review_date", "score"]
)

df_top6["year_month"] = df_top6["review_date"].dt.to_period("M").astype(str)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 2, figure=fig)

for i, pid in enumerate(top_6_product_ids):
    df_p = df_top6[df_top6["product_id"] == pid]

    product_name = (
        df_product.loc[df_product["product_id"] == pid, "product_name"].values[0]
        if pid in df_product["product_id"].values
        else pid
    )

    # 월별 리뷰 수
    monthly_count = df_p.groupby("year_month").size().rename("review_count")

    # 월별 평균 평점
    monthly_mean = df_p.groupby("year_month")["score"].mean().rename("mean_score")

    # 월별 평점 분포
    rating_dist = df_p.groupby(["year_month", "score"]).size().unstack(fill_value=0)

    ax1 = fig.add_subplot(gs[i // 2, i % 2])
    rating_dist.plot(kind="bar", stacked=True, ax=ax1)
    ax1.set_title(f"{product_name}\n월별 평점 & 리뷰 수 분포", pad=20)
    ax1.set_xlabel("월")
    ax1.set_ylabel("리뷰 수")

    x_labels = rating_dist.index.tolist()
    new_labels = []
    prev_year = None
    for idx, ym in enumerate(x_labels):
        year, month = ym.split("-")
        if idx % 2 == 0:  # 2달마다 표시
            if year != prev_year:  # 연도가 바뀌면 연도 포함
                new_labels.append(f"{year}-{month}")
                prev_year = year
            else:
                new_labels.append(month)
        else:
            new_labels.append("")
    ax1.set_xticklabels(new_labels, rotation=45, ha="right")

    # # 평균 평점 텍스트 표시
    # for x, (ym, mean) in enumerate(monthly_mean.items()):
    #     y_pos = rating_dist.loc[ym].sum() * 0.95
    #     ax1.text(x, y_pos, f"★ {mean:.2f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    monthly_mean.plot(color='blue', linewidth=1, label="월별 평균 평점", ax=ax2)
    ax2.set_ylabel("평균 평점")
    ax2.set_ylim(1, 5)
    ax2.legend(loc="upper right")

    x_labels2 = monthly_mean.index.tolist()
    new_labels2 = []
    prev_year2 = None
    for idx, ym in enumerate(x_labels2):
        year, month = ym.split("-")
        if idx % 2 == 0:  # 2달마다 표시
            if year != prev_year2:  # 연도가 바뀌면 연도 포함
                new_labels2.append(f"{year}-{month}")
                prev_year2 = year
            else:
                new_labels2.append(month)
        else:
            new_labels2.append("")
    ax2.set_xticks(range(len(new_labels2)))  # tick 위치 먼저 설정
    ax2.set_xticklabels(new_labels2, rotation=45, ha="right")

plt.tight_layout()
plt.show()


# ===== 시각화 1 =====
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

rating_sum = df_total_rating.loc[0, ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5"]]
rating_sum.index = ["1점", "2점", "3점", "4점", "5점"]
rating_sum.plot(kind="barh", ax=axes[0, 0], color=sns.color_palette("YlOrRd", 5))
axes[0, 0].set_title("전체 상품 평점 분포", weight="bold")
axes[0, 0].set_ylabel("리뷰 수")
axes[0, 0].grid(axis="y", alpha=0.3)

axes[0, 1].hist(df_review["char_length"], bins=50, color="pink")
axes[0, 1].set_title("리뷰 길이 분포", weight="bold")
axes[0, 1].set_xlim(0, 2000)

axes[0, 2].scatter(df_review["char_length"], df_review["helpful_count"], alpha=0.3, s=10, color="green")
axes[0, 2].set_xscale("log")
axes[0, 2].set_yscale("log")
axes[0, 2].set_title("리뷰 길이 & Helpful_count", weight="bold")
axes[0, 2].set_xlabel("리뷰 길이")
axes[0, 2].set_ylabel("Helpful_count")

sns.violinplot(x="score", y="char_length", data=df_review, palette="Set2", ax=axes[1, 0])
axes[1, 0].set_title("평점별 리뷰 길이", weight="bold")
axes[1, 0].set_xlabel("평점")
axes[1, 0].set_ylabel("리뷰 길이")
axes[1, 0].set_ylim(0, 1500)

sns.boxplot(x="score", y="helpful_count", data=df_review, palette="Pastel1", ax=axes[1, 1])
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("평점별 helpful_count", weight="bold")
axes[1, 1].set_xlabel("평점")
axes[1, 1].set_ylabel("Helpful_count")

axes[1, 2].scatter(
    product_score["mean_score"],
    product_score["mean_helpful"],
    s=60,
    alpha=0.7,
    color="skyblue",
    edgecolor="blue",
)
axes[1, 2].set_title("상품 평균 평점 & Helpful_count", weight="bold")
axes[1, 2].set_xlabel("상품 평균 평점")
axes[1, 2].set_ylabel("Helpful_count")

plt.tight_layout()
plt.show()


df_product["product_name_short"] = (df_product["product_name"]).str.slice(0, 30)  # 상품명 30자 까지만

# ===== 시각화 2 =====
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 2.3])

rating_cols = ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5"]

# 상품별 평균 평점
ax1 = fig.add_subplot(gs[0, 0])
df_product["avg_rating"] = (
    df_product["rating_1"] * 1
    + df_product["rating_2"] * 2
    + df_product["rating_3"] * 3
    + df_product["rating_4"] * 4
    + df_product["rating_5"] * 5
) / (df_product[rating_cols].sum(axis=1)).replace(0, pd.NA)

top10 = (
    df_product.dropna(subset=["avg_rating"])
    .sort_values("avg_rating", ascending=False)
    .head(10)
)

top10.set_index("product_name_short")["avg_rating"].plot(
    kind="barh", color="slateblue", ax=ax1
)

# print(top10[rating_cols].head(10))

ax1.set_title("TOP 10 상품 평균 평점", weight="bold")
ax1.set_xlim(4.5, 5)
ax1.invert_yaxis()


# 평점 & 상품 히트맵
# 상품명 merge
df_review_all = df_review_all.merge(
    df_product[["product_id", "product_name_short"]],
    on="product_id",
    how="left"
)

rating_dist = df_review_all.groupby(["product_name_short", "score"]).size().unstack(fill_value=0)   # 상품별 평점 분포
rating_dist = rating_dist.loc[rating_dist.sum(axis=1).sort_values(ascending=False).head(10).index]  # 상위 10개 상품 필터링
rating_dist.columns = rating_dist.columns.astype(str)   # 컬럼이 숫자형이면 문자열로 변환해서 정렬

ax2 = fig.add_subplot(gs[0, 1])

sns.heatmap(
    rating_dist,
    cmap="YlOrRd",
    annot=True,
    fmt=".0f",
    annot_kws={"size": 8},
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    ax=ax2,
)

ax2.set_title("상품별 평점 분포 히트맵", weight="bold")
ax2.set_aspect("auto")
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=10)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=9)


# 월별 평균 판매량
ax3 = fig.add_subplot(gs[1, :])

monthly_review_cnt = (
    df_review_all.dropna(subset=["review_date"])
    .set_index("review_date")
    .resample("ME")["review_id"]
    .count()
)
monthly_review_cnt.plot(ax=ax3, linewidth=2, color="salmon")
ax3.set_title("월별 평균 판매량 추이 (리뷰 수 기반)", weight="bold")
ax3.set_xlabel("월")
ax3.set_ylabel("리뷰 수")

plt.tight_layout()
plt.show()


# ===== 워드클라우드 =====
def normalize_tokens(x):
    # list
    if isinstance(x, list):
        return x

    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # 문자열
    if isinstance(x, str):
        # 단어 추출
        tokens = re.findall(r"'([^']+)'", x)
        return tokens

    return []


df_review["tokens"] = df_review["tokens"].apply(normalize_tokens)

df_wc = df_review[
    (df_review["label"].isin([0, 1])) & (df_review["tokens"].apply(len) > 0)
]

# print(df_wc["label"].value_counts())

pos_tokens = list(chain.from_iterable(df_wc[df_wc["label"] == 1]["tokens"]))
neg_tokens = list(chain.from_iterable(df_wc[df_wc["label"] == 0]["tokens"]))

print("긍정 리뷰 수:", (df_wc["label"] == 1).sum())
print("부정 리뷰 수:", (df_wc["label"] == 0).sum())
print("긍정 토큰 수:", len(pos_tokens))
print("부정 토큰 수:", len(neg_tokens))


def wc_color(palette):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        r, g, b = random.choice(palette)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    return color_func


def wc(tokens, palette):
    if not tokens:
        return None

    text = " ".join(tokens)
    wc = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        width=800,
        height=600,
        max_words=100,
        color_func=wc_color(palette),
    ).generate(text)

    return wc


wc_pos = wc(pos_tokens, sns.color_palette("OrRd", 10))
wc_neg = wc(neg_tokens, sns.color_palette("cool", 10))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if wc_pos is not None:
    axes[0].imshow(wc_pos)
axes[0].set_title("긍정 리뷰 워드클라우드")
axes[0].axis("off")

if wc_neg is not None:
    axes[1].imshow(wc_neg)
axes[1].set_title("부정 리뷰 워드클라우드")
axes[1].axis("off")

plt.tight_layout()
plt.show()
