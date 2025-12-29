from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Generator, List, Tuple, Optional

import numpy as np
import pandas as pd


# ==============================
# 기본 통계 산출 설정, 자료구조 정의
# ==============================


@dataclass(frozen=True)
class BasicStatsConfig:
    """
    기본 통계량 산출 설정
    - file_suffix: 통계를 낼 파일 패턴(기본은 텍스트 없는 버전 추천)
    - review_cnt_bins: 상품별 리뷰 수 분포를 만들 때 bin 경계
    - valid_scores: 유효 별점 범위
    - save_outputs: 결과 DF 저장 여부
    - output_dirname: 저장 폴더명(processed_root 하위)
    - save_format: 저장 포맷("parquet" or "csv")
    """

    file_suffix: str = "_without_text.json"
    review_cnt_bins: Tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
    valid_scores: Tuple[int, ...] = (1, 2, 3, 4, 5)

    save_outputs: bool = True
    output_dirname: str = "eda_outputs"
    save_format: str = "parquet"  # "parquet" or "csv"


def init_review_stat_counters() -> Dict[str, Any]:
    """
    기본 통계 계산을 위한 누적 카운터/셋 자료구조 초기화
    """
    return {
        "category_products": defaultdict(set),  # category -> set(product_id)
        "product_review_cnt": defaultdict(int),  # product_id -> collected review count
        "score_cnt": Counter(),  # score -> count
        "category_score_cnt": defaultdict(int),  # (category, score) -> count
    }


# =========================
# int 변환, 카테고리 추출 규칙
# =========================


def to_int_safe(x) -> Optional[int]:
    """문자열/None 등 안전하게 int 변환. 실패 시 None."""
    try:
        return int(x)
    except Exception:
        return None


def extract_category(product_info: Dict[str, Any]) -> str:
    """
    상품 정보에서 카테고리 추출 규칙
    우선순위:
      1) category_norm
      2) category_path 마지막 토큰(> 기준)
      3) UNKNOWN
    """
    cat = product_info.get("category_norm")
    if cat:
        return str(cat)

    path = product_info.get("category_path")
    if path and isinstance(path, str):
        return path.split(">")[-1].strip()

    return "UNKNOWN"


# ====================================
# 수집 완료된 리뷰 JSON 데이터 로드
# ====================================


def find_review_json_files(processed_root: str | Path, suffix: str) -> List[Path]:
    """
    processed_root 아래에서 suffix로 끝나는 json 파일 전체 수집.
    """
    root = Path(processed_root)
    return sorted(root.rglob(f"*{suffix}"))


def load_review_json(path: str | Path) -> Dict[str, Any]:
    """
    파일 최상단이 dict(요약 + data 리스트)이므로 json.load 사용.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# 중첩 JSON에서 상품 단위로 (상품정보, 리뷰리스트)를 순차적으로 반환
# ==========================================================


def iter_products_with_reviews(
    review_obj: Dict[str, Any],
) -> Generator[Tuple[Dict[str, Any], List[Dict[str, Any]]], None, None]:
    """
    '상품 단위'로 순회:
      yield (product_info, reviews_list)

    review_obj 형태:
      {
        ...,
        "data": [
          {
            "product_info": {...},
            "reviews": {"data":[{...}, ...]}
          },
          ...
        ]
      }
    """
    data_list = review_obj.get("data", [])
    if not isinstance(data_list, list):
        return

    for item in data_list:
        product_info = item.get("product_info", {}) or {}
        reviews_blk = item.get("reviews", {}) or {}
        reviews_list = reviews_blk.get("data") or []

        if not isinstance(reviews_list, list):
            reviews_list = []

        yield product_info, reviews_list


# =========================
# 상품과 리뷰의 기본 통계 계산
# =========================


def update_basic_stat_counters(
    counters: Dict[str, Any],
    product_info: Dict[str, Any],
    reviews_list: List[Dict[str, Any]],
    cfg: BasicStatsConfig,
    meta: Counter,
) -> None:
    """
    상품 1개 단위로 기본 통계 누적 업데이트
    """
    pid = product_info.get("product_id")
    if not pid:
        meta["missing_product_id"] += 1
        return

    # 문자열 ID를 그대로 사용
    pid = str(pid)
    category = extract_category(product_info)

    # 1) 카테고리별 distinct 상품 수
    counters["category_products"][category].add(pid)

    # 2) 상품별 수집 리뷰 수
    rcnt = len(reviews_list)
    counters["product_review_cnt"][pid] += rcnt
    meta["total_products_seen"] += 1
    meta["total_reviews_collected"] += rcnt

    # 3) 별점 분포(전체 + 카테고리별)
    for r in reviews_list:
        s = to_int_safe(r.get("score"))
        if s is None or s not in cfg.valid_scores:
            meta["invalid_score"] += 1
            continue

        counters["score_cnt"][s] += 1
        counters["category_score_cnt"][(category, s)] += 1


# =======================================
# 누적된 통계 결과 표로 정리, 파생 테이블 생성
# =======================================


def build_category_product_table(category_products: Dict[str, set]) -> pd.DataFrame:
    """카테고리별 distinct 상품 수 테이블."""
    return (
        pd.DataFrame(
            [
                {"category": k, "product_cnt": len(v)}
                for k, v in category_products.items()
            ]
        )
        .sort_values("product_cnt", ascending=False)
        .reset_index(drop=True)
    )


def build_product_review_table(product_review_cnt: Dict[str, int]) -> pd.DataFrame:
    """상품별 수집 리뷰 수 테이블."""
    if not product_review_cnt:
        return pd.DataFrame(columns=["product_id", "review_cnt"])

    return (
        pd.DataFrame(
            [{"product_id": k, "review_cnt": v} for k, v in product_review_cnt.items()]
        )
        .sort_values("review_cnt", ascending=False)
        .reset_index(drop=True)
    )


def build_review_count_summary(product_review_table: pd.DataFrame) -> pd.DataFrame:
    """상품별 리뷰 수에 대한 기술통계(describe) 테이블."""
    if len(product_review_table) == 0:
        return pd.DataFrame()

    return (
        product_review_table["review_cnt"]
        .describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        .to_frame()
        .T
    )


def build_review_count_bins(
    product_review_table: pd.DataFrame, bins: Tuple[int, ...]
) -> pd.DataFrame:
    """
    상품별 리뷰 수를 bins로 구간화하여 각 구간에 속한 상품 수를 반환.
    (시각화용 histogram/bar용)
    """
    if len(product_review_table) == 0:
        return pd.DataFrame(columns=["bin", "product_cnt"])

    values = product_review_table["review_cnt"].to_numpy()
    edges = np.array(bins, dtype=float)
    cut_bins = np.concatenate([edges, [np.inf]])

    labels = [f"{int(edges[i])}~{int(edges[i+1]-1)}" for i in range(len(edges) - 1)] + [
        f"{int(edges[-1])}+"
    ]

    binned = pd.cut(values, bins=cut_bins, right=False, labels=labels)
    return (
        pd.Series(binned)
        .value_counts(sort=False)
        .rename_axis("bin")
        .reset_index(name="product_cnt")
    )


def build_score_distribution_table(score_cnt: Counter) -> pd.DataFrame:
    """전체 별점 분포 테이블(score, cnt)."""
    return (
        pd.DataFrame([{"score": k, "cnt": v} for k, v in score_cnt.items()])
        .sort_values("score")
        .reset_index(drop=True)
    )


def build_category_score_table(
    category_score_cnt: Dict[Tuple[str, int], int],
) -> pd.DataFrame:
    """카테고리-별점 분포 테이블(category, score, cnt)."""
    if not category_score_cnt:
        return pd.DataFrame(columns=["category", "score", "cnt"])

    return (
        pd.DataFrame(
            [
                {"category": c, "score": s, "cnt": v}
                for (c, s), v in category_score_cnt.items()
            ]
        )
        .sort_values(["category", "score"])
        .reset_index(drop=True)
    )


def build_category_score_summary(category_score_table: pd.DataFrame) -> pd.DataFrame:
    """카테고리별 평균 별점(mean_score) 및 카운트(total_cnt) 테이블."""
    if len(category_score_table) == 0:
        return pd.DataFrame(
            columns=["category", "total_cnt", "score_sum", "mean_score"]
        )

    tmp = category_score_table.copy()
    tmp["score_x_cnt"] = tmp["score"] * tmp["cnt"]

    out = tmp.groupby("category", as_index=False).agg(
        total_cnt=("cnt", "sum"), score_sum=("score_x_cnt", "sum")
    )
    out["mean_score"] = out["score_sum"] / out["total_cnt"]

    return out.sort_values("total_cnt", ascending=False).reset_index(drop=True)


def build_basic_stat_tables(
    counters: Dict[str, Any], cfg: BasicStatsConfig
) -> Dict[str, pd.DataFrame]:
    """
    누적 카운터/셋 자료구조를 최종 DataFrame들로 변환.
    """
    category_product_counts = build_category_product_table(
        counters["category_products"]
    )
    product_review_counts = build_product_review_table(counters["product_review_cnt"])
    product_review_count_summary = build_review_count_summary(product_review_counts)
    product_review_count_bins = build_review_count_bins(
        product_review_counts, cfg.review_cnt_bins
    )

    score_distribution = build_score_distribution_table(counters["score_cnt"])
    category_score_distribution = build_category_score_table(
        counters["category_score_cnt"]
    )
    category_score_summary = build_category_score_summary(category_score_distribution)

    return {
        "category_product_counts": category_product_counts,
        "product_review_counts": product_review_counts,
        "product_review_count_summary": product_review_count_summary,
        "product_review_count_bins": product_review_count_bins,
        "score_distribution": score_distribution,
        "category_score_distribution": category_score_distribution,
        "category_score_summary": category_score_summary,
    }


# ===========================================
# 최종 결과 테이블을 파일로 저장하고 저장 경로 반환
# ===========================================


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, out_base: Path, fmt: str) -> str:
    """DataFrame 저장. out_base는 확장자 없는 경로."""
    if fmt == "parquet":
        out_path = out_base.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return str(out_path)

    if fmt == "csv":
        out_path = out_base.with_suffix(".csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return str(out_path)

    raise ValueError(f"Unsupported save_format={fmt}. Use 'parquet' or 'csv'.")


def save_basic_stat_tables(
    tables: Dict[str, pd.DataFrame], output_dir: str | Path, fmt: str
) -> Dict[str, str]:
    """기본 통계 결과 테이블들을 파일로 저장하고 경로 dict 반환."""
    out_dir = ensure_dir(output_dir)
    saved_paths: Dict[str, str] = {}

    for name, df in tables.items():
        saved_paths[name] = save_dataframe(df, out_dir / f"basic_stats_{name}", fmt)

    return saved_paths


# ==========================
# 기본 통계 분석 전체 실행 함수
# ==========================


def run_basic_review_stats(
    processed_root: str | Path = "data/processed_data",
    cfg: BasicStatsConfig = BasicStatsConfig(),
) -> Dict[str, Any]:
    """
    main에서 호출하는 "기본 통계량 산출" 엔트리 함수.
    - processed_root 아래 suffix 파일들을 수집
    - JSON 로드 -> 상품 단위 순회 -> 카운터 누적
    - 최종 DF 변환
    - 옵션이면 저장
    """
    meta = Counter()
    counters = init_review_stat_counters()

    # meta 기본 키를 미리 만들어두면 로그가 안정적임(0이어도 항상 출력)
    meta["file_read_error"] += 0
    meta["missing_product_id"] += 0
    meta["invalid_score"] += 0

    files = find_review_json_files(processed_root, cfg.file_suffix)
    meta["n_files"] = len(files)

    for fp in files:
        try:
            obj = load_review_json(fp)
        except Exception:
            meta["file_read_error"] += 1
            continue

        for product_info, reviews_list in iter_products_with_reviews(obj):
            update_basic_stat_counters(counters, product_info, reviews_list, cfg, meta)

    tables = build_basic_stat_tables(counters, cfg)
    result: Dict[str, Any] = {"meta": dict(meta), **tables}

    if cfg.save_outputs:
        output_dir = Path(processed_root) / cfg.output_dirname
        result["saved_paths"] = save_basic_stat_tables(
            tables, output_dir, cfg.save_format
        )

    return result


run_basic_review_stats()
