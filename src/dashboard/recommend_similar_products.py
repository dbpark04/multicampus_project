import os
import numpy as np
import pandas as pd
import glob
from typing import List, Optional, Dict, Any


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0

    # 벡터가 리스트인 경우 numpy 배열로 변환
    if isinstance(vec1, list):
        vec1 = np.array(vec1)
    if isinstance(vec2, list):
        vec2 = np.array(vec2)

    # 0 벡터인 경우 처리
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def load_products_data(
    processed_data_dir: str = "./data/processed_data",
    categories: Optional[List[str]] = None,
    vector_type: str = "roberta",
) -> pd.DataFrame:
    """
    상품 데이터 로드 (integrated_products_final)

    Args:
        processed_data_dir: processed_data 디렉토리 경로
        categories: 로드할 카테고리 리스트 (None이면 전체)
        vector_type: 사용할 벡터 타입 ("roberta", "bert", "koelectra", "word2vec")

    Returns:
        pd.DataFrame: 상품 데이터
    """
    products_final_dir = os.path.join(processed_data_dir, "integrated_products_final")

    # Hive 파티셔닝: category=*/data.parquet 패턴
    if categories is None:
        # 모든 카테고리 로드
        parquet_files = glob.glob(
            os.path.join(products_final_dir, "category=*", "data.parquet")
        )
    else:
        # 특정 카테고리만 로드
        parquet_files = []
        for category in categories:
            file_path = os.path.join(
                products_final_dir, f"category={category}", "data.parquet"
            )
            if os.path.exists(file_path):
                parquet_files.append(file_path)

    if not parquet_files:
        return pd.DataFrame()

    # 모든 파일 로드 및 병합
    dfs = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        dfs.append(df)

    all_products = pd.concat(dfs, ignore_index=True)

    # 필요한 컬럼 확인
    vector_col = f"product_vector_{vector_type}"
    if vector_col not in all_products.columns:
        raise ValueError(
            f"'{vector_col}' 컬럼이 존재하지 않습니다. "
            f"사용 가능한 벡터 타입을 확인하세요."
        )

    return all_products


def recommend_similar_products(
    product_id: Optional[str] = None,
    categories: Optional[List[str]] = None,
    top_n: int = 10,
    processed_data_dir: str = "./data/processed_data",
    vector_type: str = "roberta",
    exclude_self: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    유사 상품 추천 또는 전체 상품 랭킹 (벡터화 최적화 버전)

    점수 계산 방식:
    - product_id가 있을 때 (유사 상품 추천):
      점수 = 유사도 * 0.5 + 긍정확률 * 0.3 + 정규화_평점 * 0.2
    - product_id가 None일 때 (전체 랭킹):
      점수 = 긍정확률 * 0.6 + 정규화_평점 * 0.4
    - 정규화_평점 = 평균평점 / 5.0

    Args:
        product_id: 기준 상품 ID (예: "로션_1"), None이면 전체 랭킹
        categories: 검색할 카테고리 리스트 (None이면 모든 카테고리)
        top_n: 반환할 추천 상품 개수 (카테고리별)
        processed_data_dir: processed_data 디렉토리 경로
        vector_type: 사용할 벡터 타입
        exclude_self: 자기 자신을 결과에서 제외할지 여부

    Returns:
        Dict[str, List[Dict]]: 카테고리별 추천 상품 딕셔너리
    """
    # 1. 상품 데이터 로드
    print(f"상품 데이터 로드 중... (카테고리: {categories or '전체'})")
    all_products = load_products_data(processed_data_dir, categories, vector_type)

    if all_products.empty:
        print("[경고] 상품 데이터를 찾을 수 없습니다.")
        return {}

    print(f"✓ {len(all_products):,}개 상품 로드 완료")

    vector_col = f"product_vector_{vector_type}"

    # 2. 전처리: 결측치 처리 및 정규화 (메서드 체이닝)
    print(f"\n전처리 중...")
    df = all_products.copy().assign(
        sentiment_score=lambda x: x["sentiment_score"].fillna(0.5),
        avg_rating_with_text=lambda x: x["avg_rating_with_text"].fillna(0),
        normalized_rating=lambda x: x["avg_rating_with_text"] / 5.0,
    )

    # 3. 벡터 유효성 검사 및 필터링
    # 벡터가 None이거나 빈 리스트인 행 제거
    valid_vector_mask = df[vector_col].apply(
        lambda v: v is not None
        and (not isinstance(v, (list, np.ndarray)) or len(v) > 0)
    )
    df = df[valid_vector_mask].reset_index(drop=True)

    if df.empty:
        print("[경고] 유효한 벡터를 가진 상품이 없습니다.")
        return {}

    print(f"✓ 유효한 상품 {len(df):,}개")

    # 4. product_id 유무에 따른 분기 처리
    if product_id is not None:
        # 유사 상품 추천 모드
        print(f"\n[모드] 유사 상품 추천 (product_id={product_id})")

        target_product = df[df["product_id"] == product_id]
        if target_product.empty:
            print(f"[오류] 상품 ID '{product_id}'를 찾을 수 없습니다.")
            return {}

        target_vector = target_product.iloc[0][vector_col]
        target_product_name = target_product.iloc[0].get("product_name", product_id)

        # 리스트를 numpy 배열로 변환
        if isinstance(target_vector, list):
            target_vector = np.array(target_vector)

        print(f"✓ 기준 상품: {target_product_name}")
        print("점수 = 유사도 * 0.5 + 긍정확률 * 0.3 + 정규화_평점 * 0.2")

        # 자기 자신 제외
        if exclude_self:
            df = df[df["product_id"] != product_id].reset_index(drop=True)

        # 5. 벡터화 연산: 모든 상품 벡터를 행렬로 변환
        print(f"\n유사도 계산 중 (벡터화 연산)...")
        vectors_list = df[vector_col].tolist()
        # 리스트를 numpy 배열로 변환
        vectors_list = [np.array(v) if isinstance(v, list) else v for v in vectors_list]
        vectors_matrix = np.stack(vectors_list)  # (N, D) 행렬

        # 6. 코사인 유사도 벡터화 계산
        # similarity = (V @ t) / (||V|| * ||t||)
        target_norm = np.linalg.norm(target_vector)
        vectors_norms = np.linalg.norm(vectors_matrix, axis=1)  # (N,)

        # 내적 계산 (행렬곱)
        dot_products = vectors_matrix @ target_vector  # (N,)

        # 코사인 유사도 (0으로 나누기 방지)
        similarities = np.where(
            (vectors_norms > 0) & (target_norm > 0),
            dot_products / (vectors_norms * target_norm),
            0.0,
        )

        # 7. 추천 점수 계산 (벡터화)
        df = df.assign(
            cosine_similarity=similarities,
            recommend_score=lambda x: (
                x["cosine_similarity"] * 0.5
                + x["sentiment_score"] * 0.3
                + x["normalized_rating"] * 0.2
            ),
        )

    else:
        # 전체 랭킹 모드
        print(f"\n[모드] 전체 상품 랭킹")
        print("점수 = 긍정확률 * 0.6 + 정규화_평점 * 0.4")

        # 8. 랭킹 점수 계산 (유사도 없음)
        df = df.assign(
            recommend_score=lambda x: (
                x["sentiment_score"] * 0.6 + x["normalized_rating"] * 0.4
            )
        )

    # 9. 카테고리별 상위 N개 선택 (메서드 체이닝)
    print(f"\n카테고리별 상위 {top_n}개 선택 중...")
    results_df = (
        df.sort_values("recommend_score", ascending=False)
        .groupby("category", group_keys=False)
        .apply(lambda x: x.head(top_n))
        .reset_index(drop=True)
    )

    # 10. 결과를 딕셔너리 형태로 변환
    final_results = {}

    for category in results_df["category"].unique():
        category_df = results_df[results_df["category"] == category]

        products_list = []
        for _, row in category_df.iterrows():
            result = {
                "product_id": row["product_id"],
                "product_name": row.get("product_name", ""),
                "brand": row.get("brand", ""),
                "category": row["category"],
                "price": row.get("price"),
                "recommend_score": float(row["recommend_score"]),
                "sentiment_score": float(row["sentiment_score"]),
                "normalized_rating": float(row["normalized_rating"]),
                "avg_rating": float(row["avg_rating_with_text"]),
                "total_reviews": row.get("total_reviews", 0),
                "avg_rating_with_text": row.get("avg_rating_with_text", 0),
                "top_keywords": row.get("top_keywords", []),
                "product_url": row.get("product_url", ""),
            }

            # 유사도는 product_id가 있을 때만 포함
            if "cosine_similarity" in row:
                result["cosine_similarity"] = float(row["cosine_similarity"])

            products_list.append(result)

        final_results[category] = products_list

    print(
        f"✓ 추천 상품 {len(results_df)}개 생성 완료 ({len(final_results)}개 카테고리)"
    )
    for category, products in final_results.items():
        print(f"  - {category}: {len(products)}개")

    return final_results


def print_recommendations(recommendations: Dict[str, List[Dict[str, Any]]]):
    """
    추천 결과를 보기 좋게 출력

    Args:
        recommendations: recommend_similar_products() 결과 (카테고리별 딕셔너리)
    """
    if not recommendations:
        print("추천 결과가 없습니다.")
        return

    print("\n" + "=" * 100)
    print("추천 상품 목록 (카테고리별)")
    print("=" * 100)

    for category, products in recommendations.items():
        print(f"\n[{category}] - {len(products)}개 상품")
        print("-" * 110)
        print(
            f"{'순위':<5} {'상품명':<30} {'브랜드':<15} {'점수':<8} {'유사도':<8} {'감성':<8} {'평점':<8}"
        )
        print("-" * 110)

        for rank, rec in enumerate(products, 1):
            # None 값 처리
            product_name = rec["product_name"] or ""
            brand = rec["brand"] or ""

            # 길이 제한 적용
            product_name = (
                product_name[:28] + ".." if len(product_name) > 30 else product_name
            )
            brand = brand[:13] + ".." if len(brand) > 15 else brand

            # 유사도는 있을 때만 표시
            similarity_str = (
                f"{rec.get('cosine_similarity', 0.0):<8.3f}"
                if "cosine_similarity" in rec
                else "N/A     "
            )

            print(
                f"{rank:<5} "
                f"{product_name:<30} "
                f"{brand:<15} "
                f"{rec['recommend_score']:<8.3f} "
                f"{similarity_str} "
                f"{rec['sentiment_score']:<8.3f} "
                f"{rec.get('avg_rating', 0):<8.1f}"
            )

    print("\n" + "=" * 100)


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 특정 카테고리에서 추천
    print("=" * 100)
    print("예시 1: 로션 카테고리에서 유사 상품 추천")
    print("=" * 100)

    results = recommend_similar_products(
        product_id="로션_1",
        categories=["로션"],
        top_n=2,
    )

    print_recommendations(results)
    print("\n\n")
    print("=" * 100)

    # 예시 2: 모든 카테고리에서 추천
    print("\n\n" + "=" * 100)
    print("예시 2: 모든 카테고리에서 유사 상품 추천")
    print("=" * 100)

    results = recommend_similar_products(
        product_id="로션_1",
        categories=None,
        top_n=2,
    )

    print_recommendations(results)

    # 예시 3: 상품 미입력 (필터링된것중 전체 랭킹)
    print("\n\n" + "=" * 100)
    print("예시 3: 모든 카테고리에서 유사 상품 추천")
    print("=" * 100)

    results = recommend_similar_products(
        product_id=None,
        categories=None,
        top_n=2,
    )

    print_recommendations(results)

    # print("results\n", results)

    # print("\n\n" + "=" * 100)

    # for category, items in results.items():
    #     print(f"\nCategory: {category}")
    #     for item in items:
    #         print(
    #             f"  - {item['product_id']}: Score={item['recommend_score']:.3f}, "
    #             f"Sentiment={item['sentiment_score']:.3f}, "
    #             f"Avg Rating={item['avg_rating']:.1f}"
    #         )
