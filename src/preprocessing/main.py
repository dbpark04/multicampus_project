"""
전처리 파이프라인 메인 orchestrator
"""

import json
import os
import glob
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from preprocessing_phases import (
    preprocess_and_tokenize_file,
    train_global_word2vec,
    vectorize_file,
    MAX_WORKERS,
)
from sentiment_analysis import analyze_skin_type_frequency

# 임시 토큰 저장 디렉토리
TEMP_TOKENS_DIR = "./data/temp_tokens"

# ========== 벡터화 방법 설정 ==========
# "word2vec": Word2Vec 사용 (기본, 빠름)
# "bert": BERT 사용 (느리지만 성능 좋음)
# "both": 둘 다 생성 (word2vec, bert 컬럼 모두 포함)
VECTORIZER_TYPE = "word2vec"  # 여기를 변경하여 선택
BERT_MODEL_NAME = "klue/bert-base"  # BERT 모델 이름

# ========== 리뷰 필터링 설정 ==========
MIN_REVIEWS_PER_PRODUCT = 30  # 이 개수 이하의 리뷰를 가진 상품 제외


def main():
    """
    최적화된 전처리 파이프라인:
    Phase 1: 병렬 전처리 + 토큰화 (1회만)
    Phase 2: Iterator 방식 Word2Vec 학습 (메모리 효율적)
    Phase 3: 병렬 벡터화 + 대표 리뷰 선정
    """
    # 시작 시간 기록
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    PRE_DATA_DIR = "./data/pre_data"
    PROCESSED_DATA_DIR = "./data/processed_data"
    PRODUCT_PARQUET = "./data/processed_data/integrated_products_vector.parquet"
    REVIEW_PARQUET = "./data/processed_data/integrated_reviews_detail.parquet"

    print("\n" + "=" * 60)
    print(f"{'최적화된 전처리 파이프라인 시작':^60}")
    print(f"{'시작 시간: ' + start_datetime:^60}")
    print(f"{'벡터화 방법: ' + VECTORIZER_TYPE:^60}")
    print("=" * 60 + "\n")

    # pre_data 디렉토리의 모든 JSON 파일 찾기
    json_files = glob.glob(os.path.join(PRE_DATA_DIR, "**", "*.json"), recursive=True)

    if not json_files:
        print(f"\n[오류] {PRE_DATA_DIR} 디렉토리에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개 파일 발견")
    print(f"병렬 처리 워커 수: {MAX_WORKERS}개\n")

    # ========== Phase 1: 병렬 전처리 + 토큰화 ==========
    print("=" * 60)
    print("Phase 1: 전처리 및 토큰화 (병렬 처리)")
    print("=" * 60)

    phase1_start = time.time()

    # 임시 디렉토리 생성
    os.makedirs(TEMP_TOKENS_DIR, exist_ok=True)

    # 병렬로 전처리 + 토큰화 실행
    args_list = [
        (
            input_path,
            PRE_DATA_DIR,
            PROCESSED_DATA_DIR,
            TEMP_TOKENS_DIR,
            MIN_REVIEWS_PER_PRODUCT,
        )
        for input_path in json_files
    ]

    skipped_count = 0
    phase1_results = []

    with Pool(MAX_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(preprocess_and_tokenize_file, args_list),
            total=len(json_files),
            desc="전처리 및 토큰화",
            unit="파일",
        ):
            if result["status"] == "skipped":
                skipped_count += 1
                tqdm.write(f"  [건너뜀] {result['file']}")
            elif result["status"] == "success":
                phase1_results.append(result)
                tqdm.write(
                    f"  [완료] {result['file']} - 토큰: {result['token_count']:,}개"
                )
            else:
                tqdm.write(
                    f"  [에러] {result['file']} - {result.get('error', 'Unknown')}"
                )

    phase1_time = time.time() - phase1_start
    print(f"\nPhase 1 완료 - 소요 시간: {phase1_time:.2f}초")
    print(f"  처리 완료: {len(phase1_results)}개")
    print(f"  건너뜀: {skipped_count}개\n")

    # ========== Phase 2: 벡터화 모델 준비 ==========
    phase2_start = time.time()
    w2v_model = None
    bert_vectorizer = None

    if VECTORIZER_TYPE in ["word2vec", "both"]:
        print("\n" + "=" * 60)
        print("Phase 2-1: Word2Vec 모델 학습")
        print("=" * 60)
        w2v_model = train_global_word2vec(TEMP_TOKENS_DIR)
        if not w2v_model:
            print("[오류] Word2Vec 모델 학습 실패")
            if VECTORIZER_TYPE == "word2vec":
                return

    if VECTORIZER_TYPE in ["bert", "both"]:
        print("\n" + "=" * 60)
        print("Phase 2-2: BERT 모델 로딩")
        print("=" * 60)
        from bert_vectorizer import get_bert_vectorizer

        bert_vectorizer = get_bert_vectorizer(BERT_MODEL_NAME)

    phase2_time = time.time() - phase2_start
    print(f"\nPhase 2 완료 - 소요 시간: {phase2_time:.2f}초\n")

    # ========== Phase 3: 병렬 벡터화 + 대표 리뷰 선정 ==========
    print("=" * 60)
    print("Phase 3: 벡터화 및 대표 리뷰 선정 (병렬 처리)")
    print("=" * 60)

    phase3_start = time.time()

    # Phase 1에서 처리된 파일들에 대해 벡터화 실행
    vectorize_args = [
        (
            result["base_name"],
            TEMP_TOKENS_DIR,
            result["output_dir"],
            w2v_model,
            bert_vectorizer,
            VECTORIZER_TYPE,
        )
        for result in phase1_results
    ]

    all_products = []
    all_reviews = []

    with Pool(MAX_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(vectorize_file, vectorize_args),
            total=len(vectorize_args),
            desc="벡터화 및 대표 리뷰 선정",
            unit="파일",
        ):
            if result["status"] == "success":
                all_products.extend(result["product_summaries"])
                all_reviews.extend(result["review_details"])
                tqdm.write(f"  [완료] {result['file']}")
            else:
                tqdm.write(
                    f"  [에러] {result['file']} - {result.get('error', 'Unknown')}"
                )

    # 건너뛴 파일의 상품 정보도 로드
    if skipped_count > 0:
        print(f"\n건너뛴 파일 {skipped_count}개의 데이터 로드 중...")
        for input_path in json_files:
            rel_path = os.path.relpath(input_path, PRE_DATA_DIR)
            rel_dir = os.path.dirname(rel_path)
            output_dir = os.path.join(PROCESSED_DATA_DIR, rel_dir)

            file_name = os.path.basename(input_path)
            base_name = os.path.splitext(file_name)[0]
            if base_name.startswith("result_"):
                base_name = base_name[7:]

            processed_file = os.path.join(
                output_dir, f"processed_{base_name}_with_text.json"
            )

            if os.path.exists(processed_file):
                try:
                    with open(processed_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    all_products.extend(existing_data.get("data", []))
                except:
                    pass

    phase3_time = time.time() - phase3_start
    print(f"\nPhase 3 완료 - 소요 시간: {phase3_time:.2f}초\n")

    # ========== 새로운 Parquet 구조 생성 ==========
    print("=" * 60)
    print("새로운 Parquet 구조 생성 중...")
    print("=" * 60)

    # 출력 디렉토리 (Hive 파티셔닝 형식)
    DATA_DIR = "./data/processed_data"
    CATEGORY_SUMMARY_DIR = os.path.join(DATA_DIR, "category_summary")
    PRODUCTS_FINAL_DIR = os.path.join(DATA_DIR, "integrated_products_final")
    DETAILED_STATS_DIR = os.path.join(DATA_DIR, "detailed_stats")
    PARTITIONED_REVIEWS_DIR = os.path.join(DATA_DIR, "partitioned_reviews")

    os.makedirs(CATEGORY_SUMMARY_DIR, exist_ok=True)
    os.makedirs(PRODUCTS_FINAL_DIR, exist_ok=True)
    os.makedirs(DETAILED_STATS_DIR, exist_ok=True)
    os.makedirs(PARTITIONED_REVIEWS_DIR, exist_ok=True)

    # 1. integrated_products_final.parquet 생성
    print("\n[1/3] integrated_products_final.parquet 생성 중...")

    products_final = []
    for product in all_products:
        product_id = product.get("product_id")

        # 텍스트 유무별 리뷰 통계 계산
        product_reviews = [r for r in all_reviews if r.get("product_id") == product_id]

        text_reviews = [r for r in product_reviews if r.get("full_text")]
        no_text_reviews = [r for r in product_reviews if not r.get("full_text")]

        avg_rating_with_text = (
            sum(r.get("score", 0) for r in text_reviews) / len(text_reviews)
            if text_reviews
            else 0.0
        )
        avg_rating_without_text = (
            sum(r.get("score", 0) for r in no_text_reviews) / len(no_text_reviews)
            if no_text_reviews
            else 0.0
        )
        total_reviews = len(product_reviews)
        text_review_ratio = (
            len(text_reviews) / total_reviews if total_reviews > 0 else 0.0
        )

        # rating_distribution 계산 (1~5점 각각 몇 개인지)
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in product_reviews:
            score = r.get("score")
            if score in rating_dist:
                rating_dist[score] += 1

        # top_keywords 추출 (긍정 키워드 상위 5개)
        sentiment = product.get("sentiment_analysis", {})
        pos_keywords = sentiment.get("positive_special", [])[:5]
        top_keywords = [kw.get("word", "") for kw in pos_keywords]

        # product_id에서 카테고리 추출
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

        # category_path에서 마지막 경로 추출
        category_path_str = product.get("category_path", "")
        path = category_path_str.split(" > ")[-1] if category_path_str else ""

        products_final.append(
            {
                # 원본 product_info 필드
                "product_id": product_id,
                "product_name": product.get("product_name"),
                "brand": product.get("brand"),
                "category": category,
                "category_path": category_path_str,
                "path": path,
                "price": product.get("price"),
                "delivery_type": product.get("delivery_type"),
                "product_url": product.get("product_url"),
                # 전처리 추가 필드
                "skin_type": product.get("skin_type", "미분류"),
                "top_keywords": top_keywords,
                "sentiment_analysis": product.get("sentiment_analysis"),
                "product_vector": product.get("product_vector")
                or product.get("product_vector_word2vec"),
                "representative_review_id": product.get("representative_review_id")
                or product.get("representative_review_id_word2vec"),
                "representative_similarity": product.get("representative_similarity")
                or product.get("representative_similarity_word2vec"),
                "recommend_score": product.get("recommend_score", 0.0),
                # 통계 필드
                "avg_rating_with_text": avg_rating_with_text,
                "avg_rating_without_text": avg_rating_without_text,
                "text_review_ratio": text_review_ratio,
                "total_reviews": total_reviews,
                "rating_1": rating_dist[1],
                "rating_2": rating_dist[2],
                "rating_3": rating_dist[3],
                "rating_4": rating_dist[4],
                "rating_5": rating_dist[5],
            }
        )

    df_products_final = pd.DataFrame(products_final)

    # product_id로 정렬 (카테고리_숫자 형식에서 숫자 기준으로 정렬)
    def extract_numeric_sort_key(product_id):
        """product_id에서 카테고리와 숫자를 분리하여 정렬 키 생성"""
        parts = product_id.rsplit("_", 1)  # 마지막 언더스코어 기준으로 분리
        if len(parts) == 2:
            category, num_str = parts
            try:
                return (category, int(num_str))  # 카테고리명, 숫자(정수)로 반환
            except ValueError:
                return (category, 0)  # 숫자 변환 실패 시 0으로 처리
        return (product_id, 0)  # 언더스코어가 없으면 원본 그대로

    df_products_final["_sort_key"] = df_products_final["product_id"].apply(
        extract_numeric_sort_key
    )
    df_products_final = df_products_final.sort_values("_sort_key").reset_index(
        drop=True
    )
    df_products_final = df_products_final.drop(columns=["_sort_key"])  # 임시 컬럼 제거

    # integrated_products_final 저장 (Hive 파티셔닝 - 카테고리별)
    print("\nintegrated_products_final 카테고리별 저장 중...")
    products_count = 0
    for category in (
        df_products_final["product_id"]
        .apply(lambda x: "_".join(x.split("_")[:-1]) if "_" in x else "기타")
        .unique()
    ):
        df_category = df_products_final[
            df_products_final["product_id"].str.startswith(category + "_")
        ]

        partition_dir = os.path.join(PRODUCTS_FINAL_DIR, f"category={category}")
        os.makedirs(partition_dir, exist_ok=True)

        category_path = os.path.join(partition_dir, "data.parquet")
        df_category.to_parquet(
            category_path, engine="pyarrow", compression="snappy", index=False
        )

        file_size_mb = os.path.getsize(category_path) / 1024 / 1024
        products_count += 1
        print(f"  ✓ {category}: {len(df_category):,}개 상품 ({file_size_mb:.2f} MB)")

    print(f"✓ 총 {products_count}개 카테고리 파티션 생성 완료")

    # 1-1. category_summary.parquet 생성 (카테고리별 전체 통계)
    print("\n[1-1] category_summary.parquet 생성 중...")

    category_stats = {}

    # all_reviews를 순회하여 카테고리별 통계 집계
    for review in all_reviews:
        product_id = review.get("product_id")
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

        if category not in category_stats:
            category_stats[category] = {
                "category": category,
                "total_reviews": 0,
                "text_reviews": 0,
                "no_text_reviews": 0,
                "rating_1": 0,
                "rating_2": 0,
                "rating_3": 0,
                "rating_4": 0,
                "rating_5": 0,
            }

        stats = category_stats[category]
        stats["total_reviews"] += 1

        full_text = review.get("full_text", "")
        if full_text and full_text.strip():
            stats["text_reviews"] += 1
        else:
            stats["no_text_reviews"] += 1

        score = review.get("score")
        if score in [1, 2, 3, 4, 5]:
            stats[f"rating_{score}"] += 1

    df_category_summary = pd.DataFrame(list(category_stats.values()))
    # 1. category_summary 저장 (Hive 파티셔닝)
    category_summary_path = os.path.join(CATEGORY_SUMMARY_DIR, "data.parquet")
    df_category_summary.to_parquet(
        category_summary_path, engine="pyarrow", compression="snappy", index=False
    )

    print(f"✓ 저장 완료: {category_summary_path}")
    print(f"  - 카테고리 수: {len(df_category_summary):,}개")
    for _, row in df_category_summary.iterrows():
        print(
            f"  • {row['category']}: 전체 {row['total_reviews']:,}개 (텍스트 {row['text_reviews']:,}개, 없음 {row['no_text_reviews']:,}개)"
        )

    # 2. detailed_stats/ 카테고리별 파티션 생성
    print("\n[2/3] detailed_stats 카테고리별 파티션 생성 중...")

    stats_by_category = {}
    for product in all_products:
        product_id = product.get("product_id")

        # product_id에서 카테고리 추출 (예: "선크림_123" -> "선크림")
        # product_id는 "카테고리_원본ID" 형식 (마지막 언더스코어 기준으로 분리)
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

        sentiment = product.get("sentiment_analysis", {})

        # 긍정 키워드
        for kw in sentiment.get("positive_special", []):
            stats_by_category.setdefault(category, []).append(
                {
                    "product_id": product_id,
                    "word": kw.get("word"),
                    "diff": kw.get("diff"),
                    "pos": kw.get("pos"),
                    "neg": kw.get("neg"),
                    "pos_n": kw.get("pos_n"),
                    "neg_n": kw.get("neg_n"),
                    "support": kw.get("support"),
                    "balanced_ratio": kw.get("balanced_ratio"),
                    "score": kw.get("score"),
                    "sentiment_type": "positive",
                }
            )

        # 부정 키워드
        for kw in sentiment.get("negative_special", []):
            stats_by_category.setdefault(category, []).append(
                {
                    "product_id": product_id,
                    "word": kw.get("word"),
                    "diff": kw.get("diff"),
                    "pos": kw.get("pos"),
                    "neg": kw.get("neg"),
                    "pos_n": kw.get("pos_n"),
                    "neg_n": kw.get("neg_n"),
                    "support": kw.get("support"),
                    "balanced_ratio": kw.get("balanced_ratio"),
                    "score": kw.get("score"),
                    "sentiment_type": "negative",
                }
            )

    stats_count = 0
    for category, stats_data in stats_by_category.items():
        df_stats = pd.DataFrame(stats_data)
        # Hive 파티션 디렉토리 생성
        partition_dir = os.path.join(DETAILED_STATS_DIR, f"category={category}")
        os.makedirs(partition_dir, exist_ok=True)

        stats_path = os.path.join(partition_dir, "data.parquet")
        df_stats.to_parquet(
            stats_path, engine="pyarrow", compression="snappy", index=False
        )
        stats_count += 1
        print(f"  ✓ {category}: {len(df_stats):,}개 키워드")

    print(f"✓ 총 {stats_count}개 카테고리 파티션 생성 완료")

    # 3. partitioned_reviews/ 카테고리별 파티션 생성
    print("\n[3/3] partitioned_reviews 카테고리별 파티션 생성 중...")

    # 리뷰를 카테고리별로 그룹화
    reviews_by_category = {}
    for review in all_reviews:
        product_id = review.get("product_id")
        # product_id에서 카테고리 추출 (예: "선크림_123" -> "선크림")
        # product_id는 "카테고리_원본ID" 형식 (마지막 언더스코어 기준으로 분리)
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

        full_text = review.get("full_text", "")
        has_text = bool(full_text and full_text.strip())

        reviews_by_category.setdefault(category, []).append(
            {
                "product_id": product_id,
                "id": review.get("id"),
                "full_text": full_text,
                "title": review.get("title", ""),
                "content": review.get("content", ""),
                "has_text": has_text,
                "score": review.get("score"),
                "label": review.get("label"),
                "tokens": review.get("tokens"),
                "char_length": review.get("char_length"),
                "token_count": review.get("token_count"),
                "date": review.get("date"),
                "collected_at": review.get("collected_at"),
                "nickname": review.get("nickname"),
                "has_image": review.get("has_image"),
                "helpful_count": review.get("helpful_count"),
                "sentiment_score": None,  # 나중에 모델 예측값 추가
                "word2vec": review.get("word2vec"),
            }
        )

    reviews_count = 0
    for category, review_data in reviews_by_category.items():
        df_reviews = pd.DataFrame(review_data)
        # Hive 파티션 디렉토리 생성
        partition_dir = os.path.join(PARTITIONED_REVIEWS_DIR, f"category={category}")
        os.makedirs(partition_dir, exist_ok=True)

        reviews_path = os.path.join(partition_dir, "data.parquet")
        df_reviews.to_parquet(
            reviews_path, engine="pyarrow", compression="snappy", index=False
        )

        review_size_mb = os.path.getsize(reviews_path) / 1024 / 1024
        reviews_count += 1
        print(f"  ✓ {category}: {len(df_reviews):,}개 리뷰 ({review_size_mb:.2f} MB)")

    print(f"✓ 총 {reviews_count}개 카테고리 파티션 생성 완료")

    # ========== 임시 파일 정리 ==========
    print(f"\n임시 토큰 파일 정리 중...")
    try:
        import shutil

        shutil.rmtree(TEMP_TOKENS_DIR)
        print(f"임시 디렉토리 삭제 완료: {TEMP_TOKENS_DIR}")
    except Exception as e:
        print(f"[경고] 임시 디렉토리 삭제 실패: {e}")

    # 종료 시간 및 소요 시간 계산
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "=" * 60)
    print(f"{'전체 파이프라인 완료!':^60}")
    print(f"{'종료 시간: ' + end_datetime:^60}")
    print(f"{'총 소요 시간: ' + f'{hours}시간 {minutes}분 {seconds}초':^60}")
    print(
        f"{'Phase 1: ' + f'{phase1_time:.1f}초 | Phase 2: {phase2_time:.1f}초 | Phase 3: {phase3_time:.1f}초':^60}"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
