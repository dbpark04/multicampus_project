import json
import copy
from collections import Counter


def drop_missing_val():
    EXCLUDE_TEXT_FIELDS = {"title", "content", "full_text"}

    INPUT_PATH = "result_아이라이너.json"   # 파일명 예시
    OUTPUT_WITH_TEXT = "with_text_drop_missing.json"
    OUTPUT_WITHOUT_TEXT = "without_text_drop_missing.json"


    def has_text(review: dict) -> bool:
        for k in ["content", "full_text"]:
            v = review.get(k)
            if isinstance(v, str) and v.strip():
                return True
        return False


    def drop_missing_fields(obj: dict):
        for k in list(obj.keys()):
            if k in EXCLUDE_TEXT_FIELDS:
                continue

            v = obj[k]
            if v is None or (isinstance(v, str) and v.strip() == ""):
                del obj[k]


    def init_metadata(data: dict):
        data["total_collected_reviews"] = 0
        data["total_text_reviews"] = 0
        data["total_product"] = 0
        data["total_rating_distribution"] = {}


    # 메인
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        original = json.load(f)

    with_text = copy.deepcopy(original)
    without_text = copy.deepcopy(original)

    init_metadata(with_text)
    init_metadata(without_text)

    rating_with = Counter()
    rating_without = Counter()

    data_with = []
    data_without = []

    for product in original["data"]:
        drop_missing_fields(product["product_info"])

        reviews = product["reviews"]["data"]

        p_with = copy.deepcopy(product)
        p_without = copy.deepcopy(product)

        reviews_with = []
        reviews_without = []

        for r in reviews:
            drop_missing_fields(r)

            if has_text(r):
                reviews_with.append(r)
                rating_with[str(r.get("score"))] += 1
            else:
                reviews_without.append(r)
                rating_without[str(r.get("score"))] += 1

        # 리뷰 O
        if reviews_with:
            p_with["reviews"]["data"] = reviews_with
            p_with["reviews"]["total_count"] = len(reviews_with)
            p_with["reviews"]["text_count"] = len(reviews_with)

            data_with.append(p_with)
            with_text["total_product"] += 1
            with_text["total_collected_reviews"] += len(reviews_with)
            with_text["total_text_reviews"] += len(reviews_with)

        # 리뷰 X
        if reviews_without:
            p_without["reviews"]["data"] = reviews_without
            p_without["reviews"]["total_count"] = len(reviews_without)
            p_without["reviews"]["text_count"] = 0

            data_without.append(p_without)
            without_text["total_product"] += 1
            without_text["total_collected_reviews"] += len(reviews_without)

    # 메타데이터 반영
    with_text["data"] = data_with
    with_text["total_rating_distribution"] = dict(rating_with)

    without_text["data"] = data_without
    without_text["total_rating_distribution"] = dict(rating_without)
    without_text["total_text_reviews"] = 0


    # 파일 저장
    with open(OUTPUT_WITH_TEXT, "w", encoding="utf-8") as f:
        json.dump(with_text, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_WITHOUT_TEXT, "w", encoding="utf-8") as f:
        json.dump(without_text, f, ensure_ascii=False, indent=2)

    print("완료:")
    print(f"- 리뷰 있는 파일: {OUTPUT_WITH_TEXT}")
    print(f"- 리뷰 없는 파일: {OUTPUT_WITHOUT_TEXT}")