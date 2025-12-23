import json
import re
from typing import Dict, List
from collections import Counter


# =========================
# 기본 유틸
# =========================
def to_lower(text):
    if not isinstance(text, str):
        return text
    return text.lower().strip()


# =========================
# 브랜드 통합
# =========================
BRAND_MAPPING = {
    "토코보": "tocobo",
    "tocobo": "tocobo",
    "라네즈": "laneige",
    "laneige": "laneige",
    "이니스프리": "innisfree",
    "innisfree": "innisfree",
}

def normalize_brand(brand: str) -> str:
    brand = to_lower(brand)
    return BRAND_MAPPING.get(brand, brand)


# =========================
# 카테고리 통합
# =========================
CATEGORY_MAPPING = {
    "스킨": "스킨", "토너": "스킨", "skin": "스킨", "toner": "스킨",
    "로션": "로션", "lotion": "로션", "emulsion": "로션", "에멀전": "로션",
    "에센스": "에센스", "essence": "에센스",
    "세럼": "세럼", "serum": "세럼",
    "크림": "크림", "cream": "크림",
    "선스틱": "선스틱", "sun stick": "선스틱", "sunstick": "선스틱",
    "선크림": "선크림", "썬크림": "선크림", "sun cream": "선크림", "sunscreen": "선크림",
    "클렌저": "클렌저", "cleanser": "클렌저",
}

def normalize_category(category_path: str) -> str:
    if not isinstance(category_path, str):
        return category_path
    leaf = category_path.split(">")[-1].strip().lower()
    return CATEGORY_MAPPING.get(leaf, leaf)


# =========================
# 동의어 처리
# =========================
SYNONYM_PATTERNS = {
    r"machine\s*learning|머신\s*learning|머신\s*러닝|ml": "머신러닝",
    r"skin\s*care|스킨\s*케어": "스킨케어",
    r"skin|toner|토너": "스킨",
    r"lotion|emulsion|에멀전": "로션",
    r"essence|에센스": "에센스",
    r"serum|세럼": "세럼",
    r"cream|크림": "크림",
    r"sun\s*stick|sunstick|선\s*스틱": "선스틱",
    r"sun\s*cream|sunscreen|썬\s*크림|선\s*크림": "선크림",
    r"cleanser|cleansing\s*foam|폼\s*클렌저": "클렌저",
}

def normalize_synonyms(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.lower()
    for pattern, replacement in SYNONYM_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


# =========================
# 브랜드 패턴 생성 & 제거
# =========================
def build_brand_patterns(brands):
    patterns = {}
    for b in brands:
        if not b:
            continue
        spaced = r"\s*".join(list(b))
        patterns[b] = rf"{b}|{spaced}"
    return patterns

def remove_brand_from_name(name: str, brand: str, brand_patterns: dict) -> str:
    if not isinstance(name, str) or not isinstance(brand, str):
        return name
    pattern = brand_patterns.get(brand)
    if not pattern:
        return name
    name = re.sub(pattern, " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


# =========================
# 상품명 노이즈 제거
# =========================
NOISE_PATTERNS = [
    r"\[.*?\]",
    r"\d+\s?(ml|g)",
    r"\d+\s?개(입)?",
    r"spf\s?\d+\+?",
    r"pa\+{1,4}",
    r"[★☆■◆▶️]",
]

def clean_product_name(name: str, brand_norm: str, brand_patterns: dict) -> str:
    name = to_lower(name)
    for pattern in NOISE_PATTERNS:
        name = re.sub(pattern, " ", name)
    name = remove_brand_from_name(name, brand_norm, brand_patterns)
    name = normalize_synonyms(name)
    return re.sub(r"\s+", " ", name).strip()


# =========================
# 토큰화
# =========================
def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return text.split()


# =========================
# product_info 전처리
# =========================
def preprocess_product_info(product_info: Dict, brand_patterns: dict) -> Dict:
    product_info = product_info.copy()

    brand_norm = normalize_brand(product_info.get("brand"))
    product_info["brand_norm"] = brand_norm

    product_info["category_norm"] = normalize_category(
        product_info.get("category_path")
    )

    product_info["product_name_clean"] = clean_product_name(
        product_info.get("product_name"),
        brand_norm,
        brand_patterns
    )

    product_info["product_tokens"] = tokenize(
        product_info["product_name_clean"]
    )

    return product_info


# =========================
# 전체 JSON 전처리
# =========================
def preprocess_json(raw_json: Dict) -> Dict:
    brands = {
        normalize_brand(
            item.get("product_info", {}).get("brand")
        )
        for item in raw_json.get("data", [])
    }

    brand_patterns = build_brand_patterns(brands)

    for item in raw_json.get("data", []):
        if "product_info" in item:
            item["product_info"] = preprocess_product_info(
                item["product_info"],
                brand_patterns
            )
    return raw_json


# =========================
# 토큰 분석 (선택)
# =========================
def analyze_tokens(processed_json: Dict, top_n: int = 30):
    counter = Counter()
    for item in processed_json.get("data", []):
        counter.update(
            item.get("product_info", {}).get("product_tokens", [])
        )
    return counter.most_common(top_n)


# =========================
# 실행부
# =========================
if __name__ == "__main__":
    with open("raw_data.json", "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    processed_json = preprocess_json(raw_json)

    with open("cleaned_data.json", "w", encoding="utf-8") as f:
        json.dump(processed_json, f, ensure_ascii=False, indent=2)

    print("전처리 완료")

    print("상위 토큰:")
    for token, cnt in analyze_tokens(processed_json, 20):
        print(token, cnt)
