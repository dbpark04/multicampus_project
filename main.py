# main.py
import json
from get_coupang_reviews import get_coupang_reviews


def main():
    # 수집할 상품 URL (예: 사과)
    target_url = (
        "https://www.coupang.com/vp/products/5611991510?vendorItemId=92083385400"
    )

    # 목표 리뷰 개수 설정
    TARGET_COUNT = 130

    # 크롤링 실행 (모듈에서 함수 호출)
    print(">>> 크롤러 실행을 시작합니다...")
    final_data = get_coupang_reviews(target_url, target_review_count=TARGET_COUNT)

    if final_data["product_info"]:
        print(f"\n수집 성공! (총 리뷰: {final_data['reviews']['count']}개)")

        # 파일 저장
        file_name = "coupang_result.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"결과 파일 저장 완료: {file_name}")
    else:
        print("수집 실패")


# Mac multiprocessing safe guard
if __name__ == "__main__":
    main()
