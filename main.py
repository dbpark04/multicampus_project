import json
import time
import random
import undetected_chromedriver as uc
from get_product_urls import get_product_urls
from get_product_reviews import get_product_reviews


def main():
    KEYWORD = "사과"
    PRODUCT_LIMIT = 3
    REVIEW_TARGET = 30

    print(">>> 브라우저를 실행합니다...")
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument("--password-store=basic")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--blink-settings=imagesEnabled=false")

    driver = uc.Chrome(options=options, use_subprocess=False)

    try:
        print(f">>> [{KEYWORD}] 검색 시작...")
        urls = get_product_urls(driver, KEYWORD, max_products=PRODUCT_LIMIT)
        print(f">>> 수집된 URL: {len(urls)}개")

        crawled_data_list = []
        top_category = ""

        for idx, url in enumerate(urls):
            print(f"\n[{idx+1}/{len(urls)}] 상품 크롤링 중...")

            try:
                # [수정] idx + 1 을 rank_num 인자로 전달 (1, 2, 3...)
                data = get_product_reviews(
                    driver, url, idx + 1, target_review_count=REVIEW_TARGET
                )

                if data["product_info"]:
                    current_category = data["product_info"].get("category_path")
                    if not top_category and current_category:
                        top_category = current_category

                    crawled_data_list.append(data)
                    print(f"  -> 완료: 리뷰 {data['reviews']['count']}개")
                else:
                    print("  -> 실패")

                time.sleep(random.uniform(3, 5))

            except Exception as e:
                print(f"  -> 에러: {e}")

        result_json = {
            "search_name": KEYWORD,
            "category": top_category,
            "data": crawled_data_list,
        }

        if crawled_data_list:
            filename = f"result_{KEYWORD}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            print(f"\n 저장 완료: {filename}")
        else:
            print("\n수집된 데이터가 없습니다.")

    finally:
        print(">>> 브라우저를 종료합니다.")
        driver.quit()


if __name__ == "__main__":
    main()
