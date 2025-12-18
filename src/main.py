import json
import time
import random
import undetected_chromedriver as uc
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoSuchWindowException,
    WebDriverException,
)
import gc

from get_product_urls import get_product_urls, get_category_product_urls
from get_product_reviews import get_product_reviews


def main():
    # [방법 1] 키워드 검색을 원할 때
    # MODE = "KEYWORD"
    # TARGETS = ["사과"]

    # [방법 2] 카테고리 수집을 원할 때
    MODE = "CATEGORY"
    TARGETS = {"스킨": "486248", "로션": "486249", "에센스_세럼_앰플": "486250"}
    PRODUCT_LIMIT = 300
    REVIEW_TARGET = 600
    MAX_REVIEWS_PER_SEARCH = 50000

    print(">>> 전체 작업을 시작합니다...")

    try:
        # 반복문 시작 부분 수정
        if MODE == "KEYWORD":
            iterable = enumerate(TARGETS)
        else:
            iterable = enumerate(TARGETS.items())  # (0, ("스킨케어", "486248"))

        for k_idx, item in iterable:
            # 이 키워드/카테고리 시작 시간 기록
            search_start_time = time.time()

            # 모드에 따라 변수 할당
            if MODE == "KEYWORD":
                search_key = item  # "사과"
                search_id = None
            else:
                search_key = item[0]  # "스킨케어"
                search_id = item[1]  # "486248"
            crawled_data_list = []
            keyword_total_collected = 0
            keyword_total_text = 0

            # 전체 별점 분포 집계
            total_rating_distribution = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}

            # ---------------------------------------------------------
            # [단계 1] URL 수집
            # ---------------------------------------------------------
            total_count = len(TARGETS) if MODE == "KEYWORD" else len(TARGETS.items())
            print(f"\n{'='*50}")
            print(f">>> [{k_idx+1}/{total_count}] '{search_key}' URL 수집 시작")
            print(f"{'='*50}")

            # URL 수집 재시도 로직
            URL_COLLECT_MAX_RETRIES = 3
            urls = []

            for url_attempt in range(URL_COLLECT_MAX_RETRIES):
                print(
                    f"\n>>> URL 수집 시도 [{url_attempt+1}/{URL_COLLECT_MAX_RETRIES}]"
                )

                options = uc.ChromeOptions()
                options.add_argument("--no-first-run")
                options.add_argument("--no-service-autorun")
                options.add_argument("--password-store=basic")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--blink-settings=imagesEnabled=false")

                driver = uc.Chrome(options=options, use_subprocess=False)
                try:
                    # 모드에 따라 호출 함수 분기
                    if MODE == "KEYWORD":
                        urls = get_product_urls(
                            driver, search_key, max_products=PRODUCT_LIMIT
                        )
                    else:
                        # 카테고리 ID로 수집 함수 호출
                        urls = get_category_product_urls(
                            driver, search_id, max_products=PRODUCT_LIMIT
                        )

                    print(f">>> [{search_key}] URL {len(urls)}개 확보 완료")
                    # 중복 urls 제거
                    urls = list(set(urls))
                    print(
                        f">>> [{search_key}] 중복 제거 후 URL {len(urls)}개 확보 완료"
                    )
                    time.sleep(1)
                    # 성공하면 루프 탈출
                    if urls:
                        break
                    else:
                        print(f">>> URL 수집 실패 (0개) - 재시도합니다.")

                except Exception as e:
                    print(f">>> URL 수집 중 에러: {e}")
                    urls = []
                finally:
                    print(">>> URL 수집 브라우저 종료 및 메모리 정리 중...")
                    driver = driver_cleanup(driver)

                # 마지막 시도가 아니면 대기
                if url_attempt < URL_COLLECT_MAX_RETRIES - 1 and not urls:
                    print(">>> 20초 대기 후 재시도...")
                    time.sleep(20)

            if not urls:
                print(
                    f">>> [{search_key}] {URL_COLLECT_MAX_RETRIES}번 시도 후에도 URL을 수집하지 못했습니다. 넘어갑니다."
                )
                continue

            # ---------------------------------------------------------
            # [단계 2] 개별 상품 리뷰 수집 (리뷰 수에 따라 드라이버 재사용/재시작)
            # ---------------------------------------------------------
            print(f">>> [{search_key}] 상세 리뷰 수집 시작")

            # 첫 상품을 위한 드라이버 생성
            driver = None
            driver_collected_count = 0  # 현재 드라이버가 수집한 총 리뷰 개수

            for idx, url in enumerate(urls):
                print(f"\n   [{idx+1}/{len(urls)}] 상품 처리 시작... ({search_key})")

                MAX_RETRIES = 3
                success = False

                for attempt in range(MAX_RETRIES):
                    print(
                        f"     -> [시도 {attempt+1}/{MAX_RETRIES}] 브라우저 실행 중..."
                    )

                    try:
                        # 드라이버가 없으면 새로 생성
                        if driver is None:
                            options = uc.ChromeOptions()
                            options.add_argument("--no-first-run")
                            options.add_argument("--no-service-autorun")
                            options.add_argument("--password-store=basic")
                            options.add_argument("--window-size=1920,1080")
                            options.add_argument("--blink-settings=imagesEnabled=false")
                            driver = uc.Chrome(options=options, use_subprocess=False)
                            driver_collected_count = 0

                        # 수집 함수 호출 (현재 드라이버 수집량 전달)
                        data = get_product_reviews(
                            driver,
                            url,
                            idx + 1,
                            target_review_count=REVIEW_TARGET,
                            driver_collected_count=driver_collected_count,
                        )

                        if (
                            data
                            and data.get("product_info")
                            and data["product_info"].get("product_id")
                        ):
                            r_data = data.get("reviews", {})
                            current_collected = r_data.get("total_count", 0)

                            # 리뷰가 0개면 저장하지 않고 다음으로
                            if current_collected == 0:
                                print(
                                    "     -> [스킵] 리뷰가 0개입니다. 저장하지 않고 다음 상품으로 넘어갑니다."
                                )
                                success = True
                                break

                            keyword_total_collected += current_collected
                            keyword_total_text += r_data.get("text_count", 0)
                            driver_collected_count += current_collected

                            # 상품별 rating_distribution을 전체에 합산
                            product_rating = data.get("product_info", {}).get(
                                "rating_distribution", {}
                            )
                            for score, count in product_rating.items():
                                total_rating_distribution[score] += count

                            crawled_data_list.append(data)
                            print(
                                f"     -> [성공] 수집 완료 (전체: {current_collected}개, 글 포함: {r_data.get('text_count', 0)}개)"
                            )
                            print(
                                f"     -> 키워드 누적: {keyword_total_collected}개 / 현 드라이버: {driver_collected_count}개"
                            )

                            success = True

                            # 타겟 리뷰 개수 도달 체크
                            if keyword_total_collected >= MAX_REVIEWS_PER_SEARCH:
                                print(
                                    f"\n>>> [{search_key}] 타겟 리뷰 개수({MAX_REVIEWS_PER_SEARCH}개) 도달!"
                                )
                                print(
                                    f">>> 현재 수집: {keyword_total_collected}개 / URL 남음: {len(urls) - idx - 1}개"
                                )
                                print(f">>> 수집을 종료하고 저장합니다.")
                                break

                            break
                        else:
                            print("     -> [실패] 데이터가 비어있습니다. 재시도합니다.")
                            # 드라이버 재시작
                            driver = driver_cleanup(driver)
                            driver_collected_count = 0

                    except Exception as e:
                        print(f"     -> [에러 발생] {e}")
                        # 에러 발생 시 드라이버 재시작
                        if driver:
                            driver = driver_cleanup(driver)
                            driver_collected_count = 0

                        continue

                # 2번 다 실패했을 경우
                if not success:
                    print(
                        f"     -> [최종 실패] {MAX_RETRIES}번 시도했으나 수집 실패. 다음 상품으로 넘어갑니다."
                    )

                # 타겟 리뷰 개수 도달 시 URL 루프 탈출
                if keyword_total_collected >= MAX_REVIEWS_PER_SEARCH:
                    break

                # print(f"     -> 다음 상품 대기중...(2초)")
                time.sleep(2)

            # 키워드/카테고리 처리 완료 후 드라이버가 남아있으면 종료
            if driver:
                print(f">>> [{search_key}] 모든 상품 처리 완료 - 드라이버 종료")
                driver = driver_cleanup(driver)

            # ---------------------------------------------------------
            # [단계 3] 키워드/카테고리 완료 후 저장
            # ---------------------------------------------------------
            result_json = {
                "search_name": search_key,
                "total_collected_reviews": keyword_total_collected,
                "total_text_reviews": keyword_total_text,
                "total_product": len(crawled_data_list),
                "total_rating_distribution": total_rating_distribution,
                "data": crawled_data_list,
            }

            if crawled_data_list:
                filename = f"result_{search_key}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)
                print(f"\n [{search_key}] 저장 완료: {filename}")
            else:
                print(f"\n[{search_key}] 수집된 데이터가 없습니다.")

            search_end_time = time.time()
            search_elapsed = search_end_time - search_start_time
            search_hours = int(search_elapsed // 3600)
            search_minutes = int((search_elapsed % 3600) // 60)
            search_seconds = int(search_elapsed % 60)
            print(
                f"\n[{search_key}] 처리 시간: {search_hours}시간 {search_minutes}분 {search_seconds}초"
            )

            # 다음 키워드/카테고리 준비 중 (마지막이 아닐 때만)
            if k_idx < total_count - 1:  # 마지막이 아닐 때만
                print(f">>> 다음 항목 준비 중 (20.0초)...")
                time.sleep(20)

    except KeyboardInterrupt:
        print("\n>>> 사용자에 의해 작업이 중단되었습니다.")

        # 드라이버 정리
        try:
            if driver:
                print(">>> 드라이버 정리 중...")
                driver = driver_cleanup(driver)
        except:
            pass

        # 현재까지 수집된 데이터가 있으면 저장
        try:
            if crawled_data_list:
                result_json = {
                    "search_name": search_key,
                    "total_collected_reviews": keyword_total_collected,
                    "total_text_reviews": keyword_total_text,
                    "total_product": len(crawled_data_list),
                    "total_rating_distribution": total_rating_distribution,
                    "data": crawled_data_list,
                }

                filename = f"result_{search_key}_interrupted.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)
                print(f">>> 중단 시점까지의 데이터 저장 완료: {filename}")
                print(
                    f">>> 저장된 데이터: 상품 {len(crawled_data_list)}개, 리뷰 {keyword_total_collected}개"
                )
        except Exception as e:
            print(f">>> 데이터 저장 중 오류: {e}")


def driver_cleanup(driver):
    try:
        driver.quit()
        print("driver 종료 및 20초 대기")
        try:
            del driver
        except Exception as e:
            print(f"드라이버 삭제 중 에러(무시됨): {e}")
        gc.collect()
        driver = None
        time.sleep(22)
    except Exception as e:
        print(f"드라이버 종료 중 에러(무시됨): {e}")

    return None


if __name__ == "__main__":
    main()
