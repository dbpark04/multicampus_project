"""
기존 JSON 파일에 상품 이미지 URL 추가하는 스크립트
"""

import json
import time
import os
import glob
from pathlib import Path
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def get_product_image_url(driver, product_url, max_retries=3):
    """
    상품 페이지에서 이미지 URL을 가져옴

    Args:
        driver: Selenium WebDriver
        product_url: 상품 URL
        max_retries: 최대 재시도 횟수

    Returns:
        str: 이미지 URL 또는 None
    """
    if driver is None:
        return None
    for attempt in range(max_retries):
        try:
            driver.get(product_url)
            time.sleep(2)  # 페이지 로딩 대기

            # img 태그 찾기
            # <img alt="Product image" class="twc-w-full twc-max-h-[546px]" src="...">
            img_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        'img[alt="Product image"].twc-w-full.twc-max-h-\\[546px\\]',
                    )
                )
            )

            image_url = img_element.get_attribute("src")

            if image_url:
                # print(f"      ✓ 이미지 URL 추출 성공")
                return image_url
            else:
                # 마지막 시도가 아니면 에러 출력 안 함
                if attempt == max_retries - 1:
                    print(f"      ✗ src 속성이 비어있음")

        except TimeoutException:
            # 마지막 시도가 아니면 에러 출력 안 함
            if attempt == max_retries - 1:
                print(f"      ✗ 이미지 요소 찾기 실패 - 타임아웃")
        except NoSuchElementException:
            # 마지막 시도가 아니면 에러 출력 안 함
            if attempt == max_retries - 1:
                print(f"      ✗ 이미지 요소 없음")
        except Exception as e:
            # 마지막 시도가 아니면 에러 출력 안 함
            if attempt == max_retries - 1:
                print(f"      ✗ 에러 발생: {e}")

        if attempt < max_retries - 1:
            time.sleep(2)

    return None


def process_json_file(driver, json_file_path, output_file_path):
    """
    JSON 파일을 읽어서 이미지 URL이 없는 상품에 추가

    Args:
        driver: Selenium WebDriver
        json_file_path: 원본 JSON 파일 경로
        output_file_path: 저장할 JSON 파일 경로

    Returns:
        tuple: (처리된 상품 수, 업데이트된 상품 수, consecutive_failures)
    """
    start_time = time.time()  # 시작 시간 기록

    # JSON 파일 읽기
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    products = data.get("data", [])
    total_products = len(products)
    updated_count = 0
    skipped_count = 0
    failed_products = []  # 실패한 상품 정보 저장
    consecutive_failures = 0  # 연속 실패 카운터

    for idx, product in enumerate(products):
        product_info = product.get("product_info", {})
        product_url = product_info.get("product_url")
        product_name = product_info.get("product_name", "(이름 없음)")[:50]

        # image_url이 이미 있으면 스킵
        if "image_url" in product_info and product_info["image_url"]:
            skipped_count += 1
            continue

        if not product_url:
            skipped_count += 1
            continue

        # 드라이버가 None이면 건너뛰기 (재시작 대기 중)
        if driver is None:
            failed_products.append(
                {"index": idx + 1, "name": product_name, "url": product_url}
            )
            continue

        # 이미지 URL 가져오기
        image_url = get_product_image_url(driver, product_url)

        if image_url:
            product_info["image_url"] = image_url
            updated_count += 1
            consecutive_failures = 0  # 성공 시 카운터 리셋
        else:
            # 실패한 상품 정보 저장
            failed_products.append(
                {"index": idx + 1, "name": product_name, "url": product_url}
            )
            consecutive_failures += 1  # 실패 카운터 증가

        # 다음 상품 전 대기
        time.sleep(1)

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # 업데이트된 데이터를 temp_pre_data에 저장
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 결과 출력
    print(f"\n{'='*70}")
    print(f"✓ 파일 처리 완료: {os.path.basename(json_file_path)}")
    print(f"  - 총 상품: {total_products}개")
    print(f"  - 스킵: {skipped_count}개 (이미 있음 또는 URL 없음)")
    print(f"  - 업데이트 성공: {updated_count}개")
    print(f"  - 실패: {len(failed_products)}개")

    # 실패한 상품이 있으면 상세 출력
    if failed_products:
        print(f"\n  [실패 상품 목록]")
        for fail in failed_products:
            print(f"    [{fail['index']}/{total_products}] {fail['name']}...")
            print(f"      URL: {fail['url']}")

    # 처리 시간 계산 및 출력
    elapsed_time = time.time() - start_time
    print(f"  소요 시간: {elapsed_time:.2f}초")
    print(f"  저장: {output_file_path}")
    print(f"{'='*70}")

    return total_products, updated_count, consecutive_failures


def driver_cleanup(driver):
    """드라이버 종료 및 메모리 정리"""
    try:
        if driver:
            driver.quit()
    except:
        pass
    return None


def main():
    """
    메인 함수: data/pre_data 디렉토리의 모든 JSON 파일 처리
    """
    PRE_DATA_DIR = "./data/pre_data"
    TEMP_PRE_DATA_DIR = "./data/temp_pre_data"
    CONSECUTIVE_FAIL_LIMIT = 10  # 연속 실패 허용 횟수
    WAIT_TIME_ON_CONSECUTIVE_FAIL = 20 * 60  # 20분 (초 단위)

    print("\n" + "=" * 70)
    print("상품 이미지 URL 추가 스크립트 시작")
    print("=" * 70)

    # JSON 파일 찾기
    json_files = glob.glob(
        os.path.join(PRE_DATA_DIR, "**", "result_*.json"), recursive=True
    )

    # interrupted 파일 제외
    json_files = [f for f in json_files if "interrupted" not in f]

    # 이미 temp_pre_data에 존재하는 파일 제외
    files_to_process = []
    for json_file in json_files:
        output_file = json_file.replace(PRE_DATA_DIR, TEMP_PRE_DATA_DIR)
        if os.path.exists(output_file):
            print(f"✓ 이미 처리됨 - 건너뜀: {os.path.basename(json_file)}")
        else:
            files_to_process.append(json_file)

    json_files = files_to_process

    if not json_files:
        print(f"\n✗ {PRE_DATA_DIR}에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"\n총 {len(json_files)}개 파일 발견:")
    for f in json_files:
        print(f"  - {f}")

    print(f"\n저장 위치: {TEMP_PRE_DATA_DIR}")
    print("(원본 파일은 수정되지 않습니다)")

    # Chrome 드라이버 시작
    print("\n브라우저 시작 중...")
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument("--password-store=basic")
    options.add_argument("--window-size=1920,1080")
    # 이미지 로드 활성화 (이미지 URL을 가져와야 하므로)
    # options.add_argument("--blink-settings=imagesEnabled=false")  # 주석 처리

    driver = uc.Chrome(options=options, use_subprocess=False)

    try:
        total_all_products = 0
        total_all_updated = 0
        global_consecutive_failures = 0  # 전역 연속 실패 카운터

        # 각 JSON 파일 처리
        for file_idx, json_file in enumerate(json_files):
            # 출력 파일 경로 생성 (pre_data를 temp_pre_data로 치환)
            output_file = json_file.replace(PRE_DATA_DIR, TEMP_PRE_DATA_DIR)

            processed, updated, consecutive_failures = process_json_file(
                driver, json_file, output_file
            )
            total_all_products += processed
            total_all_updated += updated
            global_consecutive_failures = consecutive_failures

            # 연속 실패 체크
            if global_consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                print(f"\n!!! 연속 {global_consecutive_failures}번 실패 감지 !!!")
                print(
                    f"!!! {WAIT_TIME_ON_CONSECUTIVE_FAIL // 60}분 대기 후 재시도합니다..."
                )

                # 드라이버 종료
                driver = driver_cleanup(driver)

                # 20분 대기
                time.sleep(WAIT_TIME_ON_CONSECUTIVE_FAIL)

                # 드라이버 재시작
                print("\n브라우저 재시작 중...")
                driver = uc.Chrome(options=options, use_subprocess=False)
                global_consecutive_failures = 0  # 카운터 리셋

            # 다음 파일 전 대기 (마지막 파일이 아닐 때만)
            if file_idx < len(json_files) - 1:
                time.sleep(5)

        # 최종 결과 출력
        print("\n\n" + "=" * 70)
        print("작업 완료!")
        print("=" * 70)
        print(f"총 처리 파일 수: {len(json_files)}개")
        print(f"총 처리 상품 수: {total_all_products}개")
        print(f"총 업데이트 상품 수: {total_all_updated}개")
        print(f"저장 위치: {TEMP_PRE_DATA_DIR}")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n✗ 에러 발생: {e}")
    finally:
        print("\n브라우저 종료 중...")
        driver = driver_cleanup(driver)


if __name__ == "__main__":
    main()
