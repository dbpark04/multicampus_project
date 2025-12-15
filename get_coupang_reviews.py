# coupang_crawler.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random


def get_coupang_reviews(url, target_review_count=100):
    """
    쿠팡 상품 URL을 받아 상품 정보와 리뷰를 수집하는 함수
    Args:
        url (str): 상품 상세 페이지 URL
        target_review_count (int): 수집할 리뷰 개수 목표치
    Returns:
        dict: 상품 정보와 리뷰 리스트가 담긴 딕셔너리
    """

    # 브라우저 옵션 설정
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument("--password-store=basic")

    options.add_argument("--window-size=1280,1024")
    options.add_argument("--blink-settings=imagesEnabled=false")
    driver = uc.Chrome(options=options, use_subprocess=False)

    result_data = {"product_info": {}, "reviews": {}}
    temp_reviews_list = []

    try:
        print(f"[Coupang Crawler] 상품 페이지 접속: {url}")
        driver.get(url)
        time.sleep(random.uniform(3, 5))

        # HTML 파싱
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # --- [1] 상품 기본 정보 수집 ---
        print("기본 정보 수집 중...")

        product_name = "Unknown"
        try:
            product_name = soup.select_one("span.twc-font-bold").text.strip()
        except:
            pass

        price = "0"
        try:
            price_tag = soup.select_one("div.price-amount.final-price-amount")
            if not price_tag:
                price_tag = soup.select_one(
                    "div.option-table-list__option--selected div.option-table-list__option-price"
                )
            price = (
                price_tag.text.strip()
                .replace("원", "")
                .replace(",", "")
                .split()[0]
                .strip()
            )
        except:
            pass

        delivery_type = "Standard"
        try:
            badge_img = soup.select_one("div.price-badge img")
            if badge_img:
                src = badge_img.get("src", "")
                if "rocket-fresh" in src:
                    delivery_type = "Rocket Fresh"
                elif "badge_1998ab96bf7" in src:
                    delivery_type = "Rocket Delivery(Coupang)"
                elif "badge_1998ab98cb6" in src:
                    delivery_type = "Rocket Delivery(Partner)"
                elif "badge_199559e56f7" in src:
                    delivery_type = "Seller Rocket"
        except:
            pass

        total_reviews = "0"
        try:
            count_text = soup.select_one("span.rating-count-txt").text.strip()
            total_reviews = count_text.split("개")[0].replace(",", "").strip()
        except:
            pass

        result_data["product_info"] = {
            "product_name": product_name,
            "price": price,
            "delivery_type": delivery_type,
            "total_reviews": total_reviews,
            "product_url": url,
        }
        print(f"   -> 상품명: {product_name}")

        # --- [2] 리뷰 데이터 수집 ---
        print(f"리뷰 수집 시작 (목표: {target_review_count}개)")

        # 스크롤
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.3);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.6);")
        time.sleep(1)

        try:
            review_section = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "sdpReview"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", review_section)
            driver.execute_script("window.scrollBy(0, -200);")
            time.sleep(2)
        except:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # 최신순 정렬
        try:
            sort_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), '최신순')]")
                )
            )
            driver.execute_script("arguments[0].click();", sort_btn)
            time.sleep(3)
            print("최신순 정렬 완료")
        except:
            print("정렬 실패, 기본순으로 진행")

        current_page_num = 1
        collected_count = 0

        while collected_count < target_review_count:
            print(f"{current_page_num}페이지 수집 중... (현재 {collected_count}개)")

            curr_html = driver.page_source
            curr_soup = BeautifulSoup(curr_html, "html.parser")
            review_articles = curr_soup.select("article.twc-border-bluegray-200")

            if not review_articles:
                print("  - 더 이상 리뷰가 없습니다.")
                break

            for article in review_articles:
                if collected_count >= target_review_count:
                    break

                try:
                    rating = 0
                    rating_div = article.select_one(
                        r"div.twc-inline-flex.twc-items-center.twc-gap-\[2px\]"
                    )
                    if rating_div:
                        rating = len(rating_div.select("i.twc-bg-full-star"))

                    date_div = article.select_one("div.twc-text-bluegray-700")
                    review_date = date_div.text.strip() if date_div else ""
                    if "판매자" in review_date:
                        review_date = "Unknown"

                    title_div = article.select_one(
                        "div.twc-font-bold.twc-text-bluegray-900"
                    )
                    review_title = title_div.text.strip() if title_div else ""

                    content_span = article.select_one("span.twc-bg-white")
                    review_content = content_span.text.strip() if content_span else ""

                    full_text = f"{review_title} {review_content}".strip()

                    has_image = False
                    img_container = article.select_one(
                        "div.twc-overflow-x-auto.twc-scrollbar-hidden"
                    )
                    if img_container and img_container.select_one("img"):
                        has_image = True

                    review_obj = {
                        "id": collected_count + 1,
                        "date": review_date,
                        "rating": rating,
                        "has_image": has_image,
                        "title": review_title,
                        "content": review_content,
                        "full_text": full_text,
                    }
                    temp_reviews_list.append(review_obj)
                    collected_count += 1

                except Exception:
                    continue

            if collected_count >= target_review_count:
                print("목표 수집량 달성!")
                break

            # --- 페이지 이동 로직 ---
            next_target_num = current_page_num + 1

            # 10페이지 단위 이동
            if current_page_num % 10 == 0:
                print(f"다음 목록(>)으로 이동 중...")
                try:
                    next_arrow_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "//div[contains(@class, 'twc-mt-[24px]') and contains(@class, 'twc-flex-wrap')]//button[last()]",
                            )
                        )
                    )
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});",
                        next_arrow_btn,
                    )
                    driver.execute_script("arguments[0].click();", next_arrow_btn)
                    time.sleep(random.uniform(2, 3.5))
                    current_page_num += 1
                    continue
                except:
                    print("다음 페이지가 없습니다.")
                    break

            # 일반 숫자 페이지 이동
            try:
                next_num_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, f"//button[.//span[text()='{next_target_num}']]")
                    )
                )
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", next_num_btn
                )
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", next_num_btn)
                time.sleep(random.uniform(1, 1.7))
                current_page_num += 1
            except:
                print("다음 페이지 버튼을 찾을 수 없습니다.")
                break

        # 최종 데이터 할당
        result_data["reviews"] = {
            "count": len(temp_reviews_list),
            "data": temp_reviews_list,
        }

    except Exception as e:
        print(f"에러 발생: {e}")

    finally:
        driver.quit()
        return result_data
