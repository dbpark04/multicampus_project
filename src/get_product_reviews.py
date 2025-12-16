import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import time
import random


def get_product_reviews(driver, url, rank_num, target_review_count=100):
    result_data = {
        "product_info": {},
        "reviews": {},
    }

    # [변경] 내부 try-except를 최소화하고, 재시도 로직을 제거함
    # 에러 발생 시 main.py가 잡아서 브라우저를 재부팅하도록 유도
    print(f"[Reviewer] 상품 페이지 접속: {url}")

    # 1. 페이지 접속
    driver.set_page_load_timeout(30)  # 타임아웃 넉넉하게
    driver.get(url)

    # 접속 직후 알림창 처리 (이건 여기서 해주는 게 좋음)
    try:
        alert = driver.switch_to.alert
        print(f"   -> ⚠️ 접속 직후 경고창 감지: {alert.text}")
        alert.accept()
    except:
        pass

    time.sleep(random.uniform(3, 5))

    # 2. HTML 파싱 시작
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    product_id = str(rank_num)
    product_name = "Unknown"

    try:
        product_name = soup.select_one("span.twc-font-bold").text.strip()
    except:
        try:
            product_name = soup.select_one("h2.prod-buy-header__title").text.strip()
        except:
            pass

    price = "0"
    try:
        price_tag = soup.select_one("div.price-amount.final-price-amount")
        if not price_tag:
            price_tag = soup.select_one(
                "div.option-table-list__option--selected div.option-table-list__option-price"
            )
        if price_tag:
            price = (
                price_tag.text.strip()
                .replace("원", "")
                .replace(",", "")
                .split()[0]
                .strip()
            )
    except:
        pass

    delivery_type = "일반배송"
    try:
        badge_img = soup.select_one("div.price-badge img")
        if badge_img:
            src = badge_img.get("src", "")
            if "rocket-fresh" in src:
                delivery_type = "로켓프레시"
            elif "badge_1998ab96bf7" in src:
                delivery_type = "로켓배송(쿠팡)"
            elif "badge_1998ab98cb6" in src:
                delivery_type = "로켓배송(파트너사)"
            elif "badge_199559e56f7" in src:
                delivery_type = "판매자 로켓"
    except:
        pass

    total_reviews = "0"
    try:
        review_count_text = soup.select_one("span.rating-count-txt").text.strip()
        total_reviews = review_count_text.split("개")[0].replace(",", "").strip()
    except:
        pass

    category_str = ""
    try:
        crumb_links = soup.select("ul.breadcrumb li a")
        category_str = " > ".join(
            [link.text.strip() for link in crumb_links if link.text.strip()]
        )
    except:
        pass

    result_data["product_info"] = {
        "product_id": product_id,
        "category_path": category_str,
        "product_name": product_name,
        "price": price,
        "delivery_type": delivery_type,
        "total_reviews": total_reviews,
        "product_url": url,
    }

    print(f"   -> 상품ID: {product_id} / 상품명: {product_name}")
    print(f"   -> 가격: {price}원 / 배송: {delivery_type} / 총리뷰: {total_reviews}")

    # --- 리뷰 수집 시작 ---
    temp_reviews_list = []

    # 초기 리뷰 로딩을 위한 스크롤
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.3);")
    time.sleep(1)

    try:
        review_section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "sdpReview"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", review_section)
        driver.execute_script("window.scrollBy(0, -200);")
        time.sleep(2)
    except:
        pass

    # 최신순 정렬
    try:
        sort_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), '최신순')]")
            )
        )
        driver.execute_script("arguments[0].click();", sort_btn)
        time.sleep(2)
    except:
        pass

    current_page_num = 1
    collected_count = 0  # 내용이 있는 리뷰 개수 (종료 기준)

    while collected_count < target_review_count:
        # 1. 현재 페이지 리뷰 파싱
        curr_soup = BeautifulSoup(driver.page_source, "html.parser")
        review_articles = curr_soup.select("article.twc-border-bluegray-200")

        if not review_articles:
            break

        for article in review_articles:
            if collected_count >= target_review_count:
                break

            try:
                content_span = article.select_one("span.twc-bg-white")
                content = content_span.text.strip() if content_span else ""

                rating = 0
                rating_div = article.select_one(
                    r"div.twc-inline-flex.twc-items-center.twc-gap-\[2px\]"
                )
                if rating_div:
                    rating = len(rating_div.select("i.twc-bg-full-star"))

                date_div = article.select_one("div.twc-text-bluegray-700")
                date = date_div.text.strip() if date_div else ""

                title_div = article.select_one(
                    "div.twc-font-bold.twc-text-bluegray-900"
                )
                title = title_div.text.strip() if title_div else ""

                has_image = False
                img_container = article.select_one(
                    "div.twc-overflow-x-auto.twc-scrollbar-hidden"
                )
                if img_container and img_container.select_one("img"):
                    has_image = True

                review_obj = {
                    "id": len(temp_reviews_list) + 1,
                    "date": date,
                    "rating": rating,
                    "has_image": has_image,
                    "title": title,
                    "content": content,
                    "full_text": f"{title} {content}",
                }

                temp_reviews_list.append(review_obj)

                if content:
                    collected_count += 1

            except:
                continue

        print(
            f"   -> {current_page_num}페이지 탐색 중... (유효: {collected_count}/{target_review_count}, 전체수집: {len(temp_reviews_list)})"
        )

        if collected_count >= target_review_count:
            break

        # 2. 페이지 이동 로직
        if current_page_num % 10 == 0:
            print(
                f"   -> 페이지 블록 이동 중... ({current_page_num} -> {current_page_num + 1})"
            )
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
                time.sleep(random.uniform(0.4, 0.5))

                next_page_number = current_page_num + 1
                next_block_first_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            f"//button[.//span[text()='{next_page_number}']]",
                        )
                    )
                )
                driver.execute_script("arguments[0].click();", next_block_first_btn)
                time.sleep(random.uniform(0.25, 0.3))

                current_page_num = next_page_number
                continue
            except:
                print("   -> 다음 페이지(화살표 또는 새 블록)가 없습니다.")
                break

        else:
            next_num = current_page_num + 1
            try:
                next_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, f"//button[.//span[text()='{next_num}']]")
                    )
                )
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", next_btn
                )
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(random.uniform(0.25, 0.3))
                current_page_num += 1
            except:
                print("   -> 다음 페이지 번호가 없습니다.")
                break

    result_data["reviews"] = {
        "total_count": len(temp_reviews_list),
        "text_count": collected_count,
        "data": temp_reviews_list,
    }

    return result_data
