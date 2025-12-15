import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random


def get_product_reviews(driver, url, rank_num, target_review_count=100):
    result_data = {
        "product_info": {},
        "reviews": {},
    }

    try:
        print(f"[Reviewer] 상품 페이지 접속: {url}")
        driver.get(url)
        time.sleep(random.uniform(3, 5))

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # 1. Product ID (순번으로 대체)
        product_id = str(rank_num)

        # 2. 상품명
        product_name = "Unknown"
        try:
            product_name = soup.select_one("span.twc-font-bold").text.strip()
        except:
            try:
                product_name = soup.select_one("h2.prod-buy-header__title").text.strip()
            except:
                pass

        # 3. 가격
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

        # 4. 배송 유형
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

        # 5. 총 리뷰 수
        total_reviews = "0"
        try:
            review_count_text = soup.select_one("span.rating-count-txt").text.strip()
            total_reviews = review_count_text.split("개")[0].replace(",", "").strip()
        except:
            pass

        # 6. 카테고리 추출
        category_str = ""
        try:
            crumb_links = soup.select("ul.breadcrumb li a")
            category_str = " > ".join(
                [link.text.strip() for link in crumb_links if link.text.strip()]
            )
        except:
            pass

        result_data["product_info"] = {
            "product_id": product_id,  # 1, 2, 3... 순서
            "category_path": category_str,
            "product_name": product_name,
            "price": price,
            "delivery_type": delivery_type,
            "total_reviews": total_reviews,
            "product_url": url,
        }

        print(f"   -> 상품ID: {product_id} / 상품명: {product_name}")
        print(
            f"   -> 가격: {price}원 / 배송: {delivery_type} / 총리뷰: {total_reviews}"
        )

        # --- 리뷰 수집 로직 ---
        temp_reviews_list = []

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
        collected_count = 0

        while collected_count < target_review_count:
            curr_soup = BeautifulSoup(driver.page_source, "html.parser")
            review_articles = curr_soup.select("article.twc-border-bluegray-200")

            if not review_articles:
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
                    date = date_div.text.strip() if date_div else ""

                    title_div = article.select_one(
                        "div.twc-font-bold.twc-text-bluegray-900"
                    )
                    title = title_div.text.strip() if title_div else ""

                    content_span = article.select_one("span.twc-bg-white")
                    content = content_span.text.strip() if content_span else ""

                    has_image = False
                    img_container = article.select_one(
                        "div.twc-overflow-x-auto.twc-scrollbar-hidden"
                    )
                    if img_container and img_container.select_one("img"):
                        has_image = True

                    review_obj = {
                        "id": collected_count + 1,
                        "date": date,
                        "rating": rating,
                        "has_image": has_image,
                        "title": title,
                        "content": content,
                        "full_text": f"{title}{content}",
                    }
                    temp_reviews_list.append(review_obj)
                    collected_count += 1
                except:
                    continue

            print(
                f"   -> {current_page_num}페이지 완료 ({collected_count}/{target_review_count})"
            )

            if collected_count >= target_review_count:
                break

            next_num = current_page_num + 1
            try:
                if current_page_num % 10 == 0:
                    next_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "//div[contains(@class, 'twc-mt-[24px]')]//button[last()]",
                            )
                        )
                    )
                else:
                    next_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, f"//button[.//span[text()='{next_num}']]")
                        )
                    )

                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", next_btn
                )
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(random.uniform(1.2, 1.5))
                current_page_num += 1
            except:
                break

        result_data["reviews"] = {
            "count": len(temp_reviews_list),
            "data": temp_reviews_list,
        }

    except Exception as e:
        print(f"[Reviewer] 에러 발생: {e}")

    return result_data
