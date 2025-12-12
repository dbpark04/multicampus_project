import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import random
import pandas as pd


def get_coupang_product_info(url):
    # 1. 브라우저 설정
    options = uc.ChromeOptions()
    # options.add_argument('--headless') # 브라우저 안 띄우고 하려면 주석 해제

    driver = uc.Chrome(options=options, use_subprocess=True)

    product_data = {}

    try:
        print(f"🚀 상품 페이지 접속 중...: {url}")
        driver.get(url)

        # 2. 페이지 로딩 대기 (기본 정보는 금방 뜹니다)
        time.sleep(random.uniform(2, 4))

        # 3. HTML 파싱
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # --- 데이터 추출 시작 ---

        #! A. 상품명
        product_name = "상품명 수집 실패"
        try:
            # 1. 가장 일반적인 상품명 위치 (h2 태그)
            product_name = soup.select_one("h2.prod-buy-header__title").text.strip()
        except:
            try:
                # 2. 캡처해주신 span.twc-font-bold 요소 내부 텍스트 찾기
                product_name = soup.select_one("span.twc-font-bold").text.strip()
            except:
                try:
                    # 3. HTML <title> 태그에서 전체 텍스트를 가져온 후 불필요한 부분 제거
                    title_full = soup.select_one("title").text.strip()
                    product_name = title_full.split(" | 쿠팡")[0].split(" - ")[0]
                except:
                    # 최종 실패 시
                    product_name = "상품명 수집 실패"

        #! B. 가격 (판매가)
        price_selector = "div.option-table-list__option--selected div.option-table-list__option-price"
        price = "가격 수집 실패"
        try:
            price_tag = soup.select_one(price_selector)

            # 텍스트 추출: '13,800원' 또는 '13,800원 절약' 같은 텍스트에서 숫자만 남김
            price_text = price_tag.text.strip()
            # '원' 제거, 쉼표(,) 제거, 띄어쓰기 기준으로 맨 앞의 가격만 추출 (가장 확실함)
            price = price_text.replace("원", "").replace(",", "").split()[0].strip()
        except:
            price = "가격 수집 실패"

        #! C. 배송 정보 (로켓배송 여부 등)
        # 로켓 프레시 이미지: https://image.coupangcdn.com/image/mobile_app/v3/brandsdp/loyalty/pc/rocket-fresh@2x.png
        # 하늘색 로켓(쿠팡) 배송 이미지: https://image.coupangcdn.com/image/rds/delivery_badge_ext/badge_1998ab96bf7.png
        # 남색 로켓(파트너사) 배송 이미지: https://image.coupangcdn.com/image/rds/delivery_badge_ext/badge_1998ab98cb6.png
        # 판매자 로켓 배송 이미지: https://image.coupangcdn.com/image/rds/delivery_badge_ext/badge_199559e56f7.png
        delivery_type = "일반배송"  # 기본값
        try:
            # 1. 배송 배지 이미지 태그 찾기
            badge_img = soup.select_one("div.price-badge img")

            if badge_img:
                src = badge_img.get("src", "")

                # 2. 이미지 주소로 배송 타입 분기
                if "rocket-fresh@2x.png" in src:
                    delivery_type = "로켓프레시"
                elif "badge_1998ab96bf7.png" in src:
                    delivery_type = "로켓배송(쿠팡)"  # 하늘색
                elif "badge_1998ab98cb6.png" in src:
                    delivery_type = "로켓배송(파트너사)"  # 남색
                elif "badge_199559e56f7.png" in src:
                    delivery_type = "판매자 로켓"
                else:
                    delivery_type = (
                        "일반배송"  # 배지는 있으나 위 4개 케이스가 아닌 경우
                    )
            else:
                delivery_type = "일반배송"  # 배지 이미지가 아예 없는 경우

        except Exception as e:
            print(f"배송 정보 확인 중 에러: {e}")
            delivery_type = "일반배송"
        #! D. 리뷰 수 (상단 요약 정보)
        try:
            review_count_text = soup.select_one("span.rating-count-txt").text.strip()
            rating_count = review_count_text.split("개")[0].strip()
        except:
            rating_count = "0개"

        # 결과 저장
        product_data = {
            "상품명": product_name,
            "가격": price,
            "배송유형": delivery_type,
            "총_리뷰수": rating_count,
            "상품_링크": url,
        }

        print("\n 수집 성공!")
        print(f"상품명: {product_name}")
        print(f"가격: {price}")
        print(f"배송: {delivery_type}")
        print(f"리뷰수: {rating_count}")

    except Exception as e:
        print(f"에러 발생: {e}")

    finally:
        driver.quit()
        return product_data


# --- 실행 ---
target_url = "https://www.coupang.com/vp/products/5611991510?vendorItemId=92083385400"

data = get_coupang_product_info(target_url)

# 데이터가 있으면 엑셀/CSV 저장 (필요시)
if data:
    df = pd.DataFrame([data])
    df.to_csv("coupang_product_info.csv", index=False, encoding="utf-8-sig")
    print("\n📁 coupang_product_info.csv 저장 완료")
