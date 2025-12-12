import pandas as pd

# 1. 파일 경로를 지정합니다. (예시 파일명)
file_path = "coupang_product_info.csv"

try:
    # 2. CSV 파일 로드
    df = pd.read_csv(file_path)

    # 3. 데이터프레임의 상위 5행 출력 (기본)
    print("--- 상위 5행 ---")
    print(df.head())

    # 4. 전체 열(Column)과 데이터 타입을 확인
    print("\n--- 정보 요약 ---")
    df.info()

    # 5. (선택 사항) 터미널에서 모든 열을 다 보고 싶을 때 설정
    # pd.set_option('display.max_columns', None) # 모든 열을 표시
    # pd.set_option('display.width', 1000)      # 출력 너비를 넓게 설정

except FileNotFoundError:
    print(f"오류: {file_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"데이터를 로드하는 중 오류 발생: {e}")
