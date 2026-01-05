import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(page_title="Streamlit 연동 테스트", layout="wide")

# 제목 및 설명
st.title("Streamlit Cloud 연동 테스트")
st.write("레포지토리 연동이 성공하면 이 화면이 브라우저에 표시됩니다.")

# 1. 인터랙티브 위젯 테스트
user_input = st.text_input("이름을 입력하세요", "사용자")
st.write(f"반갑습니다, {user_input}님. 현재 시스템이 정상 작동 중입니다.")

# 2. 데이터 시각화 테스트 (메서드 체이닝 활용)
st.subheader("샘플 데이터 차트 출력 테스트")

# 랜덤 데이터를 생성하고 즉시 라인 차트로 시각화함
st.line_chart(
    pd.DataFrame(np.random.randn(20, 3), columns=["리뷰 수", "평점 점수", "감성 지수"])
    .assign(index=np.arange(20))
    .set_index("index")
)

# 3. 데이터프레임 출력 테스트
st.subheader("데이터프레임 렌더링 테스트")
st.dataframe(
    pd.DataFrame(
        {
            "카테고리": ["선스틱", "선크림", "선쿠션"],
            "상품 수": [114, 145, 118],
            "분석 상태": ["완료", "진행중", "대기"],
        }
    )
)
