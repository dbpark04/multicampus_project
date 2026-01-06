"""
로지스틱 회귀 모델 학습 및 평가
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib
import os

# 데이터 경로
PARQUET_PATH = "./data/processed_data/integrated_reviews_detail.parquet"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_sentiment.pkl")


def load_and_prepare_data():
    """
    Parquet 파일에서 데이터 로드 및 전처리
    - label이 0 또는 1인 데이터만 사용
    - word2vec 벡터를 feature로 사용
    """
    print("데이터 로딩 중...")
    df = pd.read_parquet(PARQUET_PATH)

    # label이 0(부정) 또는 1(긍정)인 데이터만 필터링
    df = df[df["label"].isin([0, 1])].copy()

    # word2vec 벡터가 있는 데이터만 사용
    df = df[df["word2vec"].notna()].copy()

    print(f"전체 데이터: {len(df):,}개")
    print(f"긍정 리뷰: {(df['label'] == 1).sum():,}개")
    print(f"부정 리뷰: {(df['label'] == 0).sum():,}개")

    # word2vec 벡터를 numpy array로 변환
    X = np.array(df["word2vec"].tolist())
    y = df["label"].values

    return X, y, df


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    로지스틱 회귀 모델 학습 및 평가
    """
    print("\n로지스틱 회귀 모델 학습 중...")

    # 모델 생성 및 학습
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",  # 클래스 불균형 처리
        solver="lbfgs",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("모델 학습 완료!")

    # 학습 데이터 평가
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n[Train 성능]")
    print(f"정확도: {train_acc:.4f}")

    # 테스트 데이터 평가
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average="binary")
    test_recall = recall_score(y_test, y_pred, average="binary")
    test_f1 = f1_score(y_test, y_pred, average="binary")

    print(f"\n[Test 성능]")
    print(f"정확도: {test_acc:.4f}")
    print(f"정밀도: {test_precision:.4f}")
    print(f"재현율: {test_recall:.4f}")
    print(f"F1 점수: {test_f1:.4f}")

    # 상세 분류 리포트
    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred, target_names=["부정(0)", "긍정(1)"]))

    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print("\n[혼동 행렬]")
    print(f"                예측 부정  예측 긍정")
    print(f"실제 부정        {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"실제 긍정        {cm[1][0]:6d}    {cm[1][1]:6d}")

    return model


def save_model(model):
    """모델 저장"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")


def predict_sample(model, X_test, y_test, df_test, n_samples=5):
    """
    샘플 데이터로 긍정/부정 확률 예측

    Args:
        model: 학습된 모델
        X_test: 테스트 벡터
        y_test: 테스트 레이블
        df_test: 테스트 DataFrame (원본 텍스트 확인용)
        n_samples: 예측할 샘플 개수
    """
    print("\n" + "=" * 60)
    print(f"샘플 {n_samples}개 예측 결과")
    print("=" * 60)

    # 랜덤 샘플 선택
    sample_indices = np.random.choice(len(X_test), size=n_samples, replace=False)

    for i, idx in enumerate(sample_indices, 1):
        # 예측
        sample_vector = X_test[idx].reshape(1, -1)
        prediction = model.predict(sample_vector)[0]
        probabilities = model.predict_proba(sample_vector)[0]

        # 실제 정보
        actual_label = y_test[idx]
        review_info = df_test.iloc[idx]

        print(f"\n[샘플 {i}]")
        print(f"상품 ID: {review_info['product_id']}")
        print(f"리뷰 텍스트: {review_info['full_text'][:100]}...")
        print(f"실제 평점: {review_info['score']}점")
        print(f"실제 레이블: {'긍정(1)' if actual_label == 1 else '부정(0)'}")
        print(f"예측 레이블: {'긍정(1)' if prediction == 1 else '부정(0)'}")
        print(f"예측 확률:")
        print(f"  - 부정(0): {probabilities[0]:.2%}")
        print(f"  - 긍정(1): {probabilities[1]:.2%}")
        print(f"결과: {'정답' if prediction == actual_label else '오답'}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("로지스틱 회귀 감성 분석 모델 학습")
    print("=" * 60)

    # 1. 데이터 로드 및 전처리
    X, y, df = load_and_prepare_data()

    # 2. Train/Test 분할 (8:2, stratify로 클래스 비율 유지)
    print("\n데이터 분할 중 (Train:Test = 8:2)...")
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train 데이터: {len(X_train):,}개")
    print(f"Test 데이터: {len(X_test):,}개")
    print(
        f"Train 긍정 비율: {(y_train == 1).sum() / len(y_train):.2%}, "
        f"부정 비율: {(y_train == 0).sum() / len(y_train):.2%}"
    )
    print(
        f"Test 긍정 비율: {(y_test == 1).sum() / len(y_test):.2%}, "
        f"부정 비율: {(y_test == 0).sum() / len(y_test):.2%}"
    )

    # 3. 모델 학습 및 평가
    model = train_logistic_regression(X_train, y_train, X_test, y_test)

    # 4. 모델 저장
    save_model(model)

    # 5. 샘플 예측 데모
    predict_sample(model, X_test, y_test, df_test, n_samples=5)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
