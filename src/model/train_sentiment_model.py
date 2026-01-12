"""
감성 분석 모델 학습 스크립트

수정 사항:
  - ZeroDivisionError 방지 로직 추가 (데이터 0개일 때 중단)
  - 모델 저장 파일명 변경: logistic_regression_sentiment.joblib
  - NaN 값 처리를 위한 pd.notna() 도입
"""

import os
import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
    KOREAN_FONT = "Malgun Gothic"
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
    KOREAN_FONT = "AppleGothic"
else:  # 리눅스 (예: Google Colab, Ubuntu)
    plt.rc("font", family="NanumGothic")
    KOREAN_FONT = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지


def load_review_data(partitioned_reviews_dir):
    print("======================================================================")
    print(f"전처리된 리뷰 데이터 로드 중: {partitioned_reviews_dir}")

    # Hive 파티셔닝: category=*/data.parquet 패턴
    parquet_files = glob.glob(
        os.path.join(partitioned_reviews_dir, "category=*", "data.parquet")
    )

    if not parquet_files:
        print("[오류] Parquet 파일을 찾을 수 없습니다.")
        return []

    all_reviews = []
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            category = os.path.basename(os.path.dirname(file_path)).replace(
                "category=", ""
            )
            print(f"  - {category}: {len(df):,}개 리뷰")
            all_reviews.extend(df.to_dict("records"))
        except Exception as e:
            print(f"파일 로드 오류: {file_path} - {e}")

    print(f"✓ 총 {len(all_reviews):,}개 리뷰 로드 완료")
    return all_reviews


def prepare_training_data(reviews):
    print("\n학습 데이터 준비 중...")
    X_word2vec = []
    X_bert = []
    y_word2vec = []
    y_bert = []

    # 데이터 샘플로 구조 확인 (상세하게)
    if reviews:
        sample = reviews[0]
        print(f"\n[데이터 구조 확인]")
        print(f"  - 전체 키: {list(sample.keys())}")
        print(f"  - label 존재: {'label' in sample}, 값: {sample.get('label')}")
        print(f"  - word2vec 존재: {'word2vec' in sample}")
        print(f"  - bert 존재: {'bert' in sample}")

        w2v_val = sample.get("word2vec")
        bert_val = sample.get("bert")

        if w2v_val is not None:
            print(
                f"  - word2vec 타입: {type(w2v_val)}, 길이: {len(w2v_val) if hasattr(w2v_val, '__len__') else 'N/A'}"
            )
        if bert_val is not None:
            print(
                f"  - bert 타입: {type(bert_val)}, 길이: {len(bert_val) if hasattr(bert_val, '__len__') else 'N/A'}"
            )

        # 처음 100개 샘플에서 통계
        label_count = sum(1 for r in reviews[:100] if pd.notna(r.get("label")))
        w2v_count = sum(1 for r in reviews[:100] if r.get("word2vec") is not None)
        bert_count = sum(1 for r in reviews[:100] if r.get("bert") is not None)
        print(f"\n[처음 100개 샘플 확인]")
        print(f"  - label 있는 리뷰: {label_count}개")
        print(f"  - word2vec 있는 리뷰: {w2v_count}개")
        print(f"  - bert 있는 리뷰: {bert_count}개")

    # 각 벡터 타입별로 데이터 수집
    for review in reviews:
        w2v = review.get("word2vec")
        bert = review.get("bert")
        label = review.get("label")

        # label이 유효한지 확인
        if not pd.notna(label):
            continue

        # word2vec 벡터 수집
        if w2v is not None and isinstance(w2v, (list, np.ndarray)) and len(w2v) > 0:
            X_word2vec.append(np.array(w2v))
            y_word2vec.append(int(label))

        # bert 벡터 수집
        if bert is not None and isinstance(bert, (list, np.ndarray)) and len(bert) > 0:
            X_bert.append(np.array(bert))
            y_bert.append(int(label))

    # 결과 출력
    w2v_count = len(y_word2vec)
    bert_count = len(y_bert)

    print(f"\n✓ Word2Vec 데이터: {w2v_count:,}개")
    if w2v_count > 0:
        pos = sum(y_word2vec)
        neg = w2v_count - pos
        print(f"  - 긍정: {pos:,}개 ({pos/w2v_count*100:.1f}%)")
        print(f"  - 부정: {neg:,}개 ({neg/w2v_count*100:.1f}%)")
        print(f"  - 벡터 차원: {len(X_word2vec[0])}")

    print(f"\n✓ BERT 데이터: {bert_count:,}개")
    if bert_count > 0:
        pos = sum(y_bert)
        neg = bert_count - pos
        print(f"  - 긍정: {pos:,}개 ({pos/bert_count*100:.1f}%)")
        print(f"  - 부정: {neg:,}개 ({neg/bert_count*100:.1f}%)")
        print(f"  - 벡터 차원: {len(X_bert[0])}")

    return (
        np.array(X_word2vec) if X_word2vec else np.array([]),
        np.array(y_word2vec) if y_word2vec else np.array([]),
        np.array(X_bert) if X_bert else np.array([]),
        np.array(y_bert) if y_bert else np.array([]),
    )


def train_model(X_train, y_train):
    print("\n모델 학습 중...")
    # 클래스 불균형 대응을 위해 class_weight='balanced' 설정
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    print("✓ 모델 학습 완료")
    return model


def evaluate_model(
    model, X_test, y_test, output_dir, model_name="model", X_train=None, y_train=None
):
    print("\n모델 평가 중...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 긍정 클래스 확률

    # ============ 기본 메트릭 ============
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n" + "=" * 70)
    print("기본 성능 메트릭")
    print("=" * 70)
    print(f"정확도 (Accuracy):     {accuracy:.4f}")
    print(f"F1 Score:              {f1:.4f}")
    print(f"Matthews Corr Coef:    {mcc:.4f}")

    # ============ 분류 리포트 ============
    print("\n" + "=" * 70)
    print("분류 리포트 (Classification Report)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["부정(0)", "긍정(1)"]))

    # ============ 혼동 행렬 상세 분석 ============
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 70)
    print("혼동 행렬 상세 분석")
    print("=" * 70)
    print(f"True Negative (TN):    {tn:,}개 - 부정을 부정으로 맞춤")
    print(f"False Positive (FP):   {fp:,}개 - 부정을 긍정으로 잘못 예측")
    print(f"False Negative (FN):   {fn:,}개 - 긍정을 부정으로 잘못 예측")
    print(f"True Positive (TP):    {tp:,}개 - 긍정을 긍정으로 맞춤")
    print(f"\nSpecificity (특이도):  {tn/(tn+fp):.4f} - 부정 클래스 탐지 성능")
    print(f"Sensitivity (민감도):  {tp/(tp+fn):.4f} - 긍정 클래스 탐지 성능")

    # ============ ROC Curve & AUC ============
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print("\n" + "=" * 70)
    print("ROC & AUC")
    print("=" * 70)
    print(f"AUC (Area Under ROC): {roc_auc:.4f}")

    # ============ Precision-Recall Curve ============
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    print(f"Average Precision:    {avg_precision:.4f}")

    # ============ K-Fold 교차 검증 ============
    if X_train is not None and y_train is not None:
        print("\n" + "=" * 70)
        print("K-Fold 교차 검증 (5-Fold)")
        print("=" * 70)

        # 전체 데이터로 K-Fold 수행 (train + test 합쳐서)
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model,
            X_full,
            y_full,
            cv=skf,
            scoring="accuracy",
        )
        cv_f1_scores = cross_val_score(
            model,
            X_full,
            y_full,
            cv=skf,
            scoring="f1",
        )

        print(f"\nAccuracy per fold: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"평균 Accuracy:     {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"\nF1 Score per fold: {[f'{score:.4f}' for score in cv_f1_scores]}")
        print(
            f"평균 F1 Score:     {cv_f1_scores.mean():.4f} (±{cv_f1_scores.std():.4f})"
        )

    # ============ 시각화 ============
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["부정", "긍정"],
        yticklabels=["부정", "긍정"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("실제", fontsize=12)
    axes[0, 0].set_xlabel("예측", fontsize=12)

    # 2. ROC Curve
    axes[0, 1].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})"
    )
    axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel("False Positive Rate", fontsize=12)
    axes[0, 1].set_ylabel("True Positive Rate", fontsize=12)
    axes[0, 1].set_title("ROC Curve", fontsize=14, fontweight="bold")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    # 3. Precision-Recall Curve
    axes[1, 0].plot(
        recall, precision, color="blue", lw=2, label=f"PR (AP = {avg_precision:.3f})"
    )
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel("Recall", fontsize=12)
    axes[1, 0].set_ylabel("Precision", fontsize=12)
    axes[1, 0].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(alpha=0.3)

    # 4. 확률 분포 히스토그램
    axes[1, 1].hist(
        y_proba[y_test == 0], bins=50, alpha=0.5, label="부정(실제)", color="red"
    )
    axes[1, 1].hist(
        y_proba[y_test == 1], bins=50, alpha=0.5, label="긍정(실제)", color="green"
    )
    axes[1, 1].set_xlabel("예측 확률 (긍정 클래스)", fontsize=12)
    axes[1, 1].set_ylabel("빈도", fontsize=12)
    axes[1, 1].set_title("예측 확률 분포", fontsize=14, fontweight="bold")
    axes[1, 1].legend(loc="upper center")
    axes[1, 1].grid(alpha=0.3)

    # 5. K-Fold 교차 검증 결과 시각화
    if X_train is not None and y_train is not None:
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_full, y_full, cv=skf, scoring="accuracy")

        fold_numbers = list(range(1, len(cv_scores) + 1))
        axes[0, 2].bar(fold_numbers, cv_scores, color="skyblue", edgecolor="navy")
        axes[0, 2].axhline(
            y=cv_scores.mean(),
            color="red",
            linestyle="--",
            label=f"평균: {cv_scores.mean():.4f}",
        )
        axes[0, 2].set_xlabel("Fold", fontsize=12)
        axes[0, 2].set_ylabel("Accuracy", fontsize=12)
        axes[0, 2].set_title(
            "K-Fold 교차 검증 (Accuracy)", fontsize=14, fontweight="bold"
        )
        axes[0, 2].set_ylim([0.0, 1.0])
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3, axis="y")
    else:
        axes[0, 2].text(
            0.5, 0.5, "K-Fold 데이터 없음", ha="center", va="center", fontsize=14
        )
        axes[0, 2].set_title("K-Fold 교차 검증", fontsize=14, fontweight="bold")

    # 6. 빈 공간에 메트릭 요약 표시
    metrics_text = f"""성능 요약
    
정확도: {accuracy:.4f}
F1 Score: {f1:.4f}
MCC: {mcc:.4f}
AUC: {roc_auc:.4f}
Avg Precision: {avg_precision:.4f}

TN: {tn:,}  FP: {fp:,}
FN: {fn:,}  TP: {tp:,}

Specificity: {tn/(tn+fp):.4f}
Sensitivity: {tp/(tp+fn):.4f}"""

    axes[1, 2].text(
        0.1,
        0.5,
        metrics_text,
        fontsize=11,
        verticalalignment="center",
        family=KOREAN_FONT,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    axes[1, 2].axis("off")

    plt.tight_layout()

    # 파일명에 모델 이름 포함
    eval_path = os.path.join(output_dir, f"model_evaluation_{model_name}.png")
    plt.savefig(eval_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ 평가 결과 시각화 저장: {eval_path}")

    # ============ 임계값 분석 ============
    print("\n" + "=" * 70)
    print("임계값별 성능 (상위 5개)")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 70)

    # 다양한 임계값에서 성능 계산
    threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in threshold_candidates:
        y_pred_custom = (y_proba >= threshold).astype(int)
        prec = precision_score_custom(y_test, y_pred_custom)
        rec = recall_score_custom(y_test, y_pred_custom)
        f1_custom = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{threshold:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1_custom:>10.4f}")


def precision_score_custom(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score_custom(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def main():
    print("=" * 70)
    print("감성 분석 모델 학습 (Logistic Regression)")
    print("=" * 70)

    # 경로 설정
    PROCESSED_DATA_DIR = "./data/processed_data"
    PARTITIONED_REVIEWS_DIR = os.path.join(PROCESSED_DATA_DIR, "partitioned_reviews")
    MODEL_OUTPUT_DIR = "./models"
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드
    reviews = load_review_data(PARTITIONED_REVIEWS_DIR)
    if not reviews:
        print("[중단] 로드된 리뷰 데이터가 없습니다.")
        return

    # 2. 학습 데이터 준비 (word2vec과 bert 각각)
    X_w2v, y_w2v, X_bert, y_bert = prepare_training_data(reviews)

    has_word2vec = X_w2v.size > 0
    has_bert = X_bert.size > 0

    if not has_word2vec and not has_bert:
        print("\n[중단] 학습 가능한 데이터가 없습니다.")
        return

    # 3. Word2Vec 모델 학습
    if has_word2vec:
        print("\n" + "=" * 70)
        print("Word2Vec 기반 모델 학습")
        print("=" * 70)

        print("\n데이터 분할 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_w2v, y_w2v, test_size=0.2, random_state=42, stratify=y_w2v
        )
        print(f"✓ 훈련: {len(X_train):,}개 / 테스트: {len(X_test):,}개")

        model_w2v = train_model(X_train, y_train)
        evaluate_model(
            model_w2v,
            X_test,
            y_test,
            MODEL_OUTPUT_DIR,
            model_name="word2vec",
            X_train=X_train,
            y_train=y_train,
        )

        model_path = os.path.join(
            MODEL_OUTPUT_DIR, "logistic_regression_sentiment_word2vec.joblib"
        )
        joblib.dump(model_w2v, model_path)
        print(f"\n✓ 모델 저장 완료: {model_path}")

    # 4. BERT 모델 학습
    if has_bert:
        print("\n" + "=" * 70)
        print("BERT 기반 모델 학습")
        print("=" * 70)

        print("\n데이터 분할 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_bert, y_bert, test_size=0.2, random_state=42, stratify=y_bert
        )
        print(f"✓ 훈련: {len(X_train):,}개 / 테스트: {len(X_test):,}개")

        model_bert = train_model(X_train, y_train)
        evaluate_model(
            model_bert,
            X_test,
            y_test,
            MODEL_OUTPUT_DIR,
            model_name="bert",
            X_train=X_train,
            y_train=y_train,
        )

        model_path = os.path.join(
            MODEL_OUTPUT_DIR, "logistic_regression_sentiment_bert.joblib"
        )
        joblib.dump(model_bert, model_path)
        print(f"\n✓ 모델 저장 완료: {model_path}")

    print("\n" + "=" * 70)
    print("학습 완료!")
    if has_word2vec:
        print("  - Word2Vec 모델: logistic_regression_sentiment_word2vec.joblib")
    if has_bert:
        print("  - BERT 모델: logistic_regression_sentiment_bert.joblib")
    print("=" * 70)


if __name__ == "__main__":
    main()
