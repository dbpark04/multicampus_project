"""
BERT 기반 벡터화 모듈
"""

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List


class BERTVectorizer:
    """
    한국어 BERT 모델을 사용한 벡터화 클래스
    """

    def __init__(self, model_name: str = "klue/bert-base"):
        """
        Args:
            model_name: 사용할 BERT 모델 이름
                - "klue/bert-base": KLUE BERT (추천)
                - "beomi/kcbert-base": KcBERT
                - "monologg/kobert": KoBERT
        """
        print(f"BERT 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # 평가 모드

        # GPU 사용 가능하면 GPU 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"BERT 모델 로딩 완료 (device: {self.device})")

    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        단일 텍스트를 BERT 벡터로 변환

        Args:
            text: 입력 텍스트
            max_length: 최대 토큰 길이

        Returns:
            768차원 벡터 (BERT base 모델의 경우)
        """
        if not text or not text.strip():
            # 빈 텍스트는 zero 벡터 반환
            return np.zeros(768)

        # 토큰화 및 인코딩
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        # GPU로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 모델 추론
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # [CLS] 토큰의 벡터 사용 (문장 전체 표현)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return cls_embedding

    def encode_batch(
        self, texts: List[str], max_length: int = 512, batch_size: int = 128
    ) -> List[np.ndarray]:
        """
        여러 텍스트를 배치로 벡터화 (효율적)

        Args:
            texts: 텍스트 리스트
            max_length: 최대 토큰 길이
            batch_size: 배치 크기

        Returns:
            벡터 리스트
        """
        vectors = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # 빈 텍스트 처리
            processed_texts = [t if t and t.strip() else " " for t in batch_texts]

            # 토큰화
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )

            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 모델 추론
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)

            # [CLS] 토큰 벡터 추출
            batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            vectors.extend(batch_vectors)

        return vectors

    def get_vector_size(self) -> int:
        """벡터 차원 반환"""
        return self.model.config.hidden_size


# 모델별 인스턴스 캐시 (메모리 효율성)
_bert_vectorizer_instances = {}


def get_bert_vectorizer(model_name: str = "klue/bert-base") -> BERTVectorizer:
    """
    BERT Vectorizer 인스턴스 반환 (모델별로 캐시)
    """
    global _bert_vectorizer_instances

    if model_name not in _bert_vectorizer_instances:
        _bert_vectorizer_instances[model_name] = BERTVectorizer(model_name)

    return _bert_vectorizer_instances[model_name]
