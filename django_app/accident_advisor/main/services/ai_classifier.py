"""
교통사고 챗봇 질문 분류 서비스 및 RAG 처리
파인튜닝된 GPT-3.5-turbo 모델 사용 + OpenAI 임베딩 통일
"""

import openai
import os
import logging
from typing import Dict, Any, List
from django.conf import settings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from ..utils.vector_db import get_vector_db_manager

# 로거 설정
logger = logging.getLogger(__name__)

class TrafficAccidentClassifier:
    """
    파인튜닝된 GPT-3.5-turbo를 사용한 질문 분류기 + RAG 처리기
    
    분류 카테고리:
    - accident: 교통사고 상황 분석
    - precedent: 판례 검색
    - law: 도로교통법 조회
    - term: 용어 설명
    - general: 일반 질문
    """
    
    # 허용되는 분류 카테고리
    VALID_CATEGORIES = {'accident', 'precedent', 'law', 'term', 'general'}
    
    # 폴백 분류 (키워드 기반)
    FALLBACK_KEYWORDS = {
        'accident': [
            '사고', '충돌', '접촉', '과실', '비율', '좌회전', '직진', '교차로',
            '신호위반', '주차장', '후진', '차로변경', '추돌', '측면충돌'
        ],
        'precedent': [
            '판례', '대법원', '고등법원', '지방법원', '판결', '사건번호',
            '판단', '요지', '법원', '소송', '재판'
        ],
        'law': [
            '도로교통법', '법률', '조문', '제', '조', '항', '규정', '위반',
            '처벌', '범칙금', '벌점', '법적', '규칙'
        ],
        'term': [
            '정의', '의미', '뜻', '용어', '설명', '개념', '무엇', '어떤',
            '차로', '도로', '차량', '운전자', '보행자'
        ]
    }
    
    # JSON 파일 KEY 값 정의 (UI.py와 동일)
    METADATA_KEY = {
        'PRECEDENT': {
            'COURT': "court",
            'CASE_ID': "case_id",
            'CONTENT': "content",
        },
        'LOAD_TRAFFIC_LAW': {
            'ARTICLE_TITLE': "article_title",
            'ARTICLE_NUMBER': "article_number", 
            'PARAGRAPH': "paragraph",
            'CONTENT': "content",
        }
    }
    
    def __init__(self):
        """분류기 및 RAG 시스템 초기화"""
        try:
            # OpenAI 클라이언트 초기화
            self.client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # 파인튜닝된 모델 ID
            self.model_id = os.getenv('FINETUNED_MODEL_ID')
            
            if not self.model_id:
                logger.warning("FINETUNED_MODEL_ID가 설정되지 않았습니다. 폴백 모드로 동작합니다.")
            
            # GPT 모델 초기화 (RAG용)
            self.gpt_4o_model = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0,
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # VectorDB 매니저 초기화
            self.vector_db_manager = get_vector_db_manager()
            
            logger.info(f"TrafficAccidentClassifier 초기화 완료 - 모델: {self.model_id}")
            
        except Exception as e:
            logger.error(f"분류기 초기화 실패: {str(e)}")
            raise
    
    def classify_query(self, user_input: str) -> str:
        """
        사용자 질문을 분류합니다.
        
        Args:
            user_input (str): 사용자 입력 텍스트
            
        Returns:
            str: 분류 결과 ('accident', 'precedent', 'law', 'term', 'general')
        """
        import time
        start_time = time.time()
        
        if not user_input or not user_input.strip():
            return 'general'
        
        # 1차: 파인튜닝된 모델로 분류 시도
        try:
            logger.info(f"파인튜닝 모델 분류 시작: '{user_input[:30]}...'")
            api_start = time.time()
            
            category = self._classify_with_finetuned_model(user_input.strip())
            
            api_time = time.time() - api_start
            logger.info(f"API 호출 시간: {api_time:.2f}초")
            
            if self._is_valid_category(category):
                total_time = time.time() - start_time
                logger.info(f"파인튜닝 모델 분류 성공: '{user_input[:30]}...' → {category} (총 {total_time:.2f}초)")
                return category
        except Exception as e:
            logger.warning(f"파인튜닝 모델 분류 실패: {str(e)}")
        
        # 2차: 키워드 기반 폴백 분류
        fallback_start = time.time()
        category = self._fallback_classify(user_input.strip())
        fallback_time = time.time() - fallback_start
        
        total_time = time.time() - start_time
        logger.info(f"폴백 분류 사용: '{user_input[:30]}...' → {category} (폴백: {fallback_time:.2f}초, 총: {total_time:.2f}초)")
        return category
    
    def process_precedent(self, user_input: str) -> str:
        """
        Enhanced 판례 검색 (판례번호 정확성 검증 포함)
        
        Args:
            user_input (str): 사용자 질문
            
        Returns:
            str: 판례 검색 결과
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Enhanced 판례 검색 시작: '{user_input[:30]}...'")
            
            # Enhanced Precedent Processor 사용
            from .enhanced_precedent_processor import EnhancedPrecedentProcessor
            
            processor = EnhancedPrecedentProcessor(
                vector_db_manager=self.vector_db_manager,
                gpt_model=self.gpt_4o_model
            )
            
            result = processor.process_precedent_query(user_input)
            
            total_time = time.time() - start_time
            logger.info(f"Enhanced 판례 검색 완료: '{user_input[:30]}...' (총: {total_time:.2f}초)")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced 판례 검색 중 오류: {str(e)}")
            return self._generate_precedent_fallback_response(user_input)
    
    def process_law(self, user_input: str) -> str:
        """
        도로교통법 조회 및 분석 (OpenAI 임베딩 통일 방식)
        
        Args:
            user_input (str): 사용자 질문
            
        Returns:
            str: 도로교통법 조회 결과
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"도로교통법 조회 시작: '{user_input[:30]}...'")
            
            # 1. VectorDB에서 traffic_law_rag 컬렉션 가져오기
            law_db = self.vector_db_manager.get_vector_db('traffic_law_rag')
            if not law_db:
                logger.error("traffic_law_rag VectorDB를 찾을 수 없습니다.")
                return self._generate_law_fallback_response(user_input)
            
            # 2. 메타데이터 필드 정의 (Self-Query Retriever용)
            metadata_field_info = [
                AttributeInfo(
                    name="article_title",
                    description="도로교통법 조문 전체 제목 (예: 제5조(신호 또는 지시에 따를 의무), 제25조(교차로 통행방법))",
                    type="string"
                ),
                AttributeInfo(
                    name="article_number",
                    description="도로교통법 조문 번호 (예: 제5조, 제25조, 제27조 등)",
                    type="string"
                ),
                AttributeInfo(
                    name="article_name",
                    description="도로교통법 조문명 (예: 신호 또는 지시에 따를 의무, 교차로 통행방법 등)",
                    type="string"
                ),
                AttributeInfo(
                    name="subsection_title",
                    description="조문 내 항 제목 (예: 제5조 1항, 제25조 2항 등)",
                    type="string"
                ),
                AttributeInfo(
                    name="keywords",
                    description="조문 관련 키워드 (예: 신호, 지시, 교통안전시설, 경찰공무원 등)",
                    type="string"
                )
            ]
            
            # 3. Self-Query Retriever 생성
            self_retriever = SelfQueryRetriever.from_llm(
                llm=self.gpt_4o_model,
                vectorstore=law_db,
                document_contents="도로교통법 조문 내용 및 해석 데이터",
                metadata_field_info=metadata_field_info,
                search_kwargs={"k": 5}  # 검색 결과 수 증가
            )
            
            # 4. 도로교통법 조회 및 설명 프롬프트 (사용자 친화적 포맷)
            prompt = PromptTemplate(
                input_variables=["question", "context"],
                template="""
너는 도로교통법을 쉽게 설명해주는 법률 전문가야.

아래 문서(context)를 참고하여 사용자의 질문에 대해 관련된 도로교통법 조문을 **쉽고 이해하기 쉬롭게** 설명해줘.

---
질문: {question}

검색된 도로교통법 문서:
{context}
---

출력 형식:

📚 **도로교통법 조회 결과**

**🔍 조회 내용**: [사용자 질문 요약]

**📋 관련 조문**:

**1️⃣ [조문 제목] - [조문번호]**
• **주요 내용**: [주요 내용을 1-2줄로 간단히]
• **상세 설명**: 
  - [주요 요점 1]
  - [주요 요점 2]
  - [주요 요점 3 (있는 경우만)]
• **위반 시 처벌**: [범칙금, 벌점 등 명시된 경우만]

**2️⃣ [조문 제목] - [조문번호]**
• **주요 내용**: [주요 내용을 1-2줄로 간단히]
• **상세 설명**: [주요 요점들을 질대별로 정리]
• **위반 시 처벌**: [명시된 경우만 표시]

*(최대 3-4개 조문만 표시)*

**💡 참고사항**:
- 구체적인 사고 상황에 따라 적용이 달라질 수 있습니다
- 정확한 분석을 위해서는 전문가 상담을 받아보세요

**조건**:
- 각 조문을 **명확히 구분**하여 표시하세요
- **대중이 이해하기 쉬운 언어**로 설명하세요
- **긴 내용은 1-2줄로 요약**하여 읽기 쉬롭게 하세요
- 문서에 없는 정보는 임의로 만들지 마세요
- 조문번호와 제목을 **정확히** 표시하세요
- 전문 용어는 쉽게 풀어서 설명하세요
- 사용자가 특정 조문을 물어봤다면, 해당 조문을 우선적으로 표시하세요

답변:
"""
            )
            
            # 5. QA 체인 구성 및 실행
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.gpt_4o_model,
                retriever=self_retriever,   # 유사도 검색
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )
            
            # 6. 검색 및 응답 생성
            retrieval_start = time.time()
            result = qa_chain.invoke({"query": user_input})
            retrieval_time = time.time() - retrieval_start
            
            total_time = time.time() - start_time
            logger.info(f"도로교통법 조회 완료: '{user_input[:30]}...' (검색: {retrieval_time:.2f}초, 총: {total_time:.2f}초)")
            
            return result['result']
            
        except Exception as e:
            logger.error(f"도로교통법 조회 중 오류: {str(e)}")
            return self._generate_law_fallback_response(user_input)
    
    def process_accident(self, user_input: str) -> str:
        """
        교통사고 과실비율 분석 (RAG 기반)
        Args:
            user_input (str): 사용자 질문
        Returns:
            str: 과실비율 분석 결과
        """
        import time
        start_time = time.time()
        try:
            logger.info(f"과실비율 분석 시작: '{user_input[:30]}...'")
            # 1. VectorDB에서 car_case 컬렉션 가져오기
            car_case_db = self.vector_db_manager.get_vector_db('car_case')
            if not car_case_db:
                logger.error("car_case VectorDB를 찾을 수 없습니다.")
                return self._generate_accident_fallback_response(user_input)
            # 2. 메타데이터 필드 정의 (Self-Query Retriever용)
            metadata_field_info = [
                AttributeInfo(
                    name="case_type",
                    description="사고 유형 (예: 교차로 좌회전 vs 직진, 주차장 후진 등)",
                    type="string"
                ),
                AttributeInfo(
                    name="main_situation",
                    description="주요 상황 (예: 신호, 위치, 도로 상황 등)",
                    type="string"
                ),
                AttributeInfo(
                    name="base_ratio",
                    description="기본 과실비율 (예: 80:20)",
                    type="string"
                ),
                AttributeInfo(
                    name="adjustment_factors",
                    description="조정 요소 (예: 신호위반, 속도위반 등)",
                    type="string"
                ),
                AttributeInfo(
                    name="law_reference",
                    description="관련 법규/판례",
                    type="string"
                )
            ]
            # 3. Self-Query Retriever 생성
            self_retriever = SelfQueryRetriever.from_llm(
                llm=self.gpt_4o_model,
                vectorstore=car_case_db,
                document_contents="교통사고 유형별 과실비율 데이터",
                metadata_field_info=metadata_field_info,
                search_kwargs={"k": 3}
            )
            # 4. 프롬프트 템플릿 정의 (PromptTemplate 객체로 생성)
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
{context}

아래의 정보를 바탕으로 교통사고 과실비율을 분석해 주세요.

---
출력 형식(아래 지침을 반드시 따르세요):

1. **질문과 가장 관련성 높은 사고유형 1개만** 우선적으로 상세히 설명하세요.
   - 만약 질문과 정확히 일치하는 사고유형이 데이터에 없다면,
     "정확히 일치하는 사고유형이 데이터에 없어, 가장 유사한 사고유형의 과실비율을 안내합니다."라는 문구를 답변 맨 앞에 추가하세요.
2. **여러 사고유형이 유사도가 비슷하게 높다면**, 각 사고유형을 아래와 같이 번호와 구분선(---)으로 명확히 구분해서 출력하세요.
   - 각 사고유형별로 반드시 아래의 항목을 포함하세요(번호와 함께):
     1. **사고유형**: [설명]
        - **기본 과실비율**: [A차량 XX% vs B차량 XX%]
        - **조정 요소**: [신호위반, 속도위반 등]
        - **참고사항**: [특이사항, 주의점]
        - **관련 법규/판례**: [링크/설명]
   - "교통사고 과실비율 분석 결과"라는 문구는 각 사고유형마다 반복하지 마세요.
3. **질문과의 관련성이 낮은 사고유형은 생략하거나, 맨 뒤에 간단히 요약만 하세요.**
4. **문서에 없는 정보는 임의로 만들지 마세요.**
5. **각 사고유형은 반드시 번호(1, 2, 3...)와 구분선(---)으로 구분하세요.**
6. 마지막에는 위의 사고유형별 과실비율 데이터를 참고하여, 질문 상황에 가장 적합하다고 판단되는 **AI가 판단한 과실비율**을 한 줄로 명확하게 제시하세요.
   - 예시: "**AI가 판단한 과실비율: A차량 60% vs B차량 40%**"
   - 반드시 사고유형별 데이터를 근거로 판단하세요. 임의로 생성하지 마세요.
   - 간단한 이유를 같이 설명해주세요

---
예시 출력:

정확히 일치하는 사고유형이 데이터에 없어, 가장 유사한 사고유형의 과실비율을 안내합니다.

1. **사고유형**: [설명]
   - **기본 과실비율**: [A차량 XX% vs B차량 XX%]
   - **조정 요소**: [신호위반, 속도위반 등]
   - **참고사항**: [특이사항, 주의점]
   - **관련 법규/판례**: [링크/설명]

---

2. **사고유형**: [설명]
   - **기본 과실비율**: [A차량 XX% vs B차량 XX%]
   - **조정 요소**: [신호위반, 속도위반 등]
   - **참고사항**: [특이사항, 주의점]
   - **관련 법규/판례**: [링크/설명]

---

**AI가 판단한 과실비율: A차량 60% vs B차량 40%**
"""
            )
            # 5. QA 체인 구성 및 실행
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.gpt_4o_model,
                retriever=self_retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template}
            )
            # 6. 검색 및 응답 생성
            retrieval_start = time.time()
            result = qa_chain.invoke({"query": user_input})
            retrieval_time = time.time() - retrieval_start
            total_time = time.time() - start_time
            logger.info(f"과실비율 분석 완료: '{user_input[:30]}...' (검색: {retrieval_time:.2f}초, 총: {total_time:.2f}초)")
            return result['result']
        except Exception as e:
            logger.error(f"과실비율 분석 중 오류: {str(e)}")
            return self._generate_accident_fallback_response(user_input)
    
    def process_user_query(self, user_input: str) -> tuple:
        """
        통합 처리 함수 (분류 + 카테고리별 처리)
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            tuple: (category, response)
        """
        try:
            # 1. 질문 분류
            category = self.classify_query(user_input)
            
            # 2. 카테고리별 처리
            if category == 'precedent':
                response = self.process_precedent(user_input)
            elif category == 'law':
                response = self.process_law(user_input)
            elif category == 'accident':
                response = self.process_accident(user_input)
            elif category == 'term':
                response = self.process_term(user_input)
            else:  # general
                response = self._process_general_placeholder(user_input)
            
            return category, response
            
        except Exception as e:
            logger.error(f"질문 처리 중 오류: {str(e)}")
            return 'general', self._generate_error_response(user_input, str(e))
    
    def _classify_with_finetuned_model(self, user_input: str) -> str:
        """
        파인튜닝된 GPT-3.5-turbo 모델로 분류
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            str: 분류 결과
            
        Raises:
            Exception: API 호출 실패 시
        """
        if not self.model_id:
            raise Exception("파인튜닝된 모델 ID가 설정되지 않음")
        
        try:
            # 파인튜닝된 모델 호출
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "다음 질문을 accident, precedent, law, term, general 중 하나로 분류하세요."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                max_tokens=10,
                temperature=0.0,
                timeout=5
            )
            
            # 응답 파싱
            category = response.choices[0].message.content.strip().lower()
            
            # 응답 검증
            if not self._is_valid_category(category):
                logger.warning(f"파인튜닝 모델이 잘못된 카테고리 반환: {category}")
                raise Exception(f"잘못된 분류 결과: {category}")
            
            return category
            
        except openai.APITimeoutError:
            logger.error("OpenAI API 타임아웃")
            raise Exception("API 타임아웃")
        except openai.APIError as e:
            logger.error(f"OpenAI API 에러: {str(e)}")
            raise Exception(f"API 에러: {str(e)}")
        except Exception as e:
            logger.error(f"분류 중 예상치 못한 에러: {str(e)}")
            raise
    
    def _fallback_classify(self, user_input: str) -> str:
        """
        키워드 기반 폴백 분류
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            str: 분류 결과
        """
        user_input_lower = user_input.lower()
        
        # 각 카테고리별 키워드 매칭 점수 계산
        category_scores = {}
        
        for category, keywords in self.FALLBACK_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in user_input_lower:
                    score += 1
            category_scores[category] = score
        
        # 가장 높은 점수의 카테고리 반환
        if category_scores and max(category_scores.values()) > 0:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"키워드 매칭 결과: {category_scores}, 선택: {best_category}")
            return best_category
        
        # 매칭되는 키워드가 없으면 general
        return 'general'
    
    def _is_valid_category(self, category: str) -> bool:
        """
        유효한 분류 카테고리인지 확인
        
        Args:
            category (str): 분류 결과
            
        Returns:
            bool: 유효 여부
        """
        return category and category.lower() in self.VALID_CATEGORIES
    
    def _generate_precedent_fallback_response(self, user_input: str) -> str:
        """판례 검색 실패 시 폴백 응답 (개선된 포맷)"""
        return f"""⚖️ **판례 검색 결과**

**🔍 검색 내용**: "{user_input}"

**⚠️ 일시적 오류 발생**

죄송합니다. 현재 판례 검색 시스템에 일시적인 문제가 발생했습니다.

**💡 다시 시도해 보세요**:

**🎯 구체적인 사건번호로 검색**
• "대법원 2019다12345 판례 내용은?"
• "서울고등법원 2015나60480 판례 검색"

**🎯 사고 유형과 함께 검색**
• "교차로 좌회전 사고 판례"
• "신호위반 관련 판례"
• "주차장 접촉사고 판례"

**🎯 법원별 검색**
• "대법원 교통사고 판례"
• "고등법원 과실비율 판례"

**📞 도움이 필요하시면 다른 방식으로 질문해주세요!**"""
    
    def _generate_law_fallback_response(self, user_input: str) -> str:
        """도로교통법 조회 실패 시 폴백 응답 (개선된 포맷)"""
        return f"""📚 **도로교통법 조회 결과**

**🔍 조회 내용**: "{user_input}"

**⚠️ 일시적 오류 발생**

죄송합니다. 현재 도로교통법 조회 시스템에 일시적인 문제가 발생했습니다.

**💡 다시 시도해 보세요**:

**🎯 구체적인 조문번호로 검색**
• "도로교통법 제5조 내용은?"
• "제25조 교차로 통행방법 알려주세요"

**🎯 키워드와 함께 검색**
• "신호위반 처벌 규정"
• "교차로 통행 규칙"
• "차로변경 관련 법률"

**🎯 상황별 검색**
• "좌회전 관련 도로교통법"
• "속도위반 벌금 규정"
• "보행자 보호 의무 법률"

**📞 도움이 필요하시면 다른 방식으로 질문해주세요!**"""
    
    def process_term(self, user_input: str) -> str:
        """
        교통사고 관련 용어 설명 (RAG 기반)
        
        Args:
            user_input (str): 사용자 질문
            
        Returns:
            str: 용어 설명 결과
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"용어 설명 시작: '{user_input[:30]}...'")
            
            # 1. VectorDB에서 term 컬렉션 가져오기
            term_db = self.vector_db_manager.get_vector_db('term')
            if not term_db:
                logger.error("term VectorDB를 찾을 수 없습니다.")
                return self._generate_term_fallback_response(user_input)
            
            # 2. 메타데이터 필드 정의 (Self-Query Retriever용)
            metadata_field_info = [
                AttributeInfo(
                    name="term_category",
                    description="용어 카테고리 (예: 법률용어, 사고유형, 도로시설 등)",
                    type="string"
                ),
                AttributeInfo(
                    name="related_terms",
                    description="관련 용어들",
                    type="string"
                ),
                AttributeInfo(
                    name="law_reference",
                    description="관련 법규 조문",
                    type="string"
                ),
                AttributeInfo(
                    name="precedent_reference",
                    description="관련 판례",
                    type="string"
                )
            ]
            
            # 3. Self-Query Retriever 생성
            self_retriever = SelfQueryRetriever.from_llm(
                llm=self.gpt_4o_model,
                vectorstore=term_db,
                document_contents="교통사고 관련 용어 및 정의 데이터",
                metadata_field_info=metadata_field_info,
                search_kwargs={"k": 3}
            )
            
            # 4. 용어 설명 프롬프트
            prompt = PromptTemplate(
                input_variables=["question", "context"],
                template="""
너는 교통사고 관련 용어를 설명하는 전문가야.

아래 문서(context)를 참고하여 사용자의 질문에 대해 용어를 쉽게 설명해줘.

---
질문: {question}

검색된 용어 정보:
{context}
---

출력 형식:

📖 **용어 설명**

**🔍 질문**: [사용자 질문]

**📚 용어 정의**:
• **정의**: [용어의 정확한 정의]
• **법적 근거**: [관련 법규]
• **실무 적용**: [실무에서의 적용 방법]

**💡 쉽게 설명**:
[일반인이 이해하기 쉬운 설명]

**🔗 관련 용어**:
• [관련 용어 1]: [간단 설명]
• [관련 용어 2]: [간단 설명]

**📌 참고사항**:
• [주의사항 1]
• [주의사항 2]

**조건**:
- 전문 용어는 반드시 쉬운 언어로 풀어서 설명하세요
- 법적 근거가 있다면 반드시 포함하세요
- 관련 용어는 2-3개만 선택적으로 표시하세요
- 문서에 없는 정보는 임의로 만들지 마세요
- 사용자가 특정 용어를 물어봤다면, 해당 용어를 우선적으로 설명하세요

답변:
"""
            )
            
            # 5. QA 체인 구성 및 실행
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.gpt_4o_model,
                retriever=self_retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )
            
            # 6. 검색 및 응답 생성
            retrieval_start = time.time()
            result = qa_chain.invoke({"query": user_input})
            retrieval_time = time.time() - retrieval_start
            
            total_time = time.time() - start_time
            logger.info(f"용어 설명 완료: '{user_input[:30]}...' (검색: {retrieval_time:.2f}초, 총: {total_time:.2f}초)")
            
            return result['result']
            
        except Exception as e:
            logger.error(f"용어 설명 중 오류: {str(e)}")
            return self._generate_term_fallback_response(user_input)
    
    def _generate_term_fallback_response(self, user_input: str) -> str:
        """용어 설명 실패 시 폴백 응답"""
        return f"""📖 **용어 설명 결과**

**🔍 질문 내용**: "{user_input}"

**⚠️ 일시적 오류 발생**

죄송합니다. 현재 용어 설명 시스템에 일시적인 문제가 발생했습니다.

**💡 다시 시도해 보세요**:

**🎯 구체적인 용어로 검색**
• "과실비율이란 무엇인가요?"
• "신호위반의 정의는?"
• "교차로 통행방법 설명해주세요"

**🎯 카테고리별 검색**
• "법률 용어: 과실이란?"
• "사고 유형: 추돌사고란?"
• "도로 시설: 교차로란?"

**🎯 관련 용어 검색**
• "과실비율과 관련된 용어들"
• "신호위반 관련 용어"
• "교차로 통행 관련 용어"

**📞 도움이 필요하시면 다른 방식으로 질문해주세요!**"""
    
    def _process_general_placeholder(self, user_input: str) -> str:
        """일반 질문 플레이스홀더 - 개선된 포맷"""
        return f"""👋 **노느 상담 챗봇**

안녕하세요! 교통사고 과실비율 상담 챗봇 **노느**입니다! 🚗

**🎯 현재 이용 가능한 기능**:

**✅ 판례 검색** (완료)
• "대법원 92도2077 판례 내용은?"
• "교차로 좌회전 사고 판례 검색"
• "신호위반 관련 판례를 알려주세요"

**🔧 개발 중인 기능**:
• 🚗 **교통사고 과실비율 분석**
• 📖 **교통사고 용어 설명**

**💡 도움이 필요하시면**:
• 구체적인 사건번호나 사고 상황을 알려주세요
• 궁금한 판례나 법률 조문을 말씀해주세요

**어떤 도움이 필요하신가요?** 😊"""
    
    def _generate_error_response(self, user_input: str, error_msg: str) -> str:
        """오류 발생 시 응답 - 개선된 포맷"""
        return f"""❌ **시스템 일시 오류**

**🔍 요청 내용**: "{user_input}"

**⚠️ 오류 상황**
죄송합니다. 일시적인 시스템 오류가 발생했습니다.

**💡 해결 방법**:
• 잠시 후 다시 시도해주세요
• 다른 방식으로 질문해주세요
• 구체적인 사건번호나 키워드를 사용해보세요

**🎯 추천 질문 방식**:
• "대법원 [사건번호] 판례"
• "[사고유형] 관련 판례"
• "[법원명] 교통사고 판례"

**도움이 필요하시면 언제든 말씀해주세요!** 🙏

*기술 정보: {error_msg}*"""
    
    def _generate_accident_fallback_response(self, user_input: str) -> str:
        """과실비율 분석 실패 시 폴백 응답"""
        return f"""🚗 **교통사고 과실비율 분석 결과**\n\n**🔍 질문 내용**: \"{user_input}\"\n\n**⚠️ 일시적 오류 발생**\n\n죄송합니다. 현재 과실비율 분석 시스템에 일시적인 문제가 발생했습니다.\n\n**💡 다시 시도해 보세요**:\n• 사고유형을 더 구체적으로 입력해 주세요 (예: 교차로 좌회전 vs 직진)\n• 신호, 위치, 도로 상황 등도 함께 입력해 주세요\n\n**📞 도움이 필요하시면 다른 방식으로 질문해주세요!**"""
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        분류기 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        return {
            'model_id': self.model_id,
            'valid_categories': list(self.VALID_CATEGORIES),
            'fallback_keywords_count': {
                category: len(keywords) 
                for category, keywords in self.FALLBACK_KEYWORDS.items()
            },
            'implemented_functions': ['classify_query', 'process_precedent', 'process_law'],
            'upcoming_functions': ['process_accident', 'process_term'],
            'status': 'partial_implementation'
        }


# 전역 분류기 인스턴스 (싱글톤 패턴)
_classifier_instance = None

def get_classifier() -> TrafficAccidentClassifier:
    """
    분류기 싱글톤 인스턴스 반환
    
    Returns:
        TrafficAccidentClassifier: 분류기 인스턴스
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = TrafficAccidentClassifier()
    
    return _classifier_instance


# 편의 함수들
def classify_user_query(user_input: str) -> str:
    """
    사용자 질문 분류 편의 함수
    
    Args:
        user_input (str): 사용자 입력
        
    Returns:
        str: 분류 결과
    """
    classifier = get_classifier()
    return classifier.classify_query(user_input)


def process_user_query(user_input: str) -> tuple:
    """
    사용자 질문 통합 처리 편의 함수
    
    Args:
        user_input (str): 사용자 입력
        
    Returns:
        tuple: (category, response)
    """
    classifier = get_classifier()
    return classifier.process_user_query(user_input)
