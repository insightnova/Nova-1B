---
language:
  - ko
  - en
pipeline_tag: text-generation
tags:
  - nova
  - gdn2
  - custom_code
---

# Nova-1B

**Nova-1B**는 **주식회사 인사이트노바(INSIGHTNOVA)**가 개발한 한국어·영어 중심의 자가회귀(auto-regressive) 언어모델이다. Gated DeltaNet 2(GDN2)와 Grouped-Query Attention(GQA)을 결합한 하이브리드 구조와 자체 개발한 Byte-Level BPE 토크나이저를 사용했다.

공개 데이터셋을 수집·정제하여 **3-stage pre-training**을 수행하고, **SFT → DPO → RL(SAPO)** 순서로 post-training을 진행했다. pre-training에서 **약 362.3B 토큰**, post-training을 포함한 전체 학습에서 **약 372B 토큰**을 처리했다. 이는 서로 다른 고유 토큰의 수가 아니라 누적 학습량이다.

## 모델 구성

| 항목 | 설정 |
|---|---|
| 모델 유형 | Decoder-only causal language model |
| 주요 언어 | 한국어, 영어 |
| 파라미터 수 | 1,220,640,156개 (약 1.22B) |
| 레이어 | 26층 (`[GDN2, Attention] × 13`) |
| 은닉 차원 / FFN 중간 차원 | 1,536 / 6,144 |
| Attention | GQA (Query head 12개, KV head 4개), Sliding Window Attention |
| Attention window / 최대 문맥 길이 | 4,096 / 8,192토큰 |
| 토크나이저 | 자체 Byte-Level BPE, 어휘 65,536개 |

GDN2와 Attention 블록을 교대로 배치하고, Attention에 RoPE와 Query/Key 정규화를 적용했다. 입력 임베딩과 출력 헤드의 가중치를 공유했다.

## 학습 데이터

한국어는 공공·전문 문서, 신문, 위키, 법령·용어 자료 및 웹 코퍼스를 수집했다. 주요 출처는 **KorMix, KORMo, FineWeb-2, HPLT, 국립국어원 모두의 말뭉치, AI Hub** 등이다. 영어는 **Dolma/Common Crawl, 영어 위키백과, STEM 웹 문서** 등을 활용했다. 이 밖에 수학·과학, 코드, 지시·추론, 한영 병렬 데이터를 포함했다.

수집한 문서를 정규화·정제하고 품질과 문서 길이에 따라 pre-training 코퍼스를 구성했다. post-training에서는 한국어·영어 질의응답과 대화, 지시·선호 데이터를 학습 형식에 맞게 가공했다.

## 학습 방법

### 1. 3-stage pre-training

다음 토큰 예측(next-token prediction)을 통해 일반 코퍼스에서 고품질 단문, 고품질 장문으로 학습 데이터를 전환했다.

| 단계 | 학습 데이터 | 문맥 길이 | 학습량 |
|---|---|---:|---:|
| Stage 1 | 한국어·영어 일반 문서, 수학·지시 등 | 4,096 | 223.3B 토큰 |
| Stage 2 | 고품질 문서 중 2,048토큰 이하 | 4,096 | 68.2B 토큰 |
| Stage 3 | 고품질 장문 및 한영 병렬 데이터 | 8,192 | 70.8B 토큰 |
| **합계** | | | **약 362.3B 토큰** |

Stage 3에는 정제한 한영 병렬 데이터 **3,052,224문장쌍**을 두 번역 방향으로 균등하게 구성하여 배치했다. 학습률에는 WSD(Warmup–Stable–Decay) 스케줄을 적용했다.

### 2. SFT — 지시 및 대화 응답 학습

한국어·영어 질의응답, 다중 턴 대화, 문서 독해, 글쓰기·교정 및 지시 수행 데이터로 SFT(Supervised Fine-Tuning)를 수행했다. 대화 이력을 문맥으로 사용하고 assistant 응답에 대해 손실을 계산했다. **약 1,204만 개의 응답 단위 샘플**을 학습했다.

### 3. DPO — 응답 선호 최적화

한국어·영어 선호 데이터의 선호 응답(chosen)과 비선호 응답(rejected)을 비교하여 DPO(Direct Preference Optimization)를 수행했다. 고정된 참조 모델을 기준으로 선호 응답의 상대적인 생성 확률을 높이도록 학습했다. **719,507개 선호 응답쌍**을 사용했다.

### 4. RL - SAPO 기반 검증 및 AI 평가에 따른 강화학습

SAPO(Soft Adaptive Policy Optimization)를 적용하여 모델이 생성한 응답을 평가하고 정책을 업데이트했다. 문자열·수치·날짜 정답은 검증기(verifier)로 확인하고, 응답의 내용과 근거는 LLM 평가 모델(judge)로 평가했다. 한국어 **19,077건**, 영어 **6,942건**을 포함한 **26,019개 프롬프트**로 학습했다.

## 공개 체크포인트

공개 모델 이름은 **Nova-1B**이며, SFT·DPO·SAPO를 완료한 최종 체크포인트 하나를 배포한다. 모델 코드는 Transformers의 `AutoModelForCausalLM` 및 `AutoTokenizer`와 연동된다.

## 평가 결과

최종 모델의 평문 입력 기반 전체 테스트 세트 결과다. 10-shot은 예시 10개를 프롬프트에 제공하는 조건이다.

| 과제 / 지표 | 0-shot | 10-shot |
|---|---:|---:|
| NSMC / Accuracy | 78.10 | 83.27 |
| KLUE-YNAT / Accuracy | 30.59 | 46.50 |
| KoBEST COPA / F1 | 67.80 | 68.09 |
| KoBEST HellaSwag / F1 | 50.87 | 55.16 |
| KoBEST SentiNeg / F1 | 87.63 | 94.45 |
| KoBEST BoolQ / F1 | 67.99 | 66.82 |
| KoBEST WiC / F1 | 32.80 | 35.19 |
| KorQuAD v1 / F1 | 87.76 | 87.14 |
| SQuAD v1 / F1 | 76.50 | 84.24 |
| MNLI / Accuracy | 39.81 | 65.20 |
| STS-B / Pearson | 0.5575 | 0.5349 |
| CoNLL-2003 / F1 | 0.00 | 41.98 |
| 영어→한국어 / BLEU | 20.00 | 22.87 |
| 한국어→영어 / BLEU | 26.53 | 29.88 |

STS-B는 Pearson 상관계수이며, 나머지는 100점 척도다. KoBEST는 macro-F1, 독해는 답변 토큰 F1, CoNLL은 개체 단위 micro-F1이다. seed 42, BF16, 최대 문맥 8,192토큰으로 평가했다. CoNLL 0-shot에는 출력 형식 오류의 영향이 있다.

## 사용 방법

소스 저장소에서 `pip install -e .`로 설치하면 별도 FlashAttention 설치 없이 PyTorch 기반 추론을 사용할 수 있다. Linux·NVIDIA CUDA에서 가속 경로를 사용하려면 README에 따라 `.[flash]`와 FlashAttention을 추가 설치하고 `inference_backend="flash"`를 지정한다. Transformers 버전은 `4.57.6`이다. 아래 `model_path`에는 가중치와 토크나이저가 포함된 완전한 로컬 모델 폴더 경로를 넣는다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/Nova-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, inference_backend="torch",
).to("cuda:0").eval()

prompt = (
    "Translate the following English sentence into Korean. "
    "Output only the translation.\n"
    "English: My happy life\nKorean:"
)
ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
inputs = torch.tensor([ids], dtype=torch.long, device="cuda:0")
with torch.inference_mode():
    output = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=64,
        do_sample=False,
    )
print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip())
```

실제 최종 가중치를 NVIDIA B200에서 검증한 출력:

```text
나의 행복한 삶
```

CPU 추론은 코드 저장소의 `example_cpu.py`로 실행한다. CPU에서는 기본 패키지(`pip install -e .`)만 설치하고, AutoClass 로딩 시 `inference_backend="cpu", dtype=torch.float32`를 지정하며 모델과 입력을 CPU에 둔다. 실제 Windows CPU에서도 위 번역 출력을 확인했다. GPU 메모리 부족 시 CPU로 자동 전환하지 않는다.

추론에는 BOS 뒤에 평문을 입력한다. 질문 응답은 `질문: ...\n답변:` 형식을 사용하며, 코드 저장소의 `scripts/infer.py`로 실행할 수 있다. `device_map`, 양자화, beam search, streamer 및 외부 HF `past_key_values`는 현재 지원하지 않는다. 입력과 생성 토큰을 합친 최대 문맥은 8,192토큰이다.

## 한계

생성 결과에는 사실 오류나 부정확한 응답이 포함될 수 있다. KoBEST WiC의 답변 편향과 CoNLL-2003 0-shot의 출력 형식 오류가 확인되었다. 평문 입력에서도 질문에 따라 반복이나 지시 이탈이 발생할 수 있다. 장문 이해와 실제 서비스 성능은 별도 검증이 필요하다.

## 라이선스

코드와 문서는 Apache License 2.0을 따른다.

개발 회사: **주식회사 인사이트노바(INSIGHTNOVA)**. 외부 의존성은 `THIRD_PARTY_NOTICES.txt`를 참고한다.
