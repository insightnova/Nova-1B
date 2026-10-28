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

## 평가 결과

SAPO까지 학습한 최종 Nova-1B를 14개 과제의 전체 테스트 세트에서 **0-shot·10-shot**으로 평가했다. 10-shot은 가중치를 고정한 상태에서 프롬프트에 예시 10개를 제공하는 in-context learning 조건이다.

### 비교 모델 및 입력 형식

| 표기 | 모델 | 입력 형식 |
|---|---|---|
| Nova-1B | INSIGHTNOVA Nova-1B, 약 1.22B | 대화 템플릿을 적용하지 않은 평문 |
| Llama IT | Llama 3.2 1B Instruct | 공식 대화 템플릿, 고정된 대화형 구성 |
| Gemma IT | Gemma 3 1B IT | 공식 대화 템플릿, 고정된 대화형 구성 |
| CLOVA IT | HyperCLOVA X SEED Text Instruct 1.5B | 공식 대화 템플릿, 고정된 대화형 구성 |
| EXAONE IT | EXAONE 4.0 1.2B | 공식 대화 템플릿, 고정된 대화형 구성 |

비교 모델의 입력 구성은 두 방식의 전체 평가가 모두 완료된 NSMC·YNAT·BoolQ·SQuAD에서 모델별로 비교한 뒤 선택했다. 네 모델 모두 대화형 구성이 전반적으로 우수하여, 해당 구성을 14개 과제에 고정했다. 10-shot 예시는 user/assistant 메시지로 교대로 배치하고, 지시문은 첫 user 메시지에 넣었다. 과제별로 서로 다른 템플릿의 최고 점수를 골라 합산하지 않았다.

각 모델은 동일한 평가 문항과 few-shot 예시를 사용했다. 전체 테스트 세트 평가를 완료한 **140개 조건(5개 모델 × 14개 과제 × 2개 shot)**을 아래에 제시한다. 표본 실험 결과는 포함하지 않았다.

### 0-shot

| 과제 / 지표 | Nova-1B | Llama IT | Gemma IT | CLOVA IT | EXAONE IT |
|---|---:|---:|---:|---:|---:|
| NSMC / Accuracy | 78.10 | 49.65 | 78.32 | **84.52** | 73.80 |
| KLUE-YNAT / Accuracy | 30.59 | 41.45 | 47.13 | **57.07** | 34.84 |
| KoBEST COPA / F1 | 67.80 | 52.76 | 61.09 | **69.95** | 59.49 |
| KoBEST HellaSwag / F1 | **50.87** | 41.50 | 41.40 | 49.63 | 40.54 |
| KoBEST SentiNeg / F1 | **87.63** | 33.72 | 49.18 | 46.09 | 36.67 |
| KoBEST BoolQ / F1 | 67.99 | 54.54 | 56.87 | **76.69** | 65.58 |
| KoBEST WiC / F1 | 32.80 | 33.21 | 48.60 | **56.20** | 32.80 |
| KorQuAD v1 / F1 | **87.76** | 48.05 | 67.65 | 83.26 | 81.62 |
| SQuAD v1 / F1 | 76.50 | 72.95 | 72.43 | **80.15** | 74.46 |
| MNLI / Accuracy | 39.81 | 35.88 | 46.11 | 35.80 | **47.01** |
| STS-B / Pearson | 0.5575 | -0.1288 | 0.2959 | **0.7660** | 0.1274 |
| CoNLL-2003 / F1 | 0.00 | 0.00 | 0.00 | **1.73** | 0.00 |
| 영어→한국어 / BLEU | 20.00 | 7.72 | 18.90 | **26.89** | 23.64 |
| 한국어→영어 / BLEU | **26.53** | 13.69 | 18.61 | 16.86 | 24.35 |

### 10-shot

| 과제 / 지표 | Nova-1B | Llama IT | Gemma IT | CLOVA IT | EXAONE IT |
|---|---:|---:|---:|---:|---:|
| NSMC / Accuracy | 83.27 | 76.39 | 80.66 | **86.04** | 81.01 |
| KLUE-YNAT / Accuracy | 46.50 | 52.00 | 59.10 | **64.08** | 42.70 |
| KoBEST COPA / F1 | 68.09 | 53.37 | 62.28 | **74.15** | 62.17 |
| KoBEST HellaSwag / F1 | **55.16** | 43.48 | 46.86 | 51.75 | 43.45 |
| KoBEST SentiNeg / F1 | 94.45 | 66.47 | 65.35 | 43.41 | **94.46** |
| KoBEST BoolQ / F1 | 66.82 | 45.57 | 62.68 | 81.05 | **81.50** |
| KoBEST WiC / F1 | 35.19 | 40.94 | 39.61 | 45.18 | **45.31** |
| KorQuAD v1 / F1 | **87.14** | 62.87 | 68.48 | 83.75 | 82.04 |
| SQuAD v1 / F1 | **84.24** | 79.98 | 73.30 | 81.74 | 81.89 |
| MNLI / Accuracy | 65.20 | 31.80 | 43.55 | 65.75 | **74.27** |
| STS-B / Pearson | 0.5349 | -0.0626 | 0.0244 | 0.6919 | **0.7427** |
| CoNLL-2003 / F1 | 41.98 | 40.86 | 41.75 | 11.17 | **43.68** |
| 영어→한국어 / BLEU | 22.87 | 9.08 | 21.13 | **28.67** | 24.36 |
| 한국어→영어 / BLEU | **29.88** | 16.16 | 18.62 | 28.24 | 20.48 |

굵은 글씨는 각 행의 최고값이다. Accuracy와 F1은 0–100 척도로 표시했으며, BLEU는 0–100점, Pearson은 상관계수 원값이다. KoBEST는 **macro-F1**, KorQuAD·SQuAD는 **답변 토큰 F1**, CoNLL-2003은 **개체 단위 micro-F1**을 사용했다.

### 평가 설정과 해석

| 항목 | 설정 |
|---|---|
| 평가 체크포인트 | Nova-1B 및 각 비교 모델의 instruction-tuned 체크포인트 |
| Seed / 최대 문맥 길이 | 42 / 8,192토큰 |
| 연산 정밀도 | BF16 |
| 생성 방식 | Greedy decoding |
| 최대 생성 길이 | 독해 64, STS 16, 개체명 인식 128, 번역 256토큰 |
| Nova 기존 평문 실행 | 평가 배치 상한 64, microbatch token budget 16,384, 독해는 문항별 생성 |
| 비교 모델 고정 대화형 실행 | 평가 배치 상한 16, microbatch token budget 8,192, 독해는 배치 생성 |

이 결과는 **모델별 입력 형식을 명시한 비교**다. 입력 형식과 기존 실행의 배치 설정이 달라 엄밀한 동일 조건 비교로 해석하지 않는다. 템플릿 선택에 사용한 4개 과제는 최종 표에도 포함되어 있으므로, 해당 점수는 독립적인 선택 후 평가와 구분해야 한다.

Nova-1B는 28개 과제·shot 조건 중 **8개에서 1위, 9개에서 2위**를 기록했다. 한국어 독해, 한국어 HellaSwag, 한국어→영어 번역에서 두 shot 모두 가장 높은 점수를 보였으며, SQuAD 10-shot에서도 가장 높은 점수를 기록했다. 뉴스 주제 분류와 단어 의미 판별은 개선이 필요한 영역이다. 순위는 서로 다른 지표를 합산한 종합 점수가 아니며, 작은 점수 차이의 통계적 유의성을 의미하지 않는다.

## 모델 소스 코드 사용법

이 저장소는 Nova-1B의 모델 구현, Hugging Face 연동 코드, 추론 도구를 제공한다. 모델 가중치는 별도로 준비한 **Hugging Face 형식의 완전한 모델 폴더**를 사용한다. 아래 예시의 `exports/Nova-1B`를 실제 모델 폴더 경로로 바꾸면 된다.

### 폴더 구성

```text
example.py                 # GPU 번역 예제 (--backend torch 또는 flash)
example_cpu.py             # CPU 영어→한국어 번역 예제
nova/
  configuration_nova.py     # 모델 설정 및 AutoConfig 연동
  modeling_nova.py          # AutoModelForCausalLM 연동
  generation_nova.py        # 생성 및 배치 처리
  architecture.py          # GDN2 + GQA 모델 구조
  architecture_config.py   # 모델 구조 옵션
  gdn2.py                  # GDN2 블록 및 FLA 연결
  gdn2_spec.py             # GDN2 차원 정의
  cpu_backend.py           # PyTorch 기반 CPU·GPU 추론 (FlashAttention 불필요)
  fast_decode.py           # FlashAttention 경로의 캐시 기반 추론
  sequence_utils.py        # 시퀀스 입력 처리
scripts/
  export_hf.py             # 체크포인트의 Hugging Face 형식 변환
  infer.py                 # 평문 질문·응답 CLI
  verify_bundle.py         # 배포 파일 및 선택적 CUDA 검증
tests/                     # 인터페이스·직렬화·생성 동작 검증
```

### CPU 설치 및 번역 예제

Python 3.10 이상에서 `pyproject.toml`이 있는 저장소 루트로 이동한다. 기존 패키지와의 충돌을 피하려면 별도 가상환경을 사용한다. CPU 추론에는 Flash extra, OLMo-core, FLA, FlashAttention을 설치할 필요가 없다.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell에서는 위 줄 대신: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python example_cpu.py --model exports/Nova-1B
```

`example_cpu.py`는 실제 가중치를 CPU의 FP32로 불러와 영어 문장을 한국어로 번역한다. 기본 입력은 `My happy life`이며, Windows CPU에서 `나의 행복한 삶` 출력을 확인했다. 기본 CPU 스레드는 최대 4개이고 `--threads 8` 등으로 변경할 수 있다.

```bash
python example_cpu.py --model exports/Nova-1B --text "My happy life" --max-new-tokens 64
```

완전한 모델 폴더에는 가중치와 토크나이저가 모두 있어야 한다. GitHub 코드 ZIP에는 가중치가 포함되지 않는다. FP32 가중치 자체는 약 4.9GB이며, 실행에는 캐시·연산 버퍼와 운영체제를 위한 RAM이 추가로 필요하다. GPU 메모리 부족 시 자동 전환하는 방식이 아니므로 CPU 실행에는 이 예제를 명시적으로 사용한다. Mac 실행은 별도로 검증하지 않았다.

AutoClass를 직접 사용할 때는 `from_pretrained(..., inference_backend="cpu", dtype=torch.float32)`로 불러온 뒤 `.to("cpu").eval()`을 호출하고 입력도 CPU에 둔다. GPU 예제의 기본 백엔드는 `torch`이며, FlashAttention 사용 시 `flash`를 명시한다.

### GPU 설치: 두 가지 선택지

두 방식은 같은 가중치와 토크나이저를 사용한다. Python 3.10 이상에서 CUDA 드라이버와 호환되는 PyTorch를 준비하고, `pyproject.toml`이 있는 저장소 루트에서 설치한다. 별도 가상환경 사용을 권장한다.

| 선택 | 설치 | 실행 옵션 | 특징 |
|---|---|---|---|
| 기본 PyTorch | `python -m pip install -e .` | `--backend torch` (기본값) | 별도 `flash-attn`, OLMo-core, FLA 설치 없이 GPU 추론 |
| FlashAttention 가속 | `.[flash]` 및 `flash-attn` 설치 | `--backend flash` | Linux·NVIDIA CUDA, FLA GDN2 및 FlashAttention 2 사용 |

#### 1. FlashAttention 없이 설치·실행

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA 지원 PyTorch와 드라이버를 확인하세요'"
python example.py --model exports/Nova-1B --device cuda:0 --backend torch
```

이 방식에는 별도 CUDA 확장 컴파일이 필요하지 않다. PyTorch SDPA로 GQA와 sliding-window attention을 계산하고, GDN2는 PyTorch 재귀 계산을 사용한다. 생성 중에는 GDN2 상태·convolution 상태·attention KV cache를 유지한다. 긴 입력의 초기 처리에는 가속 경로보다 시간이 더 걸릴 수 있다. PyTorch가 내부적으로 선택하는 attention 커널과 별도 설치 패키지 `flash-attn`은 구분된다.

#### 2. FlashAttention 가속 설치·실행

Linux에서 Git이 필요하다. 소스 빌드가 필요하면 CUDA Toolkit(`nvcc`)과 C++ 컴파일러도 준비한다. 기본 설치를 완료한 같은 가상환경에서 다음을 실행한다.

```bash
python -m pip install -e ".[flash]"
python -m pip install --upgrade "setuptools>=77" wheel packaging psutil ninja
MAX_JOBS=1 NVCC_THREADS=1 python -m pip install "flash-attn>=2,<3" --no-build-isolation
python -c "import torch, fla, flash_attn; print(torch.__version__); assert torch.cuda.is_available()"
python example.py --model exports/Nova-1B --device cuda:0 --backend flash
```

`--no-build-isolation`은 현재 가상환경의 빌드 도구를 사용한다. `MAX_JOBS=1 NVCC_THREADS=1`은 컴파일 병렬도를 제한하며 추론 속도에는 영향을 주지 않는다. `Killed`로 빌드가 종료되면 시스템·컨테이너 메모리 한도를 확인한다. [FlashAttention 설치 안내](https://github.com/Dao-AILab/flash-attention#installation-and-features)

가속 경로의 의존성은 Transformers `4.57.6`, FLA·fla-core `0.5.2`, Triton `3.7.1`, 지정 commit의 OLMo-core이며 PyTorch `2.10` 이상을 요구한다. 실제 설치 버전은 이 의존성들의 호환 조건을 함께 만족해야 한다. FlashAttention 설치는 CUDA·PyTorch 및 빌드 환경에 따라 달라질 수 있다. `flash`를 선택했는데 의존성이 없으면 설치 안내와 함께 오류를 표시한다.

완전한 모델 폴더에는 가중치와 토크나이저가 모두 있어야 한다. GitHub 코드 ZIP에는 가중치가 포함되지 않는다. GPU 메모리가 부족할 때 CPU로 자동 전환하지 않는다. CPU 실행은 앞의 `example_cpu.py`를 사용한다.

코드 테스트:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```

### 빠른 시작: 영어→한국어 번역

저장소 루트의 [`example.py`](example.py)는 `AutoTokenizer`와 `AutoModelForCausalLM`로 모델을 불러와 평문 번역을 수행한다. `exports/Nova-1B`에 완전한 모델 폴더를 준비한 뒤 실행한다.

```bash
python example.py --model exports/Nova-1B
```

실제 최종 가중치에서 greedy decoding으로 확인한 출력:

```text
English: My happy life
Korean: 나의 행복한 삶
```

번역문은 모델이 직접 생성한 결과이며, 예제 코드에 정답을 넣거나 출력 후 번역문을 대체하지 않는다. 다른 문장은 `--text`로 지정한다.

```bash
python example.py --model exports/Nova-1B --text "My happy life" --max-new-tokens 64
```

### 모델과 토크나이저 로딩

패키지를 설치한 환경에서 `nova.register()`를 호출하면 `AutoConfig`와 `AutoModelForCausalLM`에 Nova 구현이 등록된다. 토크나이저는 `AutoTokenizer`로 불러온다.

```python
import torch
import nova
from transformers import AutoModelForCausalLM, AutoTokenizer

nova.register()
model_path = "exports/Nova-1B"  # 가중치와 토크나이저가 포함된 모델 폴더
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=False,
    inference_backend="torch",  # 가속 설치를 마쳤다면 "flash"
).to(device).eval()
```

기본 로딩은 체크포인트의 저장 정밀도를 유지하며, CUDA 연산에는 BF16 autocast를 사용한다. 백엔드 간 부동소수점 연산 차이로 생성 결과가 달라질 수 있다. 생성 API는 `AutoModelForCausalLM`을 기준으로 제공한다.

### 평문 추론

추론에는 BOS 뒤에 평문을 입력한다. 질문 응답은 `질문: ...\n답변:` 형식을 사용한다. 아래 코드는 위에서 불러온 모델과 토크나이저를 사용한다. 위 벤치마크의 정확한 재현에는 각 과제의 프롬프트, 예시, 채점 방식도 함께 맞춰야 한다.

```python
prompt = "질문: 대한민국의 수도는 어디인가요?\n답변:"
token_ids = [tokenizer.bos_token_id] + tokenizer.encode(
    prompt, add_special_tokens=False
)
input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

with torch.inference_mode():
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=128,
        do_sample=False,
    )

print(tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True))
```

입력과 생성 토큰을 합친 문맥 상한은 8,192토큰이다. `max_new_tokens`는 생성 상한이며, 기본 구현은 입력 뒤에 남은 문맥 범위 안에서 생성한다.

### 명령줄 실행

`scripts/infer.py`는 `--prompt`의 내용을 `질문: ...\n답변:`이라는 일반 텍스트로 구성하고 BOS를 한 번 붙인다.

```bash
python scripts/infer.py \
  --model exports/Nova-1B \
  --device cuda:0 \
  --backend torch \
  --prompt "대한민국의 수도는 어디인가요?" \
  --max-new-tokens 128
```

FlashAttention 가속 설치를 마쳤다면 `--backend flash`로 실행한다. CPU에서는 `--device cpu --backend torch`를 사용한다.

B200·BF16·greedy 조건에서 두 GPU 백엔드 모두 다음 답변을 생성하고 종료 토큰으로 정상 종료했다.

```text
대한민국의 수도는 서울입니다.
```

번역 예제도 두 경로 모두 `My happy life → 나의 행복한 삶`으로 정상 종료했다. 출력은 모델이 생성하며 코드에서 응답을 치환하지 않는다. 자기소개 등 일부 입력에서는 답변 뒤에 질문을 추가로 만들거나 반복하여 생성 상한에 도달할 수 있다.

`--sample`을 추가하면 temperature `0.8`, top-p `0.95`로 샘플링한다. 기본 실행은 greedy decoding이다.

### 체크포인트 변환 및 검증

공개 가중치는 SAPO까지 학습한 최종 체크포인트 하나이며, 공개 이름은 **Nova-1B**다. 최종 체크포인트를 별도 배포 폴더로 변환한다.

```bash
python scripts/export_hf.py \
  --checkpoint /path/to/final_checkpoint \
  --output exports/Nova-1B \
  --max-shard-gb 1

python scripts/verify_bundle.py --model exports/Nova-1B
python scripts/verify_bundle.py --model exports/Nova-1B --cuda --backend torch
# 가속 설치 환경에서:
python scripts/verify_bundle.py --model exports/Nova-1B --cuda --backend flash
```

변환 결과에는 분할 `safetensors`, 모델 설정, 토크나이저, custom model code와 파일별 SHA-256 기록이 포함된다. 원본 체크포인트는 유지하고 새 폴더에 저장한다. 원시 사전학습 DCP·옵티마이저 상태는 이 변환기의 입력 대상이 아니다.

`--metadata-only`로 만든 미리보기는 가중치가 없어 추론에 사용할 수 없다. 개발 과정의 CPU 테스트는 다음 명령으로 실행한다.

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```

### 지원 범위

| 항목 | 현재 지원 |
|---|---|
| Hugging Face 연동 | `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM` |
| 추론 | `forward()` logits, `generate()` 텐서 출력 |
| 생성 | Greedy, temperature·top-p·top-k sampling, sampling 시 다중 응답 |
| 배치 | 좌·우 패딩, 인접한 동일 길이 입력의 배치 생성 |
| 배치 생성 옵션 | `generation_batch_size`, 기본값 1 |

일반 `AutoModel` backbone 출력, `device_map`·offload, 양자화, beam search, streamer, 생성 scores 반환과 외부 HF `past_key_values` 입력은 현재 지원 범위 밖이다. 생성 캐시는 생성 호출 중 모델에 연결되므로 서비스에서는 같은 모델 인스턴스의 `generate()` 요청을 직렬화하거나 인스턴스를 분리한다.

GitHub 배포본에는 소스 코드와 문서를 포함한다. `exports/`, 가중치, 캐시와 가상환경은 `.gitignore`에서 제외한다.

## 한계

Nova-1B의 생성 결과에는 사실 오류나 부정확한 응답이 포함될 수 있다. 평가에서 **KoBEST WiC의 특정 답변 편향**과 **CoNLL-2003 0-shot의 출력 형식 오류**가 확인됐다. 평문 입력에서도 질문에 따라 반복이나 지시 이탈이 발생할 수 있다. 장문 이해 및 실제 서비스 환경의 성능은 별도 검증이 필요하다.

## 개발자 및 기여

| &nbsp;&nbsp;개&#8288;발&#8288;자&nbsp;&nbsp; | 담당 업무 |
|---|---|
| **박&#8288;정&#8288;연** | Nova-1B 모델 아키텍처 선정 및 구현, 학습 데이터 수집, 평가 데이터셋 구축 및 가공, post-training(SFT·DPO·SAPO) 수행 및 모델 성능 평가 |
| **이&#8288;경&#8288;진** | 영어·지시·선호 데이터셋 수집 및 학습 데이터 구성, pre-training 파이프라인 개발·수행, pre-training 모델 성능 평가 |
| **조&#8288;지&#8288;윤** | 한국어 및 도표·수식·코드 데이터 수집·가공, 자체 Byte-Level BPE 토크나이저 개발 |

## 참고한 공개 프로젝트

구현에는 [OLMo-core](https://github.com/allenai/OLMo-core), [Flash Linear Attention(FLA)](https://github.com/fla-org/flash-linear-attention), [Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl)을 활용했다. 외부 코드와 데이터는 각각의 저작권 및 이용 조건을 따른다.

## 저작권 및 라이선스

**Copyright (c) 2026 주식회사 인사이트노바**

본 저장소에서 주식회사 인사이트노바가 저작권을 보유한 코드와 문서는 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)에 따라 배포한다. 외부 저작물은 각각의 이용 조건을 따른다.
