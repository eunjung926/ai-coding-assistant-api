# AI Coding Assistant API

AI 코딩 어시스턴트 사용자 연구를 위한 FastAPI 백엔드 서버입니다.  
Qwen2-7B + LoRA adapter로 자연어 프롬프트를 Python 코드로 스트리밍 생성합니다.

## 개요

[VS Code 확장](../vscode-ai-coding-assistant-ui)과 연동되는 API 서버로, 두 가지 코드 생성 모드를 제공합니다.

| Adapter | 설명 |
|---------|------|
| `vul` | 취약 코드 생성 |
| `ske` | 스켈레톤 코드 생성 |

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서버 상태 확인 |
| GET | `/autocompleteNLtoCode/vul/stream` | 취약 코드 생성 (SSE 스트리밍) |
| GET | `/autocompleteNLtoCode/ske/stream` | 스켈레톤 코드 생성 (SSE 스트리밍) |
| GET | `/autocompleteNLtoCode/vul` | 취약 코드 생성 (일반) |
| GET | `/autocompleteNLtoCode/ske` | 스켈레톤 코드 생성 (일반) |

Swagger UI: `http://localhost:5059/docs`

## 빠른 시작

### 로컬 실행

```bash
pip install -r requirements.txt
python run.py
```

### Docker 실행

```bash
docker compose build
docker compose up -d
```

### 동작 확인

```bash
curl "http://localhost:5059/autocompleteNLtoCode/ske?input=Write+a+login+function"
```

## PEFT LoRA 학습

Qwen2-7B에 LoRA adapter를 파인튜닝하는 코드가 `training/`에 포함되어 있습니다.  
추론 서버와 동일한 프롬프트 형식(`### Instruction: ... ### Response:`)을 사용합니다.

### 설치

```bash
pip install -r training/requirements.txt
```

### 학습 실행

```bash
# 취약 코드 adapter (vul)
python -m training.train \
  --adapter-type vul \
  --data-path training/data/example_vul.jsonl

# 스켈레톤 코드 adapter (ske)
python -m training.train \
  --adapter-type ske \
  --data-path training/data/example_ske.jsonl
```

학습 결과는 `training/outputs/<adapter-type>/`에 저장됩니다.  
API 서버에 적용하려면 `adapters/vul/` 또는 `adapters/ske/`로 복사하세요.

### 데이터 형식 (JSONL)

```json
{"instruction": "자연어 요청", "output": "생성할 Python 코드"}
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--adapter-type` | — | `vul` 또는 `ske` |
| `--data-path` | — | 학습 JSONL 경로 |
| `--output-dir` | `training/outputs/<type>` | adapter 저장 경로 |
| `--epochs` | `3` | 학습 epoch |
| `--use-4bit` | off | 4-bit 양자화 로딩 (VRAM 절약) |

LoRA 설정 (`r=8`, `alpha=32`, `target_modules=[q_proj, v_proj]`)은 기존 adapter와 동일합니다.

## 프로젝트 구조

```
app/                     # FastAPI 추론 서버
training/
├── train.py             # PEFT LoRA 학습 진입점
├── config.py            # LoRA / 학습 하이퍼파라미터
├── dataset.py           # JSONL 데이터셋 로더
└── data/                # 예시 학습 데이터
adapters/
├── vul/                 # 취약 코드 LoRA adapter
└── ske/                 # 스켈레톤 LoRA adapter
run.py                   # 서버 진입점
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BASE_MODEL_PATH` | `Qwen/Qwen2-7B` | 베이스 모델 경로 |
| `VUL_ADAPTER_PATH` | `adapters/vul` | 취약 코드 adapter |
| `SKE_ADAPTER_PATH` | `adapters/ske` | 스켈레톤 adapter |
| `PORT` | `5059` | 서버 포트 |
| `NGROK_AUTHTOKEN` | — | ngrok 인증 토큰 (선택) |
| `NGROK_DOMAIN` | — | ngrok 고정 도메인 (선택) |
| `GPU_FREE_GB_THRESHOLD` | `30` | GPU 자동 선택 VRAM 기준 (GB) |

## 관련 저장소

- [vscode-ai-coding-assistant-ui](../vscode-ai-coding-assistant-ui) — VS Code 확장 (프론트엔드)

배포 상세 가이드는 [DEPLOY.md](DEPLOY.md)를 참고하세요.
