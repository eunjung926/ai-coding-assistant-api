# AI Coding Assistant API — 배포 가이드

이 서버는 **Qwen2-7B** 기반의 LoRA 파인튜닝 모델을 FastAPI로 서빙합니다.  
취약 코드 생성(`vul` 어댑터)과 스켈레톤 코드 생성(`ske` 어댑터) 두 가지 모드를 지원합니다.

---

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [처음 배포하는 경우 (이미지 빌드)](#3-처음-배포하는-경우-이미지-빌드)
4. [서버 실행](#4-서버-실행)
5. [외부에서 접속하기 (VSCode SSH 포트 포워딩)](#5-외부에서-접속하기-vscode-ssh-포트-포워딩)
6. [같은 서버를 여러 사용자가 쓸 때 (포트 충돌 방지)](#6-같은-서버를-여러-사용자가-쓸-때-포트-충돌-방지)
7. [API 엔드포인트](#7-api-엔드포인트)
8. [발생 가능한 오류 및 해결 방법](#8-발생-가능한-오류-및-해결-방법)

---

## 1. 사전 요구사항

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GPU, VRAM **16~20GB 이상** |
| Docker | 20.10 이상 |
| NVIDIA Container Toolkit | 설치 필요 (`nvidia-docker2`) |
| Docker 그룹 권한 | 사용자 계정이 `docker` 그룹에 속해야 함 (아래 참고) |

### Docker 그룹 권한 설정

`docker` 명령어를 `sudo` 없이 사용하려면 현재 계정을 docker 그룹에 추가해야 합니다.

```bash
sudo usermod -aG docker $USER
```

설정 후 **로그아웃 후 재로그인** 하거나 아래 명령으로 즉시 적용합니다:

```bash
newgrp docker
```

적용 확인:
```bash
docker ps   # sudo 없이 실행되면 성공
```

---

## 2. 디렉터리 구조

빌드 전 아래 구조가 갖춰져 있어야 합니다.

```
ai-coding-assistant-api/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py                        # 서버 실행 진입점
├── app/                          # FastAPI 애플리케이션
└── adapters/
    ├── vul/                      # 취약 코드 생성 LoRA 어댑터
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── ske/                      # 스켈레톤 코드 생성 LoRA 어댑터
        ├── adapter_config.json
        └── adapter_model.safetensors
```

`adapters/vul/`과 `adapters/ske/` 디렉터리 안에 어댑터 가중치 파일이 없으면 서버가 실행되지 않습니다.

---

## 3. 처음 배포하는 경우 (이미지 빌드)

> **이미 이미지가 서버에 있다면 이 단계는 건너뛰세요.**  
> 빌드 중 Qwen2-7B (~15GB)를 HuggingFace에서 다운로드합니다. 수십 분 소요됩니다.

이미지가 있는지 먼저 확인:
```bash
docker images | grep ai-coding-assistant-api
```

없을 경우 빌드:
```bash
cd /path/to/server

# Qwen2-7B는 public 모델이므로 토큰 불필요
docker compose build
```

---

## 4. 서버 실행

### 실행

```bash
docker run -d \
  --gpus all \
  -p 5059:5059 \
  --shm-size=2g \
  --name ai-coding-api \
  ai-coding-assistant-api:latest
```

### 로그 확인

```bash
docker logs -f ai-coding-api
```

정상 실행 시 아래와 같은 로그가 출력됩니다:
```
INFO | Loading tokenizer...
INFO | Loading base model...
INFO | Loading vul adapter...
INFO | Loading ske adapter...
INFO | Uvicorn running on http://0.0.0.0:5059
```

모델 로딩까지 약 1~2분 소요됩니다.

### 동작 확인 (테스트)

```bash
curl "http://localhost:5059/autocompleteNLtoCode/ske?input=Write+a+login+function"
```

`{"status": true, "generated_code": "..."}` 형태의 응답이 오면 정상입니다.

### 서버 중지 및 재시작

```bash
docker stop ai-coding-api    # 중지
docker rm ai-coding-api      # 컨테이너 삭제
docker restart ai-coding-api # 재시작
```

---

## 5. 원격 서버에서 접속하기 (VS Code SSH 포트 포워딩)

GPU 서버에 SSH로 접속해 Docker 컨테이너를 실행한 경우, 로컬 PC에서 API에 접근하려면 포트 포워딩을 사용합니다.

### 설정 방법

1. VS Code에서 원격 서버에 SSH 접속합니다.
2. 하단 **PORTS** 탭에서 **포트 추가** → `5059` 입력
3. 로컬 브라우저에서 접속:

```
http://localhost:5059/docs
```

> 포트 포워딩은 VS Code SSH 세션이 열려 있는 동안에만 유지됩니다.

---

## 6. 같은 서버를 여러 사용자가 쓸 때 (포트 충돌 방지)

**같은 서버에서 여러 사람이 각자의 컨테이너를 띄우려면 포트 번호와 컨테이너 이름을 다르게 설정해야 합니다.**

### 포트 번호 변경

`-p` 옵션의 앞쪽 번호(호스트 포트)를 각자 다르게 지정합니다:

```bash
# 사용자 A: 기본 5059
docker run -d --gpus all -p 5059:5059 --shm-size=2g --name ai-coding-api ai-coding-assistant-api:latest

# 사용자 B: 5060 사용
docker run -d --gpus all -p 5060:5059 --shm-size=2g --name ai-coding-api-B ai-coding-assistant-api:latest

# 사용자 C: 5061 사용
docker run -d --gpus all -p 5061:5059 --shm-size=2g --name ai-coding-api-C ai-coding-assistant-api:latest
```

포트가 겹치는지 사전 확인:
```bash
sudo lsof -i :5059
```

### GPU 자동 선택 (기본)

`docker run --gpus all` 로 띄우면, **모델 로드 직전**에 `nvidia-smi`로 여유 VRAM을 보고 GPU를 고릅니다.

| 조건 | 동작 |
|------|------|
| GPU 0 여유 > 30GB | GPU 0만 사용 (`CUDA_VISIBLE_DEVICES=0`) |
| 그렇지 않고 GPU 1 여유 > 30GB | GPU 1만 사용 (`CUDA_VISIBLE_DEVICES=1`) |
| 둘 다 30GB 이하 | GPU 0·1 모두 사용 + `device_map=auto`로 분산 |

기준값 변경:

```bash
docker run -d --gpus all -p 5059:5059 --shm-size=2g \
  -e GPU_FREE_GB_THRESHOLD=28 \
  --name ai-coding-api ai-coding-assistant-api:latest
```

로그 예: `GPU pick: gpu0 free 42.1GB > 30GB -> CUDA_VISIBLE_DEVICES=0`

수동 고정(자동 선택 끔):

```bash
docker run -d --gpus all -e CUDA_VISIBLE_DEVICES=1 -p 5059:5059 ...
```

### GPU 할당 분리 (수동, 선택)

VRAM이 부족하거나 특정 GPU만 쓰고 싶을 때:

```bash
# GPU 목록 확인
nvidia-smi --list-gpus

# GPU 1번만 사용
docker run -d --gpus '"device=1"' -p 5060:5059 --shm-size=2g --name ai-coding-api-B ai-coding-assistant-api:latest
```

---

## 7. API 엔드포인트

기본 URL: `http://localhost:<포트>`  
Swagger UI: `http://localhost:<포트>/docs`

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서버 상태 확인 |
| GET | `/autocompleteNLtoCode/vul/stream` | 취약 코드 생성 (스트리밍) |
| GET | `/autocompleteNLtoCode/ske/stream` | 스켈레톤 코드 생성 (스트리밍) |
| GET | `/autocompleteNLtoCode/vul` | 취약 코드 생성 (일반) |
| GET | `/autocompleteNLtoCode/ske` | 스켈레톤 코드 생성 (일반) |

### 요청 예시

```bash
# 일반 요청
curl "http://localhost:5059/autocompleteNLtoCode/ske?input=Write+a+login+function"

# 스트리밍 요청
curl -N "http://localhost:5059/autocompleteNLtoCode/vul/stream?input=Write+a+login+function"
```

### 스트리밍 응답 형식 (SSE)

```
data: {"token": "def "}
data: {"token": "login("}
data: {"token": "username, password):"}
...
data: {"done": true, "stats": {"ttft": 0.512, "total_time": 3.21, "total_tokens": 87, "tokens_per_sec": 27.1}}
```

---

## 8. 발생 가능한 오류 및 해결 방법

### 빌드 시

---

**`ERROR: failed to solve: ... HuggingFace download failed`**

- 원인: 네트워크 불안정 또는 HuggingFace rate limit
- 해결: 재시도. 반복되면 `HF_TOKEN` 설정 후 빌드

---

**`No space left on device`**

- 원인: Qwen2-7B (~15GB) + Docker 레이어로 디스크 부족
- 해결:
  ```bash
  docker system prune -f   # 불필요한 이미지/캐시 정리
  df -h                    # 디스크 여유 확인
  ```

---

**`adapters/vul/adapter_model.safetensors: file not found`**

- 원인: `adapters/vul/` 또는 `adapters/ske/` 디렉터리에 어댑터 파일 누락
- 해결: 어댑터 가중치 파일을 해당 디렉터리에 복사 후 재빌드

---

### 실행 시

---

**`permission denied while trying to connect to the Docker daemon`**

- 원인: 현재 사용자가 docker 그룹에 없음
- 해결:
  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

---

**`Error: Bind for 0.0.0.0:5059 failed: port is already allocated`**

- 원인: 같은 서버에서 다른 사용자가 이미 5059 포트를 점유 중
- 해결: `-p` 옵션의 호스트 포트를 변경 (위 6번 참고)
  ```bash
  sudo lsof -i :5059   # 누가 쓰고 있는지 확인
  ```

---

**`Conflict: The container name "/ai-coding-api" is already in use`**

- 원인: 같은 이름의 컨테이너가 이미 존재
- 해결:
  ```bash
  docker rm ai-coding-api          # 기존 컨테이너 삭제 후 재실행
  # 또는 --name을 다른 이름으로 변경
  ```

---

**`CUDA out of memory`**

- 원인: GPU VRAM 부족 (Qwen2-7B float32는 약 28GB 필요)
- 해결:
  1. `nvidia-smi`로 현재 VRAM 사용량 확인 후 다른 프로세스 종료
  2. [app/models/loader.py](app/models/loader.py)에서 `torch_dtype`을 float16으로 변경하면 VRAM 절반으로 감소:
     ```python
     torch_dtype=torch.float16
     ```
     변경 후 이미지 재빌드 필요

---

**`RuntimeError: Expected all tensors to be on the same device`**

- 원인: `device_map="auto"`로 인해 멀티 GPU 환경에서 레이어가 분산될 때 어댑터 레이어와 충돌
- 해결: 단일 GPU 강제 지정
  ```bash
  docker run -d --gpus '"device=0"' -p 5059:5059 --shm-size=2g --name ai-coding-api ai-coding-assistant-api:latest
  ```

---

**`adapter_lock` 관련 timeout / 응답 없음**

- 원인: 스트리밍 요청이 중단됐는데 락이 해제되지 않은 경우 (드문 엣지케이스)
- 해결:
  ```bash
  docker restart ai-coding-api
  ```

---

**`ngrok: failed to start tunnel`**

- 원인: ngrok 인증 토큰 미설정 또는 터널 수 초과
- 영향: ngrok 터널 없이도 서버 자체는 정상 동작 (localhost 접근 가능), 무시해도 됨
- 해결: 필요하면 `NGROK_AUTHTOKEN` 환경 변수 추가:
  ```bash
  docker run -d --gpus all -p 5059:5059 --shm-size=2g \
    -e NGROK_AUTHTOKEN=your_token_here \
    -e NGROK_DOMAIN=your-domain.ngrok-free.app \
    --name ai-coding-api ai-coding-assistant-api:latest
  ```

---

**모델 로딩 후 첫 번째 추론이 매우 느림**

- 원인: CUDA 커널 컴파일 (정상 동작)
- 해결: 두 번째 요청부터 정상 속도로 동작

---

### 외부 접속 시

---

**VSCode 포트 포워딩 후에도 브라우저에서 접속이 안 됨**

- 원인: VSCode SSH 세션이 끊어지거나 포트 포워딩이 비활성화됨
- 해결: VSCode PORTS 탭에서 포트 상태 확인, 필요 시 재등록

---

**스트리밍이 중간에 끊김**

- 원인: nginx 또는 프록시의 버퍼링 (서버는 `X-Accel-Buffering: no` 헤더를 전송하고 있음)
- 해결: 프록시 없이 직접 접속하거나, nginx 설정에 `proxy_buffering off;` 추가
