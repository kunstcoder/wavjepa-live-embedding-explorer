# WavJEPA Embedding Explorer

`labhamlet/wavjepa-base`를 로컬에 내려받아 오디오 특징을 추출하고, 각 오디오의 mean pooled embedding을 2D/3D 공간에 시각화하는 FastAPI 웹앱입니다.

## 기능

- Hugging Face `labhamlet/wavjepa-base` 스냅샷 로컬 저장
- 오디오 업로드 후 WavJEPA feature extraction
- 브라우저 마이크 입력의 실시간 chunk 기반 feature extraction
- 오디오별 pooled embedding 생성
- PCA / t-SNE 기반 2D 또는 3D 시각화
- 마이크 실시간 trajectory 시각화
- 파일별 duration, temporal step 수, embedding dimension, vector norm 표시
- live chunk별 RMS energy, elapsed time 표시

## 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py
python -m app.main
```

브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000) 를 열면 됩니다.

같은 네트워크의 다른 PC에서 접속하려면 서버 머신 IP로 접속하면 됩니다.

```bash
WAVJEPA_HOST=0.0.0.0 WAVJEPA_PORT=8000 python -m app.main
```

- 업로드 기반 분석은 `http://서버IP:8000`으로 접속해도 동작합니다.
- 원격 브라우저의 마이크 실시간 모드는 브라우저 보안 정책 때문에 `HTTPS` 또는 `localhost`에서만 동작합니다.
- HTTPS가 필요하면 `WAVJEPA_SSL_CERTFILE`, `WAVJEPA_SSL_KEYFILE` 환경변수로 인증서를 지정할 수 있습니다.

## 체크포인트 지원

앱은 Hugging Face 디렉터리뿐 아니라 PyTorch/Lightning 계열 `.ckpt`, `.pt`, `.pth`, `.bin` 체크포인트도 받을 수 있습니다.

### 방법 1. 앱에서 `.ckpt` 직접 사용

```bash
export WAVJEPA_MODEL_SOURCE=/absolute/path/to/model.ckpt
uvicorn app.main:app --reload
```

- 첫 로딩 시 체크포인트를 `.hf_cache/converted_models/<fingerprint>/` 아래의 Hugging Face 호환 `safetensors` 디렉터리로 자동 변환합니다.
- 이후 추론은 기존 `from_pretrained(...)` 경로를 그대로 사용합니다.
- 기본 템플릿은 `models/wavjepa-base`이며, 없으면 자동으로 내려받습니다.

### 방법 2. 미리 `safetensors`로 변환

```bash
python scripts/convert_ckpt_to_hf.py /absolute/path/to/model.ckpt
```

원하는 출력 위치를 지정하려면:

```bash
python scripts/convert_ckpt_to_hf.py \
  /absolute/path/to/model.ckpt \
  --output-dir /absolute/path/to/converted-model
```

변환이 끝나면 해당 디렉터리를 앱 입력으로 바로 쓸 수 있습니다.

```bash
export WAVJEPA_MODEL_SOURCE=/absolute/path/to/converted-model
uvicorn app.main:app --reload
```

템플릿용 HF 모델 디렉터리를 직접 지정하려면 `WAVJEPA_TEMPLATE_DIR` 또는 `--template-dir`를 사용하면 됩니다.

## 비고

- 입력 오디오는 백엔드에서 mono / 16kHz 로 맞춘 뒤 모델에 전달합니다.
- 첫 추론 시에도 모델이 없으면 자동으로 `models/wavjepa-base` 아래로 다운로드합니다.
- `.ckpt` 변환은 `labhamlet/wavjepa`의 HEAR 추론 코드 패턴을 참고해 `state_dict`를 정규화합니다.
- 체크포인트에 optimizer/scheduler state가 섞여 있어도 추론에 필요한 tensor weight만 추려서 변환합니다.
- 시각화 좌표는 원본 768차원 pooled embedding이 아니라 축소된 좌표입니다.
- 업로드 분석은 PCA 또는 t-SNE를 선택할 수 있고, 마이크 실시간 모드는 안정성을 위해 PCA로 고정됩니다.
- 마이크 기능은 브라우저 권한이 필요하며 `localhost` 또는 로컬 주소에서 실행해야 합니다.
- 로컬 머신에서 CPU만 사용할 경우 첫 추론과 다중 파일 처리 시간이 길 수 있습니다.

## kNN 평가

Audio-JEPA 논문/X-ARES 스타일에 맞춰, `train`/`test` split의 오디오를 `WavJEPA` encoder로 임베딩한 뒤 mean pooling한 clip embedding으로 weighted `kNN` 평가를 돌릴 수 있습니다.

기본 폴더 구조는 다음을 가정합니다.

```text
dataset_root/
  train/
    sample_000.wav
    sample_000.json
  test/
    sample_100.wav
    sample_100.json
```

- 각 `wav` 파일 옆에 같은 basename의 `json` 파일이 있어야 합니다.
- 기본적으로 JSON의 top-level `label`, `class`, `category`, `genre`, `instrument` 같은 키를 자동 탐색합니다.
- 라벨 키가 중첩돼 있으면 `--label-key metadata.label` 같은 식으로 지정하면 됩니다.
- 현재 스크립트는 단일 라벨 clip classification 용도입니다.

실행 예시:

```bash
./.venv/bin/python scripts/eval_knn.py /absolute/path/to/ESC50
```

여러 데이터셋을 한 번에 평가:

```bash
./.venv/bin/python scripts/eval_knn.py /absolute/path/to/xares_datasets
```

위 경로가 `train`/`test`를 바로 가지지 않으면, 바로 아래 자식 디렉터리들 중 `train`/`test`를 가진 폴더를 자동으로 찾아 평가합니다.

라벨 키를 명시하고 JSON 리포트를 저장하려면:

```bash
./.venv/bin/python scripts/eval_knn.py \
  /absolute/path/to/GTZAN \
  --label-key label \
  --output-json /absolute/path/to/results/gtzan_knn.json
```

주요 옵션:

- `--k`: 최근접 이웃 수. 기본값 `10`
- `--temperature`: 유사도 가중치 softmax temperature. 기본값 `0.07`
- `--batch-size`: test embedding의 kNN 점수 계산 배치 크기
- `--limit-per-split`: 디버깅용 소규모 subset 실행
- `--model-source`: HF 디렉터리나 `.ckpt/.pt/.pth/.bin` 체크포인트 직접 지정
