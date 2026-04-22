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
