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
uvicorn app.main:app --reload
```

브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000) 를 열면 됩니다.

## 비고

- 입력 오디오는 백엔드에서 mono / 16kHz 로 맞춘 뒤 모델에 전달합니다.
- 첫 추론 시에도 모델이 없으면 자동으로 `models/wavjepa-base` 아래로 다운로드합니다.
- 시각화 좌표는 원본 768차원 pooled embedding이 아니라 축소된 좌표입니다.
- 업로드 분석은 PCA 또는 t-SNE를 선택할 수 있고, 마이크 실시간 모드는 안정성을 위해 PCA로 고정됩니다.
- 마이크 기능은 브라우저 권한이 필요하며 `localhost` 또는 로컬 주소에서 실행해야 합니다.
- 로컬 머신에서 CPU만 사용할 경우 첫 추론과 다중 파일 처리 시간이 길 수 있습니다.
