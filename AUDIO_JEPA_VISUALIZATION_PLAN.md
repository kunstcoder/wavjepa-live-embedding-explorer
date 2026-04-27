# Audio-JEPA 임베딩 시각화 비교 작업 계획서

## 결론

구현 가능하다. 현재 앱의 WavJEPA 시각화 흐름은 `오디오 로드 -> 모델별 frame embedding -> mean pooling -> PCA/t-SNE projection -> Plotly 렌더링` 구조라서, Audio-JEPA도 같은 형태의 `EmbeddingSummary`를 반환하는 서비스만 추가하면 같은 시각화 파이프라인을 재사용할 수 있다.

다만 Audio-JEPA는 WavJEPA와 입력/모델 구조가 다르므로 기존 `WAVJEPA_MODEL_SOURCE` 및 WavJEPA 전용 `.ckpt -> HF safetensors` 변환 코드를 그대로 확장하지 않는다. Audio-JEPA 전용 ckpt 로더와 spectrogram 전처리를 별도 서비스로 구현한다.

## 검토 근거

- 로컬 WavJEPA 앱
  - `app/services/wavjepa.py`: `embed_waveform()`에서 frame embedding을 mean pooling하고 `project_embeddings()`로 PCA/t-SNE 좌표를 만든다.
  - `app/main.py`: `/api/embeddings`가 업로드 파일별 embedding을 모아 projection response를 생성한다.
  - `app/static/app.js`: 응답의 `points[].coordinates`를 Plotly 2D/3D scatter로 렌더링한다.
- Audio-JEPA 저장소
  - GitHub: <https://github.com/LudovicTuncay/Audio-JEPA>
  - 핵심 모델은 Lightning `JEPAModule`이고 `forward()`는 `self.encoder(x)`를 반환한다.
  - encoder config는 ViT base 구조다: patch size `16x16`, `embed_dim=768`, `depth=12`, `num_heads=12`.
  - 데이터 config는 AudioSet `sr=32000`, `clip_length=10`초 기반이다.
- X-ARES fork
  - GitHub: <https://github.com/LudovicTuncay/xares>
  - `example/Audio-JEPA/Audio-JEPA_encoder.py`가 Audio-JEPA용 X-ARES encoder wrapper를 제공한다.
  - wrapper는 32 kHz waveform을 10초 chunk로 자르고, `128` mel x `256` time-bin log-mel spectrogram을 만든 뒤 ViT encoder output을 `16`개 time patch embedding으로 변환한다.
  - output은 `[B, T, 768]`이며 frequency patch 8개를 평균내고, 최종 clip embedding은 현 WavJEPA처럼 time축 mean pooling으로 만들 수 있다.
- pretrained 확인
  - Hugging Face 모델: <https://huggingface.co/ltuncay/Audio-JEPA>
  - API 응답 기준 public 모델이며 siblings에 `JEPA.ckpt`가 있다.
  - 직접 HEAD 확인한 파일: <https://huggingface.co/ltuncay/Audio-JEPA/resolve/main/JEPA.ckpt>
  - 파일 크기: `728,081,348` bytes, HF repo commit `d430e4d32d27d22f1f0b1b5853711605129539ff`.

## 비교 방식

기본 비교 방식은 side-by-side projection이다.

- 같은 업로드 파일을 WavJEPA와 Audio-JEPA로 각각 임베딩한다.
- 각 모델의 embedding matrix를 같은 projection 설정(PCA 또는 t-SNE, 2D/3D)으로 축소한다.
- 두 plot에서 파일별 색상/라벨을 동일하게 유지해 클러스터링, 거리 관계, outlier를 눈으로 비교한다.

옵션으로 joint overlay projection도 구현할 수 있다.

- `2N`개의 점(`WavJEPA N개 + Audio-JEPA N개`)을 한 projection 공간에 같이 넣는다.
- 모델별 marker symbol 또는 trace를 분리한다.
- 단, 두 모델의 768차원 좌표축은 의미상 정렬된 공간이 아니므로 joint overlay는 "상대적 분리 양상" 참고용으로만 표시한다. 기본 판단은 side-by-side가 더 안전하다.

## 설계

### 1. 공통 embedding interface

`EmbeddingSummary`와 `project_embeddings()`는 모델 독립적으로 쓸 수 있으므로 유지한다.

새 추상 계층은 최소화한다.

- `WavJEPAService`: 기존 유지.
- `AudioJEPAService`: 신규 추가.
- 공통 함수는 필요할 때만 `app/services/embeddings.py` 같은 파일로 이동한다.

### 2. AudioJEPAService

역할:

- HF Hub에서 `ltuncay/Audio-JEPA`의 `JEPA.ckpt`를 내려받거나, 환경변수로 지정한 로컬 ckpt를 로드한다.
- ckpt payload에서 `state_dict`를 찾는다.
- `target_encoder.` prefix가 있으면 우선 사용하고, 없으면 `encoder.` prefix를 사용한다.
- ViT encoder에 state dict를 로드한다.
- waveform을 32 kHz mono tensor로 받아 Audio-JEPA log-mel spectrogram으로 변환한다.
- encoder output `[1, time_patches * freq_patches, 768]`을 `[time_patches, 768]`로 만들고, frequency patch를 평균낸 뒤 전체 time축 mean pooling으로 clip embedding을 만든다.

설정:

- `AUDIO_JEPA_MODEL_REPO_ID=ltuncay/Audio-JEPA`
- `AUDIO_JEPA_CKPT_FILENAME=JEPA.ckpt`
- `AUDIO_JEPA_MODEL_SOURCE=/absolute/path/to/JEPA.ckpt` optional
- cache 위치는 기존 `.hf_cache` 아래 별도 subdir를 사용한다.

### 3. Audio-JEPA 전처리

WavJEPA는 16 kHz waveform을 직접 feature extractor에 넣지만, Audio-JEPA는 32 kHz log-mel spectrogram이 입력이다.

구현 방향:

- `load_audio_from_bytes(payload, target_sample_rate=32_000)`를 AudioJEPA 경로에서 사용한다.
- `torchaudio.compliance.kaldi.fbank` 기반으로 X-ARES wrapper와 동일하게 구현한다.
- `clip_length=10`, `target_time_bins=256`, `n_mels=128`, `patch_size=(16,16)`를 기본값으로 둔다.
- 10초보다 긴 오디오는 10초 chunk 단위로 처리하고 마지막 chunk는 zero padding 후 유효 time patch만 유지한다.

### 4. attention 구현 주의

Audio-JEPA와 X-ARES wrapper는 `flash_attn.modules.mha.MHA`를 직접 import한다. 이 앱의 현재 실행 환경은 macOS/CPU/MPS 가능성이 크고, `flash-attn`은 설치/빌드 실패 가능성이 높다.

따라서 구현 시 선택지는 두 가지다.

- 우선안: checkpoint key를 확인한 뒤 `flash_attn` 없이 동작하는 PyTorch attention fallback을 만든다.
- 대안: CUDA 환경 전용으로 `flash-attn` 의존성을 허용한다.

로컬 앱 목표라면 우선안이 맞다. 구현 첫 단계에서 `JEPA.ckpt`의 attention key shape/name을 확인하고, 필요한 경우 `flash_attn.MHA`와 호환되는 parameter adapter를 작성한다.

### 5. API 설계

기존 `/api/embeddings`는 유지한다.

추가 endpoint 후보:

- `POST /api/compare-embeddings`
  - 입력: `files`, `method`, `dimensions`, `projectionMode`
  - 출력:
    - `models.wavjepa.points`
    - `models.audioJepa.points`
    - `joint.points` optional
    - 각 모델 metadata, embedding dim, temporal steps

또는 기존 `/api/embeddings`에 `model=wavjepa|audio-jepa`를 추가할 수 있다. 하지만 비교 UI를 명확히 만들려면 별도 endpoint가 더 단순하다.

### 6. 프론트엔드 설계

기존 single plot은 유지하고, 비교 모드를 추가한다.

- 모델 모드 selector:
  - `WavJEPA`
  - `Audio-JEPA`
  - `Compare`
- Compare 모드:
  - 기본: WavJEPA plot과 Audio-JEPA plot을 side-by-side 렌더링.
  - 파일별 색상은 두 plot에서 동일하게 유지.
  - hover에는 모델명, 파일명, 좌표, duration, steps, dim, norm, rms를 표시.
- optional:
  - joint overlay toggle.
  - normalized embedding norm/temporal steps 차이 표시.

## 구현 단계

1. Audio-JEPA checkpoint 조사
   - `JEPA.ckpt`를 `.hf_cache/audio_jepa/`에 다운로드한다.
   - `state_dict` key prefix와 attention module key 구조를 작은 inspection script로 확인한다.
   - `target_encoder.`와 `encoder.` 중 실제 사용 가능한 prefix를 확정한다.

2. Audio-JEPA inference module 추가
   - `app/services/audio_jepa.py` 추가.
   - X-ARES wrapper의 전처리와 chunking 로직을 앱 스타일로 이식한다.
   - `flash-attn` 의존성 없이 동작하도록 attention fallback 또는 key adapter를 구현한다.
   - `embed_waveform()`이 기존 `EmbeddingSummary`와 같은 형태를 반환하게 한다.

3. 오디오 로딩 경로 정리
   - WavJEPA는 16 kHz, Audio-JEPA는 32 kHz로 decode/resample한다.
   - 같은 업로드 파일을 두 모델에 공급할 때 중복 decode 비용이 문제되면 원본 decode 후 모델별 resample helper로 분리한다.

4. 백엔드 compare endpoint 추가
   - `/api/compare-embeddings`를 추가한다.
   - 모델별 embedding matrix를 각각 projection한다.
   - side-by-side 응답 구조와 optional joint projection 응답 구조를 정의한다.

5. UI 추가
   - 모델 모드 selector와 compare layout을 추가한다.
   - Compare 모드에서는 Plotly plot 2개를 같은 색상/라벨 매핑으로 렌더링한다.
   - 기존 단일 모델 업로드 분석 동작은 깨지지 않게 유지한다.

6. 문서 갱신
   - README에 Audio-JEPA ckpt 다운로드, 환경변수, 비교 모드, 해석상 주의점을 추가한다.
   - `flash-attn`을 요구하지 않는 로컬 실행 방식을 명시한다.

## 검증 계획

- 단위 검증
  - Audio-JEPA ckpt 로더가 `target_encoder.` 또는 `encoder.` prefix를 정상 추출하는지 확인한다.
  - 임의 waveform 입력에 대해 output shape가 `[B, T, 768]`인지 확인한다.
  - 1초, 10초, 11초, 무음 입력에서 padding/trimming이 실패하지 않는지 확인한다.
- API 검증
  - `/api/embeddings` 기존 WavJEPA 동작 회귀 확인.
  - `/api/compare-embeddings`가 같은 파일 목록에 대해 두 모델 point를 모두 반환하는지 확인.
  - PCA/t-SNE 2D/3D projection 모두 확인.
- 브라우저 검증
  - 업로드 파일 1개, 2개, 5개 이상에서 plot이 깨지지 않는지 확인.
  - Compare 모드에서 색상/라벨이 두 plot 간 일치하는지 확인.
  - 작은 viewport에서 plot과 summary text가 겹치지 않는지 확인.

## 리스크와 대응

- `flash-attn` 의존성
  - 리스크: macOS/CPU/MPS 환경에서 설치 불가.
  - 대응: checkpoint key 확인 후 PyTorch attention fallback을 우선 구현한다.
- Audio-JEPA checkpoint key mismatch
  - 리스크: X-ARES wrapper와 실제 HF `JEPA.ckpt`의 key 구조가 다를 수 있다.
  - 대응: 구현 첫 단계에서 state dict inspection을 필수로 한다.
- projection 해석
  - 리스크: WavJEPA와 Audio-JEPA의 latent axes는 정렬되어 있지 않아 joint overlay를 과해석할 수 있다.
  - 대응: 기본은 side-by-side로 제공하고, joint overlay에는 참고용 표시를 둔다.
- 처리 시간
  - 리스크: 두 모델을 동시에 돌리면 업로드 분석 시간이 늘어난다.
  - 대응: 모델별 lazy loading, progress/status message, batch 처리 최적화를 적용한다.

## 이번 작업 범위에서 하지 않을 것

- 실제 코드 구현.
- ckpt 다운로드 및 모델 로딩 테스트.
- kNN 평가 기능의 Audio-JEPA 확장.
- 실시간 마이크 trajectory의 Audio-JEPA 비교. 이 기능은 배치 업로드 비교가 안정화된 뒤 별도 단계로 진행한다.
