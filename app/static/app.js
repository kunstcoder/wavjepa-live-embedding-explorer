function createLiveState() {
  return {
    active: false,
    sessionId: null,
    stream: null,
    audioContext: null,
    sourceNode: null,
    workletNode: null,
    sinkGain: null,
    buffers: [],
    bufferedSamples: 0,
    sampleRate: 0,
    chunkIndex: 0,
    sentSamples: 0,
    flushTimer: null,
    requestInFlight: false,
    skippedChunks: 0,
  };
}

const state = {
  files: [],
  response: null,
  health: null,
  live: createLiveState(),
};

const palette = [
  "#e66f32",
  "#176c68",
  "#b4851d",
  "#7d4ef0",
  "#1d4cb8",
  "#cc5566",
  "#0b7a44",
  "#9f5c1d",
];

const elements = {
  analyzeButton: document.getElementById("analyzeButton"),
  cacheMetric: document.getElementById("cacheMetric"),
  chunkDurationSelect: document.getElementById("chunkDurationSelect"),
  clearLiveButton: document.getElementById("clearLiveButton"),
  deviceMetric: document.getElementById("deviceMetric"),
  dimensionSelect: document.getElementById("dimensionSelect"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  fileList: document.getElementById("fileList"),
  fileMetric: document.getElementById("fileMetric"),
  levelGateSlider: document.getElementById("levelGateSlider"),
  levelGateValue: document.getElementById("levelGateValue"),
  liveBadge: document.getElementById("liveBadge"),
  micButton: document.getElementById("micButton"),
  methodSelect: document.getElementById("methodSelect"),
  plot: document.getElementById("plot"),
  plotSummary: document.getElementById("plotSummary"),
  projectionMetric: document.getElementById("projectionMetric"),
  statusCard: document.getElementById("statusCard"),
};

function setStatus(text, tone = "idle") {
  elements.statusCard.textContent = text;
  elements.statusCard.className = `status-card ${tone}`;
}

function setLiveBadge(text, tone = "idle") {
  elements.liveBadge.textContent = text;
  elements.liveBadge.className = `live-badge ${tone}`;
}

function syncFileMetric() {
  elements.fileMetric.textContent = String(state.files.length);
}

function getColor(index) {
  return palette[index % palette.length];
}

function dbfsToAmplitude(dbfs) {
  return 10 ** (dbfs / 20);
}

function formatRms(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }

  if (value >= 0.1) {
    return value.toFixed(3);
  }

  if (value >= 0.01) {
    return value.toFixed(4);
  }

  return value.toFixed(6);
}

function syncLevelGateUI() {
  const dbfs = Number(elements.levelGateSlider.value);
  const rms = dbfsToAmplitude(dbfs);
  elements.levelGateValue.textContent = `${dbfs.toFixed(0)} dBFS · RMS ${formatRms(rms)}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function resetProjectionView() {
  state.response = null;
  elements.plotSummary.textContent = "분석할 파일을 업로드하면 산점도를 렌더링합니다.";
  elements.projectionMetric.textContent = "-";

  if (window.Plotly) {
    Plotly.purge(elements.plot);
  }
}

function resetLiveBuffers() {
  state.live.buffers = [];
  state.live.bufferedSamples = 0;
  state.live.chunkIndex = 0;
  state.live.sentSamples = 0;
}

function renderFileList() {
  if (!state.files.length && !state.response) {
    elements.fileList.className = "file-list empty";
    elements.fileList.textContent = "아직 선택된 파일이 없습니다.";
    return;
  }

  if (!state.response) {
    const names = state.files.map((file) => file.name).join(", ");
    elements.fileList.className = "file-list empty";
    elements.fileList.textContent = `선택됨: ${names}`;
    return;
  }

  elements.fileList.className = "file-list";
  elements.fileList.innerHTML = state.response.points
    .map((point, index) => {
      const liveMeta =
        point.mode === "live"
          ? `<span>t=${Number(point.elapsedSeconds).toFixed(1)}s</span><span>chunk=${point.chunkIndex}</span>`
          : "";

      return `
        <article class="file-item">
          <div class="file-header">
            <strong>${escapeHtml(point.label)}</strong>
            <span class="color-chip" style="background:${getColor(index)}"></span>
          </div>
          <div class="file-meta">
            <span>coords=${point.coordinates.map((value) => value.toFixed(3)).join(", ")}</span>
            <span>duration=${point.durationSeconds}s</span>
            <span>steps=${point.temporalSteps}</span>
            <span>dim=${point.embeddingDim}</span>
            <span>norm=${point.pooledNorm}</span>
            <span>rms=${point.rmsEnergy}</span>
            ${liveMeta}
          </div>
        </article>
      `;
    })
    .join("");
}

function buildTrace(points, dimensions, isLive) {
  const colors = points.map((_, index) => getColor(index));
  const labels = points.map((point) => point.label);

  if (dimensions === 3) {
    return {
      type: "scatter3d",
      mode: isLive ? "lines+markers" : "markers+text",
      x: points.map((point) => point.coordinates[0]),
      y: points.map((point) => point.coordinates[1]),
      z: points.map((point) => point.coordinates[2]),
      text: labels,
      textposition: "top center",
      hovertemplate:
        "<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
      marker: {
        size: isLive ? 5 : 6,
        color: colors,
        opacity: 0.92,
      },
      line: isLive
        ? {
            color: "rgba(23, 108, 104, 0.35)",
            width: 4,
          }
        : undefined,
    };
  }

  return {
    type: "scattergl",
    mode: isLive ? "lines+markers" : "markers+text",
    x: points.map((point) => point.coordinates[0]),
    y: points.map((point) => point.coordinates[1]),
    text: labels,
    textposition: "top center",
    hovertemplate:
      "<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
    marker: {
      size: isLive ? 10 : 14,
      color: colors,
      opacity: 0.92,
      line: {
        width: 1,
        color: "rgba(31, 30, 23, 0.18)",
      },
    },
    line: isLive
      ? {
          color: "rgba(23, 108, 104, 0.35)",
          width: 3,
        }
      : undefined,
  };
}

function renderPlot() {
  if (!state.response || !window.Plotly) {
    return;
  }

  const { dimensions, effectiveMethod, points, pointCount, live } = state.response;
  const trace = buildTrace(points, dimensions, Boolean(live));
  const title = live
    ? `Realtime PCA ${dimensions}D Trajectory`
    : `${effectiveMethod.toUpperCase()} ${dimensions}D Projection`;

  const layout =
    dimensions === 3
      ? {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 0, r: 0, b: 0, t: 48 },
          title: { text: title, font: { family: "Space Grotesk, sans-serif", size: 18 } },
          scene: {
            bgcolor: "rgba(255,255,255,0.18)",
            xaxis: { title: "Component 1", gridcolor: "rgba(0,0,0,0.08)" },
            yaxis: { title: "Component 2", gridcolor: "rgba(0,0,0,0.08)" },
            zaxis: { title: "Component 3", gridcolor: "rgba(0,0,0,0.08)" },
            camera: { eye: { x: 1.35, y: 1.25, z: 0.95 } },
          },
          showlegend: false,
        }
      : {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 52, r: 24, b: 52, t: 48 },
          title: { text: title, font: { family: "Space Grotesk, sans-serif", size: 18 } },
          xaxis: {
            title: "Component 1",
            zeroline: false,
            gridcolor: "rgba(0,0,0,0.08)",
          },
          yaxis: {
            title: "Component 2",
            zeroline: false,
            gridcolor: "rgba(0,0,0,0.08)",
          },
          showlegend: false,
        };

  Plotly.react(elements.plot, [trace], layout, {
    displayModeBar: false,
    responsive: true,
  });

  elements.plotSummary.textContent = live
    ? `${pointCount}개 마이크 청크를 실시간 PCA ${dimensions}D trajectory로 표시 중입니다.`
    : `${pointCount}개 파일을 ${effectiveMethod.toUpperCase()} ${dimensions}D 공간에 배치했습니다.`;
  elements.projectionMetric.textContent = `${effectiveMethod.toUpperCase()} ${dimensions}D`;
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    state.health = health;

    elements.deviceMetric.textContent = health.device;
    elements.cacheMetric.textContent = health.modelCached ? "ready" : "download on first run";
    setStatus(
      health.modelCached
        ? "모델 캐시가 준비되어 있습니다."
        : "모델 캐시가 없습니다. 첫 추론 시 Hugging Face snapshot을 다운로드합니다.",
      "ready",
    );
  } catch (error) {
    console.error(error);
    setStatus("서버 상태 확인에 실패했습니다. 백엔드가 실행 중인지 확인하세요.", "error");
  }
}

function attachDropzone() {
  const assignFiles = (fileList) => {
    state.files = Array.from(fileList);
    resetProjectionView();
    syncFileMetric();
    renderFileList();
  };

  elements.dropzone.addEventListener("click", () => elements.fileInput.click());
  elements.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.fileInput.click();
    }
  });

  elements.fileInput.addEventListener("change", (event) => {
    assignFiles(event.target.files);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("dragover");
    });
  });

  elements.dropzone.addEventListener("drop", (event) => {
    assignFiles(event.dataTransfer.files);
  });
}

async function createLiveSession() {
  const response = await fetch("/api/live-sessions", { method: "POST" });
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Failed to create live session.");
  }

  return payload.sessionId;
}

async function deleteLiveSession(sessionId) {
  if (!sessionId) {
    return;
  }

  try {
    await fetch(`/api/live-sessions/${sessionId}`, { method: "DELETE" });
  } catch (error) {
    console.warn("Failed to delete live session", error);
  }
}

function getLiveChunkSeconds() {
  return Number(elements.chunkDurationSelect.value);
}

function getLiveChunkSampleCount() {
  return Math.floor(state.live.sampleRate * getLiveChunkSeconds());
}

function getLiveMinRmsEnergy() {
  return dbfsToAmplitude(Number(elements.levelGateSlider.value));
}

function pushAudioBuffer(buffer) {
  if (!state.live.active || !buffer.length) {
    return;
  }

  state.live.buffers.push(buffer);
  state.live.bufferedSamples += buffer.length;

  const maxBufferedSamples = Math.floor(state.live.sampleRate * getLiveChunkSeconds() * 6);

  while (state.live.bufferedSamples > maxBufferedSamples && state.live.buffers.length) {
    const dropped = state.live.buffers.shift();
    state.live.bufferedSamples -= dropped.length;
    state.live.sentSamples += dropped.length;
  }
}

function consumeLiveSamples(sampleCount) {
  const output = new Float32Array(sampleCount);
  let offset = 0;

  while (offset < sampleCount && state.live.buffers.length) {
    const head = state.live.buffers[0];
    const remaining = sampleCount - offset;

    if (head.length <= remaining) {
      output.set(head, offset);
      offset += head.length;
      state.live.buffers.shift();
    } else {
      output.set(head.subarray(0, remaining), offset);
      state.live.buffers[0] = head.subarray(remaining);
      offset += remaining;
    }
  }

  state.live.bufferedSamples = Math.max(0, state.live.bufferedSamples - sampleCount);
  return output;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, text) {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;

  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += bytesPerSample;
  }

  return buffer;
}

async function startMicrophone() {
  if (state.live.active) {
    await stopMicrophone();
    return;
  }

  if (!window.isSecureContext) {
    setStatus("원격 브라우저 마이크는 HTTPS 또는 localhost에서만 허용됩니다.", "error");
    setLiveBadge("https required", "error");
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("이 브라우저는 마이크 스트리밍을 지원하지 않습니다.", "error");
    setLiveBadge("mic unsupported", "error");
    return;
  }

  state.files = [];
  syncFileMetric();
  resetProjectionView();
  renderFileList();

  let sessionId = null;

  try {
    sessionId = await createLiveSession();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule("/static/mic-processor.js");
    await audioContext.resume();

    const sourceNode = audioContext.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
    const sinkGain = audioContext.createGain();
    sinkGain.gain.value = 0;

    workletNode.port.onmessage = (event) => {
      pushAudioBuffer(new Float32Array(event.data));
    };

    sourceNode.connect(workletNode);
    workletNode.connect(sinkGain);
    sinkGain.connect(audioContext.destination);

    state.live = {
      active: true,
      sessionId,
      stream,
      audioContext,
      sourceNode,
      workletNode,
      sinkGain,
      buffers: [],
      bufferedSamples: 0,
      sampleRate: audioContext.sampleRate,
      chunkIndex: 0,
      sentSamples: 0,
      flushTimer: window.setInterval(flushLiveChunk, 250),
      requestInFlight: false,
      skippedChunks: 0,
    };

    elements.micButton.textContent = "마이크 중지";
    setLiveBadge("mic on · buffering", "live");
    setStatus(
      `마이크 입력을 수집 중입니다. ${elements.levelGateSlider.value} dBFS보다 작은 청크는 제외합니다.`,
      "busy",
    );
  } catch (error) {
    console.error(error);
    await deleteLiveSession(sessionId);
    state.live = createLiveState();
    setLiveBadge("mic unavailable", "error");
    setStatus(`마이크 시작 실패: ${error.message}`, "error");
  }
}

async function stopMicrophone() {
  const sessionId = state.live.sessionId;

  if (state.live.flushTimer) {
    window.clearInterval(state.live.flushTimer);
  }

  if (state.live.workletNode) {
    state.live.workletNode.port.onmessage = null;
    state.live.workletNode.disconnect();
  }

  if (state.live.sourceNode) {
    state.live.sourceNode.disconnect();
  }

  if (state.live.sinkGain) {
    state.live.sinkGain.disconnect();
  }

  if (state.live.stream) {
    state.live.stream.getTracks().forEach((track) => track.stop());
  }

  if (state.live.audioContext) {
    try {
      await state.live.audioContext.close();
    } catch (error) {
      console.warn("Failed to close audio context", error);
    }
  }

  state.live = createLiveState();
  elements.micButton.textContent = "마이크 시작";
  setLiveBadge("mic off", "idle");

  await deleteLiveSession(sessionId);
}

async function flushLiveChunk() {
  if (!state.live.active || state.live.requestInFlight || !state.live.sessionId) {
    return;
  }

  const sampleCount = getLiveChunkSampleCount();

  if (!sampleCount || state.live.bufferedSamples < sampleCount) {
    return;
  }

  const sessionId = state.live.sessionId;
  const chunkIndex = state.live.chunkIndex;
  const elapsedSeconds = state.live.sentSamples / state.live.sampleRate;
  const samples = consumeLiveSamples(sampleCount);

  state.live.chunkIndex += 1;
  state.live.sentSamples += samples.length;
  state.live.requestInFlight = true;

  try {
    const formData = new FormData();
    const wavBuffer = encodeWav(samples, state.live.sampleRate);
    formData.append("file", new Blob([wavBuffer], { type: "audio/wav" }), `mic-${chunkIndex}.wav`);
    formData.append("dimensions", elements.dimensionSelect.value);
    formData.append("chunk_index", String(chunkIndex));
    formData.append("elapsed_seconds", elapsedSeconds.toFixed(3));
    formData.append("min_rms_energy", getLiveMinRmsEnergy().toFixed(8));

    const response = await fetch(`/api/live-sessions/${sessionId}/chunks`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (sessionId !== state.live.sessionId) {
      return;
    }

    if (!response.ok) {
      throw new Error(payload.detail || "Failed to process live audio chunk.");
    }

    if (payload.skipped) {
      state.live.skippedChunks += 1;

      const skippedPointCount = Number.isFinite(payload.pointCount) ? payload.pointCount : 0;
      const badgeSuffix = state.live.skippedChunks ? ` · ${state.live.skippedChunks} skip` : "";

      setLiveBadge(`mic on · ${skippedPointCount} pts${badgeSuffix}`, "live");
      setStatus(
        `낮은 레벨 청크를 제외했습니다. rms=${formatRms(payload.rmsEnergy)} < threshold=${formatRms(payload.minRmsEnergy)}`,
        "busy",
      );
      return;
    }

    state.response = payload;
    const badgeSuffix = state.live.skippedChunks ? ` · ${state.live.skippedChunks} skip` : "";
    renderPlot();
    renderFileList();
    setLiveBadge(`mic on · ${payload.pointCount} pts${badgeSuffix}`, "live");
    setStatus(`실시간 업데이트: ${payload.pointCount}개 청크 누적`, "ready");
  } catch (error) {
    console.error(error);
    await stopMicrophone();
    setLiveBadge("mic error", "error");
    setStatus(`실시간 분석 실패: ${error.message}`, "error");
  } finally {
    if (sessionId === state.live.sessionId) {
      state.live.requestInFlight = false;
    }
  }
}

async function clearLiveCapture() {
  await stopMicrophone();
  resetProjectionView();
  renderFileList();
  setStatus("실시간 세션을 초기화했습니다.", "ready");
}

async function analyze() {
  if (state.live.active) {
    setStatus("배치 업로드 분석 전에는 마이크 실시간 모드를 중지하세요.", "error");
    return;
  }

  if (!state.files.length) {
    setStatus("먼저 하나 이상의 오디오 파일을 선택하세요.", "error");
    return;
  }

  const formData = new FormData();
  state.files.forEach((file) => formData.append("files", file));
  formData.append("method", elements.methodSelect.value);
  formData.append("dimensions", elements.dimensionSelect.value);

  elements.analyzeButton.disabled = true;
  setStatus("임베딩을 추출하고 투영 공간을 계산하는 중입니다. 첫 요청은 모델 다운로드 때문에 오래 걸릴 수 있습니다.", "busy");

  try {
    const response = await fetch("/api/embeddings", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Unknown API error");
    }

    state.response = payload;
    renderPlot();
    renderFileList();
    setStatus("시각화가 갱신되었습니다.", "ready");
  } catch (error) {
    console.error(error);
    setStatus(`분석 실패: ${error.message}`, "error");
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

elements.analyzeButton.addEventListener("click", analyze);
elements.micButton.addEventListener("click", startMicrophone);
elements.clearLiveButton.addEventListener("click", clearLiveCapture);
elements.levelGateSlider.addEventListener("input", syncLevelGateUI);

attachDropzone();
syncFileMetric();
syncLevelGateUI();
renderFileList();
setLiveBadge("mic off", "idle");
fetchHealth();

window.addEventListener("beforeunload", () => {
  if (state.live.active) {
    stopMicrophone();
  }
});
