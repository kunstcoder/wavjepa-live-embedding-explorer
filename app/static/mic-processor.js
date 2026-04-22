class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];

    if (input && input[0] && input[0].length) {
      this.port.postMessage(input[0].slice());
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
