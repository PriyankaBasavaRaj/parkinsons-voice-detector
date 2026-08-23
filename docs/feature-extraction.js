/**
 * Client-side acoustic feature extraction.
 *
 * Implements a simplified autocorrelation-based pitch tracker to estimate
 * the same category of features the model was trained on:
 *   - MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)  -> mean/max/min F0
 *   - MDVP:Jitter(%)                            -> cycle-to-cycle F0 variation
 *   - MDVP:Shimmer                              -> cycle-to-cycle amplitude variation
 *   - HNR                                       -> harmonic-to-noise ratio (dB), approximated
 *
 * This is a simplified DSP implementation for a portfolio demo, not a
 * clinical-grade voice analysis tool (see README for details on what's
 * approximated here vs. what tools like Praat compute).
 */

const FeatureExtractor = (() => {
  const MIN_F0 = 75;   // Hz, lower bound of typical speaking voice
  const MAX_F0 = 500;  // Hz, upper bound

  function autocorrelate(buffer, sampleRate) {
    const n = buffer.length;
    const minLag = Math.floor(sampleRate / MAX_F0);
    const maxLag = Math.floor(sampleRate / MIN_F0);

    // Remove DC offset
    let mean = 0;
    for (let i = 0; i < n; i++) mean += buffer[i];
    mean /= n;
    const b = new Float32Array(n);
    for (let i = 0; i < n; i++) b[i] = buffer[i] - mean;

    let bestLag = -1;
    let bestCorr = 0;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let corr = 0;
      for (let i = 0; i < n - lag; i++) corr += b[i] * b[i + lag];
      if (corr > bestCorr) {
        bestCorr = corr;
        bestLag = lag;
      }
    }
    if (bestLag <= 0) return null;
    return sampleRate / bestLag;
  }

  function rms(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
    return Math.sqrt(sum / buffer.length);
  }

  /**
   * Split audio into overlapping frames, estimate F0 and amplitude per
   * frame, then derive jitter/shimmer/HNR-style stats across frames.
   */
  function extract(channelData, sampleRate) {
    const frameSize = Math.round(sampleRate * 0.04); // 40ms frames
    const hopSize = Math.round(frameSize / 2);        // 50% overlap

    const f0s = [];
    const amps = [];

    for (let start = 0; start + frameSize <= channelData.length; start += hopSize) {
      const frame = channelData.subarray(start, start + frameSize);
      const amp = rms(frame);
      if (amp < 0.01) continue; // skip near-silence
      const f0 = autocorrelate(frame, sampleRate);
      if (f0 && f0 >= MIN_F0 && f0 <= MAX_F0) {
        f0s.push(f0);
        amps.push(amp);
      }
    }

    if (f0s.length < 5) {
      throw new Error(
        "Couldn't detect enough voiced sound in that clip. Try a longer, steady 'ahhh' at a comfortable volume."
      );
    }

    const meanF0 = f0s.reduce((a, b) => a + b, 0) / f0s.length;
    const maxF0 = Math.max(...f0s);
    const minF0 = Math.min(...f0s);

    // Jitter: average absolute cycle-to-cycle F0 difference, as % of mean F0
    let jitterSum = 0;
    for (let i = 1; i < f0s.length; i++) jitterSum += Math.abs(f0s[i] - f0s[i - 1]);
    const jitterPercent = (jitterSum / (f0s.length - 1) / meanF0) * 100;

    // Shimmer: average absolute cycle-to-cycle amplitude difference, as
    // a proportion of mean amplitude
    const meanAmp = amps.reduce((a, b) => a + b, 0) / amps.length;
    let shimmerSum = 0;
    for (let i = 1; i < amps.length; i++) shimmerSum += Math.abs(amps[i] - amps[i - 1]);
    const shimmer = shimmerSum / (amps.length - 1) / meanAmp;

    // HNR approximation: ratio of frame-to-frame F0 stability to spread,
    // expressed in dB-like units. This is a rough stand-in for true
    // harmonic-to-noise ratio (which needs spectral analysis), calibrated
    // against typical dataset ranges (~10-30 dB).
    const f0Variance =
      f0s.reduce((sum, f) => sum + (f - meanF0) ** 2, 0) / f0s.length;
    const f0Std = Math.sqrt(f0Variance);
    const stability = meanF0 / (f0Std + 1); // higher = more stable voice
    const hnr = Math.min(35, Math.max(5, 10 * Math.log10(stability) + 10));

    return {
      "MDVP:Fo(Hz)": meanF0,
      "MDVP:Fhi(Hz)": maxF0,
      "MDVP:Flo(Hz)": minF0,
      "MDVP:Jitter(%)": jitterPercent,
      "MDVP:Shimmer": shimmer,
      "HNR": hnr,
      "_frameCount": f0s.length,
      "_f0Series": f0s,
    };
  }

  /** Decode an audio Blob/File into a mono Float32Array + sample rate. */
  async function decodeAudio(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0); // first channel
    const sampleRate = audioBuffer.sampleRate;
    audioCtx.close();
    return { channelData, sampleRate };
  }

  return { extract, decodeAudio };
})();
