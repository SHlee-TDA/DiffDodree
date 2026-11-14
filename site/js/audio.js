// A small WebAudio voice for listening to pieces (a reed-like timbre, not a 피리 recording).
export class Player {
  constructor() {
    this.ctx = null;
    this.nodes = [];
    this.startedAt = 0;
    this.timeline = null;
  }
  ensure() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.22;
      const comp = this.ctx.createDynamicsCompressor();
      this.master.connect(comp).connect(this.ctx.destination);
    }
    return this.ctx;
  }
  // phrases: [[ [pitch|null, beats], ... ], ...]; returns measure start times (s) from the start
  timelineOf(phrases, bpm, from = 0) {
    const spb = 60 / bpm;
    const starts = [];
    let t = 0;
    phrases.forEach((events, m) => {
      starts.push(t);
      if (m >= from) for (const [, d] of events) t += d * spb;
      else starts[m] = 0;
    });
    const total = t;
    return { starts, total, spb, from };
  }
  play(phrases, bpm, from = 0) {
    this.stop();
    const ctx = this.ensure();
    if (ctx.state === "suspended") ctx.resume();
    const tl = this.timelineOf(phrases, bpm, from);
    const t0 = ctx.currentTime + 0.08;
    let t = t0;
    for (let m = from; m < phrases.length; m++) {
      for (const [pitch, beats] of phrases[m]) {
        const dur = beats * tl.spb;
        if (pitch !== null) this.note(pitch, t, dur);
        t += dur;
      }
    }
    this.startedAt = t0;
    this.timeline = tl;
    this.beat0 = phrases.slice(0, from).reduce((a, ev) => a + ev.reduce((x, [, d]) => x + d, 0), 0);
    this.beats = phrases.reduce((a, ev) => a + ev.reduce((x, [, d]) => x + d, 0), 0);
  }
  // beats from the start of the piece at the current audio time, or null when not playing
  beat() {
    if (!this.timeline || !this.ctx) return null;
    const b = this.beat0 + Math.max(0, this.ctx.currentTime - this.startedAt) / this.timeline.spb;
    return b >= this.beats ? null : b;
  }
  note(midi, at, dur) {
    const ctx = this.ctx;
    const f = 440 * Math.pow(2, (midi - 69) / 12);
    const osc1 = ctx.createOscillator(), osc2 = ctx.createOscillator();
    osc1.type = "sawtooth"; osc2.type = "square";
    osc1.frequency.value = f; osc2.frequency.value = f * 1.003;
    const lfo = ctx.createOscillator(), lfoGain = ctx.createGain();
    lfo.frequency.value = 5.2; lfoGain.gain.value = 0;
    lfoGain.gain.setValueAtTime(0, at);
    lfoGain.gain.linearRampToValueAtTime(f * 0.012, at + Math.min(0.35, dur * 0.6));
    lfo.connect(lfoGain); lfoGain.connect(osc1.frequency); lfoGain.connect(osc2.frequency);
    const mix2 = ctx.createGain(); mix2.gain.value = 0.35;
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass"; filter.frequency.value = Math.min(3200, f * 5); filter.Q.value = 2.5;
    const env = ctx.createGain();
    const end = at + Math.max(dur - 0.015, 0.03);
    env.gain.setValueAtTime(0, at);
    env.gain.linearRampToValueAtTime(1, at + 0.035);
    env.gain.setValueAtTime(1, Math.max(at + 0.035, end - 0.05));
    env.gain.linearRampToValueAtTime(0, end);
    osc1.connect(filter); osc2.connect(mix2).connect(filter);
    filter.connect(env).connect(this.master);
    for (const o of [osc1, osc2, lfo]) { o.start(at); o.stop(end + 0.02); }
    this.nodes.push(osc1, osc2, lfo);
  }
  stop() {
    for (const n of this.nodes) { try { n.stop(); } catch { /* already stopped */ } }
    this.nodes = [];
    this.timeline = null;
  }
  // {measure, fraction} at the current audio time, or null when not playing
  position() {
    if (!this.timeline || !this.ctx) return null;
    const { starts, total, from } = this.timeline;
    const e = this.ctx.currentTime - this.startedAt;
    if (e < 0) return { measure: from, fraction: 0 };
    if (e >= total) return null;
    let m = from;
    while (m + 1 < starts.length && starts[m + 1] <= e) m++;
    const next = m + 1 < starts.length ? starts[m + 1] : total;
    return { measure: m, fraction: (e - starts[m]) / Math.max(next - starts[m], 1e-9) };
  }
}
