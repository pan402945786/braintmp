'use strict';

const { RNG } = require('./predictive_growth_experiment');

const FIRST = 16;
const LAST = 40;
const CLASSES = LAST - FIRST + 1;
const FRAMES = 48;
const SIDE = 32;
const FLOOR = 29;
const RADIUS = 2;
const PIXELS = 256;

function stream(seed, ...keys) {
  let state = seed >>> 0;
  for (const key of keys) {
    state = Math.imul(state ^ key, 0x9e3779b1);
    state ^= state >>> 16;
    state = Math.imul(state, 0x85ebca6b);
    state ^= state >>> 13;
  }
  return new RNG(state);
}

function shuffle(items, rng) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = rng.int(0, i);
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

function height(scene, t) {
  return scene.y + scene.v * t + (scene.gravity / 2) * t * t;
}

function sceneAt(seed, split, index, regime = 0, marked = false) {
  const cutoff = stream(seed, split, index, 41, regime).int(4, 12);
  const motion = stream(seed, split, index, 42, regime);
  const appearance = stream(seed, split, index, 43, regime);
  const target = motion.int(FIRST, LAST);
  const contactTime = target - motion.uniform(0.05, 0.95);
  const gravity = regime ? 0.052 : 0.035;
  // Condition the kinematics on the chosen timing stratum, then detect contact independently.
  const y = motion.uniform(9, 16);
  const v = (FLOOR - RADIUS - y) / contactTime - (gravity / 2) * contactTime;
  const scene = { y, v, gravity, x: motion.uniform(8, 24), vx: motion.uniform(-0.1, 0.1),
    background: appearance.uniform(0.06, 0.22), brightness: appearance.uniform(0.6, 0.78),
    texture: appearance.next(), marker: marked ? regime : null, cutoff };
  scene.contact = Array.from({ length: FRAMES }, (_, t) => t).find(t => height(scene, t) + RADIUS >= FLOOR);
  return scene;
}

function physicalOracle(scene) {
  const discriminant = scene.v ** 2 + 2 * scene.gravity * (FLOOR - RADIUS - scene.y);
  return Math.ceil((-scene.v + Math.sqrt(discriminant)) / scene.gravity);
}

function render(scene, through = FRAMES - 1) {
  const frames = [];
  for (let t = 0; t <= through; t++) {
    const pixels = new Float32Array(SIDE * SIDE);
    const x = scene.x + scene.vx * Math.min(t, scene.contact);
    const y = Math.min(FLOOR - RADIUS, height(scene, Math.min(t, scene.contact)));
    for (let row = 0; row < SIDE; row++) {
      for (let col = 0; col < SIDE; col++) {
        const texture = 0.025 * Math.sin(row * 0.7 + col * 0.5 + scene.texture * 6.28);
        const d2 = (col + 0.5 - x) ** 2 + (row + 0.5 - y) ** 2;
        const ball = d2 < 25 ? scene.brightness * Math.exp(-d2 / 3.2) : 0;
        const marker = scene.marker === 0 && col < 4 && row < 3 || scene.marker === 1 && col >= 28 && row < 3;
        pixels[row * SIDE + col] = Math.min(1, scene.background + texture + ball + (row >= FLOOR ? 0.12 : 0) + (marker ? 0.5 : 0));
      }
    }
    frames.push({ pixels, audio: t === scene.contact ? 1 : t === scene.contact + 1 ? 0.35 : 0 });
  }
  return frames;
}

const HIERARCHICAL = ['hierarchical', 'hierarchical-delta', 'hierarchical-motion', 'hierarchical-feedback', 'hierarchical-attention'];

// Shared fixed local filters, with edge replication and 2x2 aggregation.
function hierarchicalEncode(frames, temporal) {
  let previous = null;
  return frames.map(({ pixels, audio }) => {
    const small = new Float64Array(256);
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const k = y * 64 + x * 2;
      small[y * 16 + x] = (pixels[k] + pixels[k + 1] + pixels[k + 32] + pixels[k + 33]) / 4;
    }
    const count = temporal ? 6 : 5;
    const result = new Float64Array(count * 64 + 1);
    const at = (y, x) => small[Math.max(0, Math.min(15, y)) * 16 + Math.max(0, Math.min(15, x))];
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const center = at(y, x);
      let surround = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) surround += at(y + dy, x + dx);
      const contrast = center - surround / 9;
      const values = [center, Math.max(0, contrast), Math.max(0, -contrast),
        (at(y, x + 1) - at(y, x - 1)) / 2, (at(y + 1, x) - at(y - 1, x)) / 2];
      if (temporal) values.push(previous ? center - previous[y * 16 + x] : 0);
      const dest = Math.floor(y / 2) * 8 + Math.floor(x / 2);
      values.forEach((v, channel) => { result[channel * 64 + dest] += v / 4; });
    }
    result[count * 64] = audio;
    previous = small;
    return result;
  });
}

function featureCount(encoder) {
  if (HIERARCHICAL.includes(encoder)) return (encoder === 'hierarchical' ? 5 : 6) * 64 + 1;
  return PIXELS + (encoder === 'motion' ? 4 : 1);
}

function encode(frames, encoder) {
  if (HIERARCHICAL.includes(encoder)) return hierarchicalEncode(frames, encoder !== 'hierarchical');
  let firstY;
  let lastY;
  return frames.map(({ pixels, audio }, t) => {
    const features = new Float64Array(featureCount(encoder));
    let background = 0;
    for (let row = 0; row < 28; row++) background += pixels[row * SIDE] + pixels[row * SIDE + 31];
    background /= 56;
    for (let row = 0; row < 16; row++) {
      for (let col = 0; col < 16; col++) {
        const k = row * 64 + col * 2;
        features[row * 16 + col] = (pixels[k] + pixels[k + 1] + pixels[k + 32] + pixels[k + 33]) / 4 - background;
      }
    }
    features[PIXELS] = audio;
    if (encoder === 'motion') {
      let mass = 0;
      let ySum = 0;
      for (let row = 3; row < FLOOR; row++) {
        for (let col = 2; col < 30; col++) {
          const contrast = Math.max(0, pixels[row * SIDE + col] - background - 0.05);
          mass += contrast;
          ySum += (row + 0.5) * contrast;
        }
      }
      const y = mass > 0 ? ySum / mass : 16;
      if (t === 0) firstY = y;
      features[PIXELS + 1] = (y - 14) / 8;
      features[PIXELS + 2] = t ? (y - firstY) / t : 0;
      features[PIXELS + 3] = t ? y - lastY : 0;
      lastY = y;
    }
    return features;
  });
}

function makeEvent(seed, split, index, encoder, regime = 0, marked = false) {
  const scene = sceneAt(seed, split, index, regime, marked);
  const frames = render(scene, scene.contact);
  const feedback = frames.findIndex(frame => frame.audio > 0);
  const pick = stream(seed, split, index, 44, regime);
  const early = shuffle(Array.from({ length: 9 }, (_, i) => i + 4), pick).slice(0, 2);
  const late = pick.int(13, feedback - 1);
  return { cutoff: scene.cutoff, target: scene.contact - FIRST, feedback: feedback - FIRST,
    prefixes: [...early, late].sort((a, b) => a - b), features: encode(frames, encoder), regime };
}

module.exports = { FIRST, LAST, CLASSES, FRAMES, FLOOR, RADIUS, stream, shuffle, height,
  sceneAt, physicalOracle, render, featureCount, encode, makeEvent };
