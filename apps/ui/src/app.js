const state = {
  aimReady: false,
  flywheelReady: false,
  testAuto: false,
};

const els = {
  log: document.getElementById('log'),
  targetX: document.getElementById('targetX'),
  targetY: document.getElementById('targetY'),
  targetConf: document.getElementById('targetConf'),
  modeBadge: document.getElementById('modeBadge'),
  yawBadge: document.getElementById('yawBadge'),
  pitchBadge: document.getElementById('pitchBadge'),
  aimBadge: document.getElementById('aimBadge'),
  wheelBadge: document.getElementById('wheelBadge'),
  fireBadge: document.getElementById('fireBadge'),
  visionSourceBadge: document.getElementById('visionSourceBadge'),
  testImageBadge: document.getElementById('testImageBadge'),
  fireBtn: document.getElementById('fireBtn'),
  useCameraBtn: document.getElementById('useCameraBtn'),
  useTestImagesBtn: document.getElementById('useTestImagesBtn'),
  nextTestImageBtn: document.getElementById('nextTestImageBtn'),
  toggleAutoTestBtn: document.getElementById('toggleAutoTestBtn'),
  homeBtn: document.getElementById('homeBtn'),
  armBtn: document.getElementById('armBtn'),
  disarmBtn: document.getElementById('disarmBtn'),
  toggleAimBtn: document.getElementById('toggleAimBtn'),
  toggleWheelBtn: document.getElementById('toggleWheelBtn'),
  estopBtn: document.getElementById('estopBtn'),
  resetFaultBtn: document.getElementById('resetFaultBtn'),
};

function log(msg) {
  const ts = new Date().toLocaleTimeString();
  els.log.textContent = `[${ts}] ${msg}\n` + els.log.textContent;
}

function setBadge(el, text, ok) {
  el.textContent = text;
  el.style.borderColor = ok ? '#2ea043' : '#b62324';
  el.style.color = ok ? '#8fdd9f' : '#ff9a9a';
}

function applyState(payload) {
  const v = payload.vision || {};
  const src = payload.vision_source || {};
  els.targetX.textContent = v.valid ? v.x_norm.toFixed(3) : '--';
  els.targetY.textContent = v.valid ? v.y_norm.toFixed(3) : '--';
  els.targetConf.textContent = v.valid ? v.confidence.toFixed(2) : '--';

  setBadge(els.modeBadge, `MODE: ${payload.mode}`, payload.mode !== 'FAULT');
  setBadge(els.yawBadge, `Yaw: ${payload.yaw_homed ? 'Homed' : 'Not Homed'}`, payload.yaw_homed);
  setBadge(els.pitchBadge, `Pitch: ${payload.pitch_zero_assumed ? 'Confirmed Low' : 'Unconfirmed'}`, payload.pitch_zero_assumed);
  setBadge(els.aimBadge, `Aim: ${payload.aim_ready ? 'Ready' : 'Not Ready'}`, payload.aim_ready);
  setBadge(els.wheelBadge, `Flywheel: ${payload.flywheel_ready ? 'Ready' : 'Not Ready'}`, payload.flywheel_ready);
  setBadge(els.fireBadge, `Fire Gate: ${payload.fire_enabled ? 'ENABLED' : 'BLOCKED'}`, payload.fire_enabled);
  setBadge(els.visionSourceBadge, `Vision Source: ${src.mode || '--'}`, src.mode === 'test-images' || src.mode === 'camera');
  const testName = src.test_image_name || '--';
  const count = typeof src.test_image_count === 'number' ? src.test_image_count : 0;
  const idx = typeof src.test_image_index === 'number' ? src.test_image_index : 0;
  setBadge(els.testImageBadge, `Test Image: ${count > 0 ? `${idx + 1}/${count}` : '--'} ${testName}`, count > 0);
  els.fireBtn.disabled = !payload.fire_enabled;
  els.nextTestImageBtn.disabled = src.mode !== 'test-images';
  els.toggleAutoTestBtn.disabled = src.mode !== 'test-images';

  state.aimReady = payload.aim_ready;
  state.flywheelReady = payload.flywheel_ready;
  state.testAuto = !!src.test_auto_advance;
}

async function api(path, opts = {}) {
  const resp = await fetch(path, {
    method: opts.method || 'POST',
    headers: {'Content-Type': 'application/json'},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  return resp.json();
}

async function refreshState() {
  const resp = await fetch('/api/state');
  const data = await resp.json();
  applyState(data);
}

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => log('WebSocket connected');
  ws.onclose = () => {
    log('WebSocket disconnected; retrying');
    setTimeout(connectWs, 1000);
  };
  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === 'state') {
      applyState(msg.payload);
      return;
    }
    if (msg.type === 'fire_state') {
      log(`Fire: ${msg.payload.status}`);
      return;
    }
    if (msg.type === 'fault') {
      log(`FAULT: ${msg.payload.reason}`);
      return;
    }
    if (msg.type !== 'heartbeat') {
      log(`${msg.type}: ${JSON.stringify(msg.payload)}`);
    }
  };
}

els.homeBtn.onclick = async () => {
  const r = await api('/api/control/home');
  log(`home -> ${JSON.stringify(r)}`);
  refreshState();
};

els.useCameraBtn.onclick = async () => {
  const r = await api('/api/vision/source?mode=camera');
  log(`vision-source(camera) -> ${JSON.stringify(r)}`);
  refreshState();
};

els.useTestImagesBtn.onclick = async () => {
  const r = await api('/api/vision/source?mode=test-images');
  log(`vision-source(test-images) -> ${JSON.stringify(r)}`);
  refreshState();
};

els.nextTestImageBtn.onclick = async () => {
  const r = await api('/api/vision/test-images/next');
  log(`next-test-image -> ${JSON.stringify(r)}`);
  refreshState();
};

els.toggleAutoTestBtn.onclick = async () => {
  const next = !state.testAuto;
  const r = await api(`/api/vision/test-images/auto?enabled=${next}`);
  log(`auto-test(${next}) -> ${JSON.stringify(r)}`);
  refreshState();
};

els.armBtn.onclick = async () => {
  const r = await api('/api/control/arm');
  log(`arm -> ${JSON.stringify(r)}`);
  refreshState();
};

els.disarmBtn.onclick = async () => {
  const r = await api('/api/control/disarm');
  log(`disarm -> ${JSON.stringify(r)}`);
  refreshState();
};

els.toggleAimBtn.onclick = async () => {
  const next = !state.aimReady;
  const r = await api(`/api/control/aim-ready?ready=${next}`);
  log(`aim-ready(${next}) -> ${JSON.stringify(r)}`);
  refreshState();
};

els.toggleWheelBtn.onclick = async () => {
  const next = !state.flywheelReady;
  const r = await api(`/api/control/flywheel-ready?ready=${next}`);
  log(`flywheel-ready(${next}) -> ${JSON.stringify(r)}`);
  refreshState();
};

els.fireBtn.onclick = async () => {
  const r = await api('/api/control/fire');
  log(`fire -> ${JSON.stringify(r)}`);
  refreshState();
};

els.estopBtn.onclick = async () => {
  const r = await api('/api/control/estop');
  log(`estop -> ${JSON.stringify(r)}`);
  refreshState();
};

els.resetFaultBtn.onclick = async () => {
  const r = await api('/api/control/reset-fault');
  log(`reset-fault -> ${JSON.stringify(r)}`);
  refreshState();
};

refreshState();
connectWs();
