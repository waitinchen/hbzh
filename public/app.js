/**
 * 河北彩花 Voice Call — Frontend v12
 * STT+LLM: getUserMedia → PCM → WebSocket binary → Server → OpenAI Realtime
 * TTS: MiniMax streaming chunks → sentence queue → sequential playback
 * 打斷: server sends interrupt → client stops playback immediately
 */
(function () {
  'use strict';

  const TOKEN_KEY = 'xiaos_token';
  const USER_KEY  = 'xiaos_user';

  const $ = id => document.getElementById(id);
  const pageLogin       = $('page-login');
  const pageHome        = $('page-home');
  const pageCall        = $('page-call');
  const formLogin       = $('form-login');
  const inputUsername   = $('input-username');
  const inputPassword   = $('input-password');
  const loginError      = $('login-error');
  const btnLogin        = $('btn-login');
  const userNameEl      = $('user-name');
  const btnLogout       = $('btn-logout');
  const btnCall         = $('btn-call');
  const btnHangup       = $('btn-hangup');
  const callStatus      = $('call-status');
  const btnMic          = $('btn-mic');
  const btnMute         = $('btn-mute');
  const btnSendText     = $('btn-send-text');
  const transcriptBox   = $('transcript-box');
  const callTimerEl     = $('call-timer');
  const callActivity    = $('call-activity');
  const waveformCanvas  = $('waveform-canvas');
  const healthDot       = $('health-dot');
  const healthText      = $('health-text');
  const llmIndicator    = $('llm-indicator');
  const ttsIndicator    = $('tts-indicator');
  const sttIndicator    = $('stt-indicator');
  const micIndicator    = $('mic-indicator');
  const installBanner   = $('install-banner');
  const btnInstall      = $('btn-install');
  const btnDismissInstall = $('btn-dismiss-install');
  const micLabel        = $('mic-label');
  const muteLabel       = $('mute-label');
  const transcriptDrawer  = $('transcript-drawer');
  const transcriptToggle  = $('transcript-toggle');
  const callWrapper     = document.querySelector('.call-wrapper');

  let ws = null, wsReady = false, callState = 'idle';
  let intentionalClose = false, reconnectAttempts = 0, reconnectTimer = null;
  const MAX_RECONNECT = 5;

  // Mic
  let micStream = null, micContext = null, micProcessor = null;
  let micSource = null, micAnalyser = null;
  let isMicActive = false, isMuted = false;

  // Playback — HTMLAudioElement blob queue（MP3 串流相容）
  let audioChunks = [], audioFlushTimer = null, isPlaying = false, audioQueue = [];
  let currentBlobUrl = null;
  const audioEl = new Audio();
  let playbackWatchdog = null; // 防止 isPlaying 卡住的安全計時器

  // State
  let callTimerInterval = null, callStartTime = 0;
  let waveformRAF = null, ringAudio = null, ringCount = 0, ringTimer = null;
  let healthInterval = null, deferredPrompt = null;
  let isProcessing = false;
  let isHangingUp = false;
  let isTtsActive = false; // server 整段 TTS 進行中（含句間空窗）

  const VAD_THRESHOLD_MIN = 0.004; // 絕對最低門檻（極安靜環境）
  let noiseFloor = 0.008;          // 動態噪音底板，持續更新
  let noiseFrames = 0;             // 靜音幀計數，用來校準底板

  // ── Utilities ──────────────────────────────────────────────
  function showPage(page) {
    [pageLogin, pageHome, pageCall].forEach(p => p.classList.remove('active'));
    requestAnimationFrame(() => page.classList.add('active'));
    if (page === pageHome) startHealthPolling(); else stopHealthPolling();
  }
  function getApiBase() { return window.location.origin; }
  function getWsUrl() {
    const u = window.location;
    return (u.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + u.host + '/voice';
  }
  function getToken() { return localStorage.getItem(TOKEN_KEY); }
  function setToken(token, username) {
    if (token) { localStorage.setItem(TOKEN_KEY, token); if (username) localStorage.setItem(USER_KEY, username); }
    else { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); }
  }
  function checkAuth() {
    const token = getToken();
    if (token) { userNameEl.textContent = localStorage.getItem(USER_KEY) || '用戶'; showPage(pageHome); }
    else showPage(pageLogin);
  }
  let streamingLine = null; // reuse same line for streaming assistant deltas
  function addTranscript(role, text, streaming) {
    if (streaming && streamingLine) {
      // Update existing streaming line in-place
      const time = new Date().toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });
      streamingLine.textContent = time + '  ' + text;
      transcriptBox.scrollTop = transcriptBox.scrollHeight;
      return;
    }
    const line = document.createElement('div');
    line.className = 'line ' + role;
    const time = new Date().toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });
    line.textContent = time + '  ' + text;
    transcriptBox.appendChild(line);
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
    if (streaming) streamingLine = line;
    else streamingLine = null;
  }
  function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) ws.send(data);
      else ws.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }
  let activityState = '';
  function setActivityState(state) {
    if (!callActivity) return;
    activityState = state;
    const labels = { listening: '聆聽中...', transcribing: '辨識中...', thinking: '思考中...', speaking: '河北彩花說話中...' };
    callActivity.textContent = labels[state] || '';
    callActivity.className = 'call-activity' + (state ? ' active' : '');
  }

  // ── Health ──────────────────────────────────────────────────
  async function pollHealth() {
    try {
      const data = await fetch(getApiBase() + '/api/health').then(r => r.json());
      healthDot.className = 'health-dot online'; healthText.textContent = '連線正常';
      if (llmIndicator) llmIndicator.className = 'key-indicator ' + (data.llm ? 'ok' : 'fail');
      if (ttsIndicator) ttsIndicator.className = 'key-indicator ' + (data.tts ? 'ok' : 'fail');
      if (sttIndicator) sttIndicator.className = 'key-indicator ' + (data.stt ? 'ok' : 'fail');
    } catch { healthDot.className = 'health-dot offline'; healthText.textContent = '無法連線'; }
    // MIC check
    checkMicPermission();
  }
  async function checkMicPermission() {
    if (!micIndicator) return;
    try {
      const result = await navigator.permissions.query({ name: 'microphone' });
      micIndicator.className = 'key-indicator ' + (result.state === 'granted' ? 'ok' : result.state === 'denied' ? 'fail' : '');
      result.onchange = () => {
        micIndicator.className = 'key-indicator ' + (result.state === 'granted' ? 'ok' : result.state === 'denied' ? 'fail' : '');
      };
    } catch {
      // Permissions API not supported, try getUserMedia
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(t => t.stop());
        micIndicator.className = 'key-indicator ok';
      } catch { micIndicator.className = 'key-indicator fail'; }
    }
  }
  function startHealthPolling() { pollHealth(); clearInterval(healthInterval); healthInterval = setInterval(pollHealth, 30000); }
  function stopHealthPolling() { clearInterval(healthInterval); healthInterval = null; }

  // ── Call State Machine ──────────────────────────────────────
  function setCallState(state) {
    callState = state;
    switch (state) {
      case 'ringing':
        callStatus.textContent = '撥打中...';
        callStatus.classList.add('ringing');
        [btnMic, btnMute, btnSendText].forEach(b => b.disabled = true);
        callTimerEl.textContent = ''; callActivity.textContent = '';
        if (callWrapper) callWrapper.classList.remove('connected');
        startRingtone();
        break;
      case 'connected':
        callStatus.textContent = '通話中';
        callStatus.classList.remove('ringing');
        [btnMic, btnMute, btnSendText].forEach(b => b.disabled = false);
        addTranscript('system', '通話已接通');
        if (callWrapper) callWrapper.classList.add('connected');
        startCallTimer(); startWaveform(); startMic();
        break;
      case 'ended':
        callStatus.classList.remove('ringing');
        if (callWrapper) callWrapper.classList.remove('connected');
        stopRingtone(); stopCallTimer(); stopWaveform(); stopMic();
        if (ws) { intentionalClose = true; ws.close(); ws = null; }
        clearTimeout(reconnectTimer); reconnectAttempts = 0; wsReady = false;
        setTimeout(() => showPage(pageHome), 300);
        break;
    }
  }

  // ── Ringtone ────────────────────────────────────────────────
  function startRingtone() {
    ringCount = 0;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.frequency.value = 440; osc.type = 'sine'; gain.gain.value = 0.12;
      osc.connect(gain); gain.connect(ctx.destination);
      ringAudio = { ctx, osc, gain }; osc.start();
      ringTimer = setInterval(() => {
        ringCount++; gain.gain.value = gain.gain.value > 0 ? 0 : 0.12;
        if (ringCount >= 6 && wsReady) { stopRingtone(); setCallState('connected'); }
      }, 400);
    } catch {
      ringTimer = setInterval(() => {
        ringCount++;
        if (ringCount >= 6 && wsReady) { stopRingtone(); setCallState('connected'); }
      }, 400);
    }
  }
  function stopRingtone() {
    clearInterval(ringTimer); ringTimer = null;
    if (ringAudio) { try { ringAudio.osc.stop(); ringAudio.ctx.close(); } catch (_) {} ringAudio = null; }
  }

  // ── Timer ────────────────────────────────────────────────────
  function startCallTimer() {
    callStartTime = Date.now(); callTimerEl.textContent = '00:00';
    callTimerInterval = setInterval(() => {
      const e = Math.floor((Date.now() - callStartTime) / 1000);
      callTimerEl.textContent = String(Math.floor(e / 60)).padStart(2, '0') + ':' + String(e % 60).padStart(2, '0');
    }, 1000);
  }
  function stopCallTimer() { clearInterval(callTimerInterval); callTimerInterval = null; }

  // ── Waveform ─────────────────────────────────────────────────
  function startWaveform() {
    if (!waveformCanvas) return;
    const ctx = waveformCanvas.getContext('2d'); let phase = 0;
    function draw() {
      const w = waveformCanvas.clientWidth, h = waveformCanvas.clientHeight, dpr = window.devicePixelRatio || 1;
      if (waveformCanvas.width !== w * dpr) waveformCanvas.width = w * dpr;
      if (waveformCanvas.height !== h * dpr) waveformCanvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
      const midY = h / 2; phase += 0.04;
      let inputLevel = 0;
      if (micAnalyser && isMicActive && !isMuted) {
        const d = new Uint8Array(micAnalyser.fftSize); micAnalyser.getByteTimeDomainData(d);
        let peak = 0; for (let i = 0; i < d.length; i++) { const v = Math.abs(d[i]-128)/128; if (v>peak) peak=v; }
        inputLevel = Math.min(peak * 3, 1);
      }
      // 播放中用固定振幅（不依賴 AudioContext analyser，避免 createMediaElementSource 問題）
      const outputLevel = isPlaying ? 0.5 + Math.sin(Date.now() / 200) * 0.3 : 0;
      drawNeonWave(ctx, w, midY, outputLevel, phase, '0, 210, 106', 80);
      drawNeonWave(ctx, w, midY, inputLevel, phase + Math.PI/3, '139, 92, 246', 80);
      ctx.setTransform(1,0,0,1,0,0); waveformRAF = requestAnimationFrame(draw);
    }
    waveformRAF = requestAnimationFrame(draw);
  }
  function drawNeonWave(ctx, w, midY, level, phase, rgb, pts) {
    // ECG style: sharp spikes, high sensitivity
    const amp = 1 + level * midY * 0.9, alpha = 0.2 + level * 0.8;
    for (const [lw, a] of [[3+level*8, alpha*0.15],[1.5+level*2, alpha]]) {
      ctx.beginPath(); ctx.lineWidth = lw; ctx.strokeStyle = `rgba(${rgb},${a})`;
      for (let i = 0; i <= pts; i++) {
        const t = i / pts;
        const x = t * w;
        // Multi-frequency sharp wave — ECG heartbeat feel
        const base = Math.sin(t * Math.PI * 3 + phase);
        const spike = Math.pow(Math.abs(Math.sin(t * Math.PI * 5 + phase * 1.7)), 3) * Math.sign(Math.sin(t * Math.PI * 5 + phase * 1.7));
        const y = midY + (base * 0.5 + spike * 0.5) * amp;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
  function stopWaveform() { if (waveformRAF) { cancelAnimationFrame(waveformRAF); waveformRAF = null; } }

  // ── Microphone — getUserMedia → PCM 16kHz → WS binary ───────
  // ★ 完全不用 Web Speech API，零叮聲
  async function startMic() {
    if (isMicActive) return;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          // ★ 不強制 sampleRate，讓瀏覽器用原生 rate
        }
      });

      // ★ 用原生 sampleRate，不強制 16000
      micContext = new (window.AudioContext || window.webkitAudioContext)();
      const actualSampleRate = micContext.sampleRate;
      console.log('[Mic] Native sampleRate:', actualSampleRate);

      micSource  = micContext.createMediaStreamSource(micStream);
      micAnalyser = micContext.createAnalyser(); micAnalyser.fftSize = 512; micAnalyser.smoothingTimeConstant = 0.3;
      micSource.connect(micAnalyser);

      micProcessor = micContext.createScriptProcessor(4096, 1, 1);
      micSource.connect(micProcessor);
      micProcessor.connect(micContext.destination);

      let localSpeechTimer = null;
      micProcessor.onaudioprocess = (e) => {
        if (!isMicActive || isMuted) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const f32 = e.inputBuffer.getChannelData(0);

        // ★ TTS 進行中（播放中 或 句間空窗）→ 送靜音，避免回音觸發 OpenAI VAD
        if (isPlaying || isTtsActive) {
          const silence = new Int16Array(f32.length); // 全零
          ws.send(silence.buffer);
          return; // 不更新噪音底板和 UI 狀態
        }

        // 動態噪音底板計算
        let sum = 0; for (let i = 0; i < f32.length; i++) sum += f32[i]*f32[i];
        const rms = Math.sqrt(sum / f32.length);
        const dynamicThreshold = Math.max(VAD_THRESHOLD_MIN, noiseFloor * 2.0);

        // ★ 噪音閘門：低於門檻 → 送靜音（擋掉背景影片/電視系統音）
        if (rms < dynamicThreshold) {
          noiseFrames++;
          if (noiseFrames > 10) noiseFloor = noiseFloor * 0.95 + rms * 0.05;
          const silence = new Int16Array(f32.length); // 全零
          ws.send(silence.buffer); // 保持幀時序，但 OpenAI VAD 聽不到噪音
          return;
        }
        noiseFrames = 0;

        // 超過門檻 → 送真實音訊（人聲）
        const i16 = new Int16Array(f32.length);
        for (let i = 0; i < f32.length; i++) {
          const s = Math.max(-1, Math.min(1, f32[i]));
          i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        ws.send(i16.buffer);
        // 本地 VAD：偵測到聲音 → 顯示「辨識中」，靜音 1.5 秒 → 切「思考中」
        setActivityState('transcribing');
        clearTimeout(localSpeechTimer);
        localSpeechTimer = setTimeout(() => {
          if (activityState === 'transcribing') setActivityState('thinking');
        }, 1500);
      };

      isMicActive = true;
      btnMic.classList.add('active');
      if (micLabel) micLabel.textContent = '麥克風開';
      setActivityState('listening');

      // ★ 通知 server 實際 sampleRate，server 用這個連 Deepgram
      send({ type: 'audio_config', sampleRate: actualSampleRate });
      console.log('[Mic] Started @', actualSampleRate, 'Hz');
    } catch (err) {
      console.error('[Mic]', err.message);
      addTranscript('system', '⚠️ 無法開啟麥克風：' + err.message);
    }
  }
  function stopMic() {
    isMicActive = false; isMuted = false;
    if (micProcessor) { micProcessor.disconnect(); micProcessor = null; }
    if (micSource)    { micSource.disconnect(); micSource = null; }
    if (micContext)   { micContext.close().catch(()=>{}); micContext = null; }
    if (micStream)    { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    micAnalyser = null;
    btnMic.classList.remove('active');
    if (micLabel) micLabel.textContent = '麥克風';
    if (btnMute)  btnMute.classList.remove('muted');
    if (muteLabel) muteLabel.textContent = '靜音';
  }
  function toggleMute() {
    if (!isMicActive) return;
    isMuted = !isMuted;
    btnMute.classList.toggle('muted', isMuted);
    if (muteLabel) muteLabel.textContent = isMuted ? '取消靜音' : '靜音';
    setActivityState(isMuted ? '' : 'listening');
  }

  // ── Audio Playback — HTMLAudioElement blob queue ──────────────
  // MP3 chunks → 200ms 批次成 Blob → 依序播放（HTMLAudioElement 天生支援 MP3 串流）
  // decodeAudioData 不能用：MP3 chunk 不是完整檔案，decode 會產生雜訊/發抖
  function playAudioChunk(base64Data) {
    try {
      isProcessing = true;
      const bin = atob(base64Data), bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      audioChunks.push(bytes);
      setActivityState('speaking');
      // 不再用 200ms timer 切片，等 audio_end 信號整句 flush
      // 但加 2s 安全 fallback 以防 audio_end 沒到
      clearTimeout(audioFlushTimer);
      audioFlushTimer = setTimeout(flushToQueue, 2000);
    } catch (e) { console.warn('[Audio] chunk:', e); }
  }

  function flushToQueue() {
    if (audioChunks.length === 0) return;
    const blob = new Blob(audioChunks, { type: 'audio/mpeg' });
    audioChunks = [];
    const url = URL.createObjectURL(blob);
    audioQueue.push(url);
    if (!isPlaying) playNext();
  }

  function flushAndPlayAudio() {
    clearTimeout(audioFlushTimer);
    flushToQueue();
    if (!isPlaying && audioQueue.length > 0) playNext();
  }

  function playNext() {
    clearTimeout(playbackWatchdog);
    if (audioQueue.length === 0) {
      isPlaying = false; isProcessing = false;
      send({ type: 'playback_done' });
      setActivityState('listening');
      return;
    }
    isPlaying = true;
    const url = audioQueue.shift();
    if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = url;
    audioEl.src = currentBlobUrl;
    audioEl.play().catch(err => { console.warn('[Audio] play:', err.message); isPlaying = false; playNext(); });
    audioEl.onended = () => playNext();
    audioEl.onerror = () => playNext();
    // ★ Watchdog: 30 秒內沒播完就強制跳下一個（防止 isPlaying 卡住）
    playbackWatchdog = setTimeout(() => {
      console.warn('[Audio] watchdog: force advance');
      audioEl.pause(); audioEl.src = '';
      playNext();
    }, 30000);
  }

  // ── WebSocket ────────────────────────────────────────────────
  function connectVoice() {
    const token = getToken(); if (!token) { callStatus.textContent = '請先登入'; return; }
    const name = localStorage.getItem(USER_KEY) || '';
    const url = getWsUrl() + '?token=' + encodeURIComponent(token) + '&name=' + encodeURIComponent(name);
    intentionalClose = false;
    try { ws = new WebSocket(url); } catch { callStatus.textContent = '連線失敗'; return; }
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      wsReady = true; reconnectAttempts = 0;
      if (callState !== 'ringing') setCallState('connected');
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return;
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'audio' && msg.data) {
          playAudioChunk(msg.data);
        } else if (msg.type === 'audio_end') {
          // 一句 TTS 音頻結束 → 立即 flush 成完整 Blob（避免切片暴音）
          clearTimeout(audioFlushTimer);
          flushToQueue();
        } else if (msg.type === 'transcript' && msg.text) {
          addTranscript('assistant', msg.text, true);
        } else if (msg.type === 'transcript_done') {
          streamingLine = null; // finalize: next response starts a new line
        } else if (msg.type === 'user_transcript' && msg.text) {
          addTranscript('user', msg.text); setActivityState('thinking');
        } else if (msg.type === 'status') {
          if (msg.state === 'listening') { flushAndPlayAudio(); }
          else setActivityState(msg.state);
        } else if (msg.type === 'tts_start') {
          isTtsActive = true;
        } else if (msg.type === 'tts_end') {
          isTtsActive = false;
        } else if (msg.type === 'interrupt') {
          // OpenAI 偵測到用戶插嘴 → 立即停止播放
          audioEl.pause(); audioEl.src = '';
          if (currentBlobUrl) { URL.revokeObjectURL(currentBlobUrl); currentBlobUrl = null; }
          for (const url of audioQueue) URL.revokeObjectURL(url);
          audioQueue = []; audioChunks = [];
          clearTimeout(audioFlushTimer);
          clearTimeout(playbackWatchdog);
          isPlaying = false;
          isProcessing = false;
          isTtsActive = false;
        } else if (msg.type === 'hangup') {
          isHangingUp = true;
          setCallState('ended');
        } else if (msg.type === 'error') {
          const errMsg = msg.message || '';
          addTranscript('system', '⚠ ' + errMsg);
          // Server error with auto-reconnect → show status, don't panic user
          if (errMsg.includes('自動重連')) {
            setActivityState('thinking');
            callStatus.textContent = '重連中...';
          }
        }
      } catch (_) {}
    };
    ws.onclose = (ev) => {
      wsReady = false;
      if (isHangingUp) { isHangingUp = false; return; }
      if (intentionalClose || callState === 'ended' || callState === 'idle') return;
      if (ev.code === 4001) { setCallState('ended'); setToken(null); showPage(pageLogin); loginError.textContent = '登入已過期'; return; }
      if (reconnectAttempts < MAX_RECONNECT) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000) + Math.random()*1000;
        reconnectAttempts++;
        callStatus.textContent = `重連中... (${reconnectAttempts}/${MAX_RECONNECT})`;
        reconnectTimer = setTimeout(connectVoice, delay);
      } else { callStatus.textContent = '連線中斷'; addTranscript('system', '通話中斷，請重新撥打'); }
    };
    ws.onerror = () => {};
  }
  function disconnectVoice() {
    intentionalClose = true; clearTimeout(reconnectTimer);
    if (ws) { ws.close(); ws = null; }
    stopMic();
  }

  // ── PWA Install ──────────────────────────────────────────────
  function initInstallPrompt() {
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone;
    if (isStandalone) return;
    const dismissed = localStorage.getItem('pwa_dismiss');
    if (dismissed && Date.now() - Number(dismissed) < 86400000) return;
    const showBanner = () => { if (installBanner) installBanner.style.display = 'flex'; };
    const hideBanner = () => { if (installBanner) installBanner.style.display = 'none'; };
    window.addEventListener('beforeinstallprompt', e => { e.preventDefault(); deferredPrompt = e; showBanner(); });
    window.addEventListener('appinstalled', () => { deferredPrompt = null; hideBanner(); });
    if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone && installBanner) {
      showBanner();
      const span = installBanner.querySelector('span');
      if (span) span.textContent = '點擊分享→加入主畫面';
      if (btnInstall) btnInstall.style.display = 'none';
    }
    if (btnInstall) btnInstall.addEventListener('click', async () => {
      if (deferredPrompt) { deferredPrompt.prompt(); await deferredPrompt.userChoice; deferredPrompt = null; }
      hideBanner();
    });
    if (btnDismissInstall) btnDismissInstall.addEventListener('click', () => {
      localStorage.setItem('pwa_dismiss', String(Date.now())); hideBanner();
    });
  }

  // ── Event Listeners ──────────────────────────────────────────
  // Transcript toggle removed — floating HUD always visible

  formLogin.addEventListener('submit', async e => {
    e.preventDefault(); loginError.textContent = ''; btnLogin.disabled = true;
    const username = (inputUsername.value||'').trim(), password = inputPassword.value||'';
    try {
      const res = await fetch(getApiBase()+'/api/auth/login', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({username,password})
      });
      const data = await res.json().catch(()=>({}));
      if (res.ok && data.token) { setToken(data.token, data.username||username); userNameEl.textContent = data.username||username; showPage(pageHome); }
      else loginError.textContent = data.message||'帳號或密碼錯誤';
    } catch { loginError.textContent = '網路錯誤，請稍後再試'; }
    btnLogin.disabled = false;
  });

  btnLogout.addEventListener('click', () => { setToken(null); disconnectVoice(); showPage(pageLogin); inputPassword.value=''; });

  btnCall.addEventListener('click', () => {
    transcriptBox.innerHTML = ''; streamingLine = null;
    audioChunks = []; audioQueue = []; isPlaying = false; isProcessing = false;
    showPage(pageCall); setCallState('ringing'); connectVoice();
  });

  btnHangup.addEventListener('click', () => setCallState('ended'));
  btnMic.addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (isMicActive) stopMic(); else startMic();
  });
  if (btnMute) btnMute.addEventListener('click', toggleMute);
  btnSendText.addEventListener('click', () => {
    if (isProcessing) return;
    const text = window.prompt('輸入訊息：','');
    if (!text?.trim()) return;
    isProcessing = true;
    addTranscript('user', text.trim());
    send({ type: 'text', text: text.trim() });
    setActivityState('thinking');
  });

  if ('serviceWorker' in navigator) {
    // ★ 新 SW 接管時自動 reload，確保拿到最新 app.js
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      window.location.reload();
    });
    navigator.serviceWorker.register('/sw.js').then(reg => {
      reg.addEventListener('updatefound', () => {
        const sw = reg.installing;
        if (sw) sw.addEventListener('statechange', () => {
          if (sw.state === 'activated' && navigator.serviceWorker.controller) addTranscript('system', '應用已更新');
        });
      });
    }).catch(()=>{});
  }

  initInstallPrompt();
  checkAuth();
  console.log('%c🎙 河北彩花 Voice Call', 'color:#00D26A;font-weight:bold;font-size:14px');
})();
