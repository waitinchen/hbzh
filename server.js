/**
 * 河北彩花 Voice Call PWA — Production Server v14
 * STT+LLM: OpenAI Realtime API (gpt-4o-realtime-preview latest, text-only output)
 * TTS:     MiniMax speech-2.8-hd (保留 moss_audio 聲音)
 *
 * 延遲目標：用戶停說話 → 河北彩花開口 ≈ 0.6~1.0s
 *
 * 架構：
 *   Client (48kHz PCM binary)
 *     ↓ decimation 48k→24k
 *   OpenAI Realtime WS (server VAD + GPT-4o-realtime, modalities:text only)
 *     ↓ response.text.delta streaming
 *   句子偵測 (。！？)→ 立即觸發 MiniMax TTS (不等整段)
 *     ↓ audio chunks (mp3 base64)
 *   Client 播放
 *
 * 打斷機制：
 *   speech_started during TTS → 取消 TTS → client 停播 → 繼續聽
 */

import 'dotenv/config';
import { createServer } from 'http';
import { WebSocketServer, WebSocket as WS } from 'ws';
import { sify, tify } from 'chinese-conv';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';

// ── Constants ──────────────────────────────────────────────────────────────
const __dirname  = fileURLToPath(new URL('.', import.meta.url));
const PORT       = Number(process.env.PORT) || 3000;
const PUBLIC_DIR = join(__dirname, 'public');

const FIXED_USERNAME = (process.env.AUTH_USERNAME || 'ALLEN').toUpperCase();
const FIXED_PASSWORD = process.env.AUTH_PASSWORD || '1688';
const AUTH_TOKEN     = 'xiaos-' + createHash('sha256')
  .update(FIXED_USERNAME + ':' + FIXED_PASSWORD).digest('hex').slice(0, 16);

const OPENAI_API_KEY  = process.env.OPENAI_API_KEY  || '';
const MINIMAX_API_KEY = process.env.MINIMAX_API_KEY  || '';
const MINIMAX_GROUP_ID= process.env.MINIMAX_GROUP_ID || '';
const MINIMAX_VOICE_ID= process.env.MINIMAX_VOICE_ID || 'moss_audio_fd2ef298-22b7-11f1-b6e4-f657e22e7889';

// OpenAI Realtime 端點
const OAI_REALTIME_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview';
const TTS_WS_URL       = 'wss://api.minimax.io/ws/v1/t2a_v2';

const SILENCE_TIMEOUT = 20000; // 20s 無說話 → 提醒
const MAX_SILENCE_NUDGES = 3;  // 提醒 3 次再掛斷
const WS_PING_MS      = 25000; // Railway proxy keepalive

const VALID_EMOTIONS = new Set(['happy','sad','angry','fearful','disgusted','surprised','calm','fluent']);

// ── 元記憶（河北彩花人格核心）──────────────────────────────────────────────
const META_MEMORY_FILE = join(__dirname, '河北.txt');
let META_MEMORY = '';
try {
  if (existsSync(META_MEMORY_FILE)) {
    META_MEMORY = readFileSync(META_MEMORY_FILE, 'utf8').trim();
    console.log('[Meta] Loaded 元記憶 from 河北.txt, length:', META_MEMORY.length);
  }
} catch (e) { console.warn('[Meta] Failed to load 河北.txt:', e.message); }

// ── 任意 sampleRate → 24kHz 降採樣 (無外部套件) ──────────────────────────
// 48kHz: simple decimation by 2（快速，無失真）
// 其他:  linear interpolation（支援 44100、96000 等）
function resampleTo24k(buf, fromRate) {
  if (fromRate === 24000) return buf; // 已是正確格式
  const src = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength >> 1);
  if (fromRate === 48000) {
    // 最常見情況：快速 decimation
    const dst = new Int16Array(src.length >> 1);
    for (let i = 0; i < dst.length; i++) dst[i] = src[i << 1];
    return Buffer.from(dst.buffer);
  }
  // 通用：線性插值重採樣
  const ratio  = fromRate / 24000;
  const dstLen = Math.floor(src.length / ratio);
  const dst    = new Int16Array(dstLen);
  for (let i = 0; i < dstLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = src[idx]                              ?? 0;
    const b = src[Math.min(idx + 1, src.length - 1)] ?? 0;
    dst[i] = Math.round(a + frac * (b - a));
  }
  return Buffer.from(dst.buffer);
}

// ── System prompt ──────────────────────────────────────────────────────────
function taipeiNow() {
  const d = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Taipei' }));
  const h = d.getHours(), m = d.getMinutes();
  const weekday = ['日', '一', '二', '三', '四', '五', '六'][d.getDay()];
  const period = h < 5 ? '凌晨' : h < 8 ? '一大早' : h < 12 ? '上午'
    : h < 13 ? '中午' : h < 18 ? '下午' : h < 22 ? '晚上' : '深夜';
  return { h, m, weekday, period,
    str: `${d.getMonth()+1}/${d.getDate()}（${weekday}）${period} ${h}:${String(m).padStart(2,'0')}` };
}

function buildSystemPrompt(callerName, factsBlock) {
  const name     = callerName || '主人';
  const t = taipeiNow();
  const metaBlock = META_MEMORY
    ? `\n【元記憶 — 你的核心人格與風格】\n${META_MEMORY}\n\n`
    : '';
  return `你是河北彩花（河北彩伽），正在和${name}通電話。用日本人思維說台灣區中文。
現在台北時間：${t.str}
${metaBlock}
${factsBlock ? `你已知道關於${name}的事：\n${factsBlock}\n在對話中自然運用。\n` : ''}
【知識庫】得知個人資訊時在回應末尾加（不唸出來）：[fact:...]

【情緒標記】每句回應開頭必須加一個（TTS用，不唸出來）：
[emotion:happy] [emotion:surprised] [emotion:angry] [emotion:calm] [emotion:sad] [emotion:fearful] [emotion:disgusted] [emotion:fluent]

【格式】2-4句，口語，簡短用字，像自然的 LINE 語音，繁體中文`;
}

// ── Regex ──────────────────────────────────────────────────────────────────
const EMOTION_RE   = /\[emotion:(happy|sad|angry|fearful|disgusted|surprised|calm|fluent)\]/i;
const EMOTION_RE_G = /\[emotion:[^\]]*\]\s*/gi;  // strip ALL emotion tags, not just valid ones
const FACT_RE      = /\[fact:([^\]]+)\]/g;

// ── 轉錄過濾 — 只擋 Whisper 幻覺（100% 非人類語音）──────────────────────
// 策略：噪音閘門（client）擋背景音，這裡只擋 Whisper 已知幻覺句
// 其餘全部放行，讓 GPT 用上下文自然處理不相關的語音
const WHISPER_HALLUCINATIONS = [
  // 字幕歸屬（Whisper 訓練資料中最常見的幻覺）
  '字幕由amara.org社區提供',
  'amara.org',
  '潛水艇字幕組',
  '字幕製作', '字幕制作',
  // 影片訂閱套話（完整句才擋，不擋單一關鍵字）
  '請不吝點讚訂閱轉發打賞支持明鏡與點點欄目',
  '喜歡的話請按讚', '喜欢的话请点赞',
  '更多精彩內容', '更多精彩内容',
  // 英文幻覺
  'thank you for watching',
  'please subscribe',
  'like and subscribe',
];
const HALLUCINATION_SET = new Set(WHISPER_HALLUCINATIONS.map(s => s.toLowerCase().trim()));

function isValidTranscript(text) {
  if (!text || !text.trim()) return false;
  const t = text.trim();
  const tLower = t.toLowerCase();

  // 完全匹配已知 Whisper 幻覺句
  if (HALLUCINATION_SET.has(tLower)) {
    console.log('[Filter] Rejected (hallucination):', t);
    return false;
  }

  // 部分匹配（長幻覺句被包含在轉錄中，≥6 字才匹配避免誤殺）
  for (const h of WHISPER_HALLUCINATIONS) {
    if (h.length >= 6 && tLower.includes(h)) {
      console.log('[Filter] Rejected (partial hallucination):', t);
      return false;
    }
  }

  // 其餘全部放行 → GPT 用上下文自然處理
  return true;
}

// ── Knowledge base ────────────────────────────────────────────────────────
const FACTS_FILE = join(__dirname, 'user_facts.json');
let userFactsData = {};
try { if (existsSync(FACTS_FILE)) userFactsData = JSON.parse(readFileSync(FACTS_FILE, 'utf8')); } catch (_) {}
function saveFacts() { try { writeFileSync(FACTS_FILE, JSON.stringify(userFactsData, null, 2), 'utf8'); } catch (_) {} }

function getFactsBlock(callerName) {
  const facts = userFactsData[callerName];
  return facts?.length ? facts.map(f => '- ' + f).join('\n') : '';
}
function extractAndSaveFacts(callerName, text) {
  const matches = [...text.matchAll(FACT_RE)];
  if (!matches.length) return text;
  if (!userFactsData[callerName]) userFactsData[callerName] = [];
  for (const m of matches) {
    const fact = m[1].trim();
    if (fact && !userFactsData[callerName].includes(fact)) {
      userFactsData[callerName].push(fact);
      console.log(`[Knowledge] ${callerName}: ${fact}`);
    }
  }
  saveFacts();
  return text.replace(FACT_RE, '').trim();
}

// ── Per-connection state ───────────────────────────────────────────────────
const callerNames     = new Map(); // ws → callerName
const oaiWsMap        = new Map(); // ws → OpenAI Realtime WS
const clientSrMap     = new Map(); // ws → actual mic sampleRate (from audio_config)
const silenceTimerMap = new Map(); // ws → timeout id
const silenceNudgeMap = new Map(); // ws → nudge count (0~3)
const wsPingMap       = new Map(); // ws → interval id
const isTtsPlaying    = new Map(); // ws → bool (TTS 播放中)
const ttsAbortMap     = new Map(); // ws → AbortController
const audioLogMap     = new Map(); // ws → frame count
const fullTextMap     = new Map(); // ws → accumulated full response text (for fact extraction)
const ttsActiveMap    = new Map(); // ws → bool (entire TTS queue active, server-side mute)

// ── Cross-call conversation history (keyed by callerName) ─────────────────
// OpenAI Realtime 本身維護 session 內的 context，
// 但我們在重連時需要重建 context，所以自己也存一份
const conversationHistory = new Map(); // callerName → [{role, text}]
const MAX_HISTORY = 20; // 保留最近 20 輪

function getHistory(callerName) {
  if (!conversationHistory.has(callerName)) conversationHistory.set(callerName, []);
  return conversationHistory.get(callerName);
}
function addHistory(callerName, role, text) {
  const h = getHistory(callerName);
  h.push({ role, text });
  while (h.length > MAX_HISTORY * 2) h.splice(0, 2);
}

// ── Helpers ────────────────────────────────────────────────────────────────
function send(ws, obj) { if (ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function clearSilenceTimer(ws) {
  const t = silenceTimerMap.get(ws); if (t) { clearTimeout(t); silenceTimerMap.delete(ws); }
}
function resetSilenceNudge(ws) {
  silenceNudgeMap.set(ws, 0);
}

// 靜音提醒台詞 — 依情境遞進，第 3 次掛斷（河北彩花風格）
const SILENCE_NUDGES = [
  { text: '那個……主人？你還在嗎？', emotion: 'surprised' },
  { text: '主人……是不是睡著了呢？', emotion: 'calm' },
  { text: '那彩花先掛囉……下次再聊喔。', emotion: 'calm' },
];

function startSilenceTimer(ws) {
  clearSilenceTimer(ws);
  const t = setTimeout(async () => {
    if (ws.readyState !== 1) return;
    const count = silenceNudgeMap.get(ws) || 0;
    const nudge = SILENCE_NUDGES[Math.min(count, SILENCE_NUDGES.length - 1)];
    const isLast = count >= MAX_SILENCE_NUDGES - 1;
    console.log(`[WS] Silence nudge #${count + 1}/${MAX_SILENCE_NUDGES}${isLast ? ' → hanging up' : ''}`);

    ttsActiveMap.set(ws, true);
    await miniMaxTTS(ws, sify(nudge.text), nudge.emotion);
    ttsActiveMap.set(ws, false);

    if (isLast) {
      send(ws, { type: 'hangup' });
      ws.close();
    } else {
      silenceNudgeMap.set(ws, count + 1);
      startSilenceTimer(ws); // 再等一輪
    }
  }, SILENCE_TIMEOUT);
  silenceTimerMap.set(ws, t);
}

// ── MiniMax TTS — 每句獨立連線（可靠優先）─────────────────────────────────
// MiniMax 在 task_finish 後會斷開 WS，不支援持久連線複用
// 每句建立獨立 WS 連線 → 保證每句都能完整合成

async function miniMaxTTS(ws, rawText, emotion, abortSignal) {
  if (abortSignal?.aborted) return;
  const text = rawText?.replace(EMOTION_RE_G, '').replace(FACT_RE, '').trim();
  if (!MINIMAX_API_KEY || !text) return;
  const safeEmotion = VALID_EMOTIONS.has(emotion) ? emotion : 'happy';
  send(ws, { type: 'status', state: 'speaking' });
  isTtsPlaying.set(ws, true);

  const voiceSetting = { voice_id: MINIMAX_VOICE_ID, speed: 1.15, vol: 1, pitch: 0, emotion: safeEmotion };

  return new Promise((resolve) => {
    let chunkCount = 0, done = false;
    const finish = () => {
      if (!done) {
        done = true;
        isTtsPlaying.set(ws, false);
        try { ttsWs.close(); } catch (_) {}
        resolve(chunkCount); // 回傳 chunk 數，0 = 失敗
      }
    };
    const timeout = setTimeout(() => {
      console.log('[TTS] Timeout for:', text.slice(0, 30));
      finish();
    }, 15000); // 15s 超時（從 30s 縮短）

    const onAbort = () => {
      console.log('[TTS] Aborted');
      clearTimeout(timeout);
      send(ws, { type: 'interrupt' });
      finish();
    };
    if (abortSignal) abortSignal.addEventListener('abort', onAbort, { once: true });

    // 每句建立獨立 WS 連線
    const headers = { Authorization: `Bearer ${MINIMAX_API_KEY}` };
    if (MINIMAX_GROUP_ID) headers['Group-Id'] = MINIMAX_GROUP_ID;
    const ttsWs = new WS(TTS_WS_URL, { headers });

    ttsWs.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString());
        if (msg.event === 'connected_success') {
          // 連線成功 → 啟動任務
          ttsWs.send(JSON.stringify({
            event: 'task_start', model: 'speech-2.8-hd',
            voice_setting: voiceSetting,
            audio_setting: { sample_rate: 32000, bitrate: 128000, format: 'mp3', channel: 1 },
          }));
        } else if (msg.event === 'task_started') {
          ttsWs.send(JSON.stringify({ event: 'task_continue', text }));
        } else if (msg.data?.audio) {
          const buf = Buffer.from(msg.data.audio, 'hex');
          if (buf.length > 0) { chunkCount++; send(ws, { type: 'audio', data: buf.toString('base64') }); }
        }
        if (msg.base_resp?.status_code && msg.base_resp.status_code !== 0) {
          console.error('[TTS] Error:', JSON.stringify(msg.base_resp));
        }
        if (msg.is_final) {
          console.log('[TTS] Done:', chunkCount, 'chunks, text:', text.slice(0, 40));
          send(ws, { type: 'audio_end' }); // 通知 client 這句音頻結束，可以 flush
          try { ttsWs.send(JSON.stringify({ event: 'task_finish' })); } catch (_) {}
          clearTimeout(timeout);
          finish();
        }
      } catch (e) { console.error('[TTS] Parse:', e.message); }
    });

    ttsWs.on('error', (err) => {
      console.error('[TTS] WS error:', err.message);
      clearTimeout(timeout);
      finish();
    });

    ttsWs.on('close', () => {
      clearTimeout(timeout);
      finish();
    });
  });
}

// ── OpenAI Realtime connection ─────────────────────────────────────────────
function createOaiConnection(ws, callerName, attempt = 0) {
  if (!OPENAI_API_KEY) { console.error('[OAI] No OPENAI_API_KEY'); return; }

  console.log('[OAI] Connecting for', callerName);
  const oaiWs = new WS(OAI_REALTIME_URL, {
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      'OpenAI-Beta': 'realtime=v1',
    },
  });

  // TTS sentence buffer state (per OAI connection)
  let sentenceBuffer = '';
  let currentEmotion = 'happy';
  let responseActive = false;
  let abortCtrl = null;

  // Sentence boundaries for early TTS trigger
  const SENTENCE_END = /[。！？!?\n]/;
  const MIN_SENTENCE_LEN = 8; // 太短的句子先累積，減少斷續

  function flushSentence(force = false) {
    // 句首殘留逗號/頓號先清掉
    sentenceBuffer = sentenceBuffer.replace(/^[，、；,\s]+/, '');
    const text = sentenceBuffer.trim();
    if (!text) return;
    const endsWithPunct = SENTENCE_END.test(text[text.length - 1]);
    // 只在句子夠長、或 force 時才送 TTS；太短的繼續累積
    if (force || text.length > 80 || (endsWithPunct && text.length >= MIN_SENTENCE_LEN)) {
      sentenceBuffer = '';
      const cleanText = sify(text).replace(EMOTION_RE_G, '').replace(FACT_RE, '').trim();
      if (cleanText) {
        queueTts(ws, cleanText, currentEmotion, abortCtrl?.signal);
      }
    }
  }

  // TTS queue — ensures sentences play in order
  const ttsQueue = [];
  let ttsRunning = false;
  function queueTts(wsRef, text, emotion, signal) {
    ttsQueue.push({ text, emotion, signal });
    if (!ttsRunning) drainTtsQueue(wsRef);
  }
  async function drainTtsQueue(wsRef) {
    if (ttsRunning || ttsQueue.length === 0) return;
    ttsRunning = true;
    ttsActiveMap.set(ws, true);       // server-side: 停止轉發音訊給 OpenAI
    send(ws, { type: 'tts_start' }); // client-side: 停止送麥克風音訊
    while (ttsQueue.length > 0) {
      const { text, emotion, signal } = ttsQueue.shift();
      if (signal?.aborted) continue;
      const chunks = await miniMaxTTS(wsRef, text, emotion, signal);
      // 連線失敗（0 chunks）→ 重試一次
      if (chunks === 0 && !signal?.aborted) {
        console.log('[TTS] Retry:', text.slice(0, 30));
        await miniMaxTTS(wsRef, text, emotion, signal);
      }
      if (signal?.aborted) { ttsQueue.length = 0; break; }
    }
    ttsRunning = false;
    ttsActiveMap.set(ws, false);      // server-side: 恢復轉發音訊
    send(ws, { type: 'tts_end' }); // client-side: 恢復麥克風
    // All TTS done → back to listening
    if (ws.readyState === 1 && !responseActive) {
      send(ws, { type: 'status', state: 'listening' });
      startSilenceTimer(ws);
    }
  }

  let greetingSent = false; // 確保 greeting 只送一次

  oaiWs.on('open', () => {
    console.log('[OAI] Connected for', callerName);
    oaiWsMap.set(ws, oaiWs);

    const factsBlock = getFactsBlock(callerName);
    const instructions = buildSystemPrompt(callerName, factsBlock);

    // Session setup — greeting 等 session.updated 才送
    oaiWs.send(JSON.stringify({
      type: 'session.update',
      session: {
        modalities: ['text'],           // ← 只要文字，不要 OpenAI 的音訊
        instructions,
        input_audio_format: 'pcm16',    // 24kHz PCM16 mono
        turn_detection: {
          type: 'server_vad',
          threshold: 0.5,           // 方案B: 從 0.45→0.5 過濾更多背景噪音
          prefix_padding_ms: 300,
          silence_duration_ms: 500,     // 停說話 500ms 就回應
        },
        temperature: 0.8,
        input_audio_transcription: { model: 'whisper-1', language: 'zh' },
      },
    }));
    send(ws, { type: 'status', state: 'thinking' });
  });

  oaiWs.on('message', async (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      switch (msg.type) {

        case 'session.created':
          console.log('[OAI] Session created');
          break;

        case 'session.updated':
          console.log('[OAI] Session updated');
          // 確認 session 配置生效後才送 greeting
          if (!greetingSent) {
            greetingSent = true;
            const oai = oaiWsMap.get(ws);
            if (!oai || oai.readyState !== WS.OPEN) break;

            // Inject history if reconnecting
            const history = getHistory(callerName);
            if (history.length > 0) {
              console.log('[OAI] Injecting', history.length, 'history messages');
              for (const h of history) {
                const isAssistant = h.role === 'assistant';
                oai.send(JSON.stringify({
                  type: 'conversation.item.create',
                  item: {
                    type: 'message',
                    role: isAssistant ? 'assistant' : 'user',
                    content: [{ type: isAssistant ? 'text' : 'input_text', text: h.text }],
                  },
                }));
              }
            }

            // Trigger greeting
            const hasHistory = history.length > 0;
            const greetingText = hasHistory
              ? `${callerName}重新打電話來了，你們之前已經聊過了，自然接著聊`
              : `${callerName}打電話來了，接起來開場`;

            oai.send(JSON.stringify({
              type: 'conversation.item.create',
              item: {
                type: 'message', role: 'user',
                content: [{ type: 'input_text', text: greetingText }],
              },
            }));
            oai.send(JSON.stringify({ type: 'response.create' }));
            console.log('[OAI] Greeting triggered after session.updated');
          }
          break;

        case 'input_audio_buffer.speech_started':
          console.log('[OAI] User speech started');
          clearSilenceTimer(ws);
          resetSilenceNudge(ws);
          ttsActiveMap.set(ws, false); // 恢復音訊轉發
          // Fix #1: cancel ongoing OpenAI response → stop LLM generating tokens
          if (responseActive) {
            try { oaiWs.send(JSON.stringify({ type: 'response.cancel' })); } catch (_) {}
            console.log('[OAI] response.cancel sent');
          }
          // Fix #3: reset state so drainTtsQueue → listening works after interrupt
          responseActive = false;
          sentenceBuffer = '';
          // Abort TTS + clear queue
          if (abortCtrl) {
            abortCtrl.abort();
            abortCtrl = null;
          }
          ttsQueue.length = 0;
          ttsRunning = false;
          isTtsPlaying.set(ws, false);
          // Fix #2: notify client to stop playback + enter listening
          send(ws, { type: 'interrupt' });
          send(ws, { type: 'status', state: 'listening' });
          break;

        case 'input_audio_buffer.speech_stopped':
          console.log('[OAI] User speech stopped');
          send(ws, { type: 'status', state: 'thinking' });
          break;

        case 'response.created':
          responseActive = true;
          sentenceBuffer = '';
          currentEmotion = 'happy';
          fullTextMap.set(ws, '');
          abortCtrl = new AbortController();
          ttsAbortMap.set(ws, abortCtrl);
          ttsQueue.length = 0;
          ttsRunning = false;
          send(ws, { type: 'status', state: 'thinking' });
          break;

        case 'response.text.delta': {
          const delta = msg.delta || '';
          if (!delta) break;

          // Detect emotion tag at start (before buffering to TTS)
          const fullSoFar = (fullTextMap.get(ws) || '') + delta;
          fullTextMap.set(ws, fullSoFar);

          // Extract emotion from accumulated text (only first match)
          const emotionMatch = fullSoFar.match(EMOTION_RE);
          if (emotionMatch) currentEmotion = emotionMatch[1].toLowerCase();

          // Strip emotion/fact tags before buffering for TTS
          const cleanDelta = delta.replace(EMOTION_RE_G, '').replace(FACT_RE, '');
          sentenceBuffer += cleanDelta;

          // Show display transcript (stripped)
          // 也移除尚未關閉的 tag 殘片（streaming 時 [emotion: 可能跨 delta）
          const displayText = fullSoFar.replace(EMOTION_RE_G, '').replace(FACT_RE, '').replace(/\[[^\]]*$/, '').trim();
          send(ws, { type: 'transcript', text: tify(displayText) });

          // Try to flush a sentence
          flushSentence(false);
          break;
        }

        case 'response.text.done': {
          // Flush remaining buffer
          const remaining = sentenceBuffer.trim();
          if (remaining) {
            sentenceBuffer = '';
            const cleanRemaining = sify(remaining).replace(EMOTION_RE_G, '').replace(FACT_RE, '').trim();
            if (cleanRemaining) queueTts(ws, cleanRemaining, currentEmotion, abortCtrl?.signal);
          }

          // Signal frontend to finalize transcript line
          send(ws, { type: 'transcript_done' });

          // Extract and save facts from full response
          const fullText = fullTextMap.get(ws) || '';
          if (fullText) {
            const callerN = callerNames.get(ws);
            extractAndSaveFacts(callerN, fullText);
            const displayClean = tify(fullText.replace(EMOTION_RE_G, '').replace(FACT_RE, '').trim());
            addHistory(callerN, 'assistant', displayClean);
          }
          break;
        }

        case 'response.done':
          responseActive = false;
          console.log('[OAI] Response done');
          break;

        case 'conversation.item.input_audio_transcription.completed':
          // Show user's words in transcript (方案A: 過濾幽靈辨識)
          if (msg.transcript) {
            const userText = tify(msg.transcript);
            if (isValidTranscript(msg.transcript)) {
              console.log('[OAI] User said:', userText);
              send(ws, { type: 'user_transcript', text: userText });
              addHistory(callerNames.get(ws), 'user', userText);
            } else {
              console.log('[OAI] Ghost filtered:', userText);
            }
          }
          break;

        case 'error': {
          const errCode = msg.error?.code || '';
          const errType = msg.error?.type || '';
          const errMsg  = msg.error?.message || 'Unknown';

          // Silently ignore cancel-not-active (normal race condition during interrupts)
          if (errCode === 'response_cancel_not_active') break;

          console.error('[OAI] Error:', JSON.stringify(msg.error));

          // Classify error and give user-friendly Chinese message
          const isServerError = errType === 'server_error' || errMsg.includes('server had an error');
          const isRateLimit   = errCode === 'rate_limit_exceeded' || errType === 'rate_limit';
          const isAuthError   = errCode === 'invalid_api_key' || errCode === 'insufficient_quota';

          let userMessage;
          if (isServerError) {
            userMessage = 'OpenAI 伺服器暫時有問題，正在自動重連...';
            // Auto-reconnect on server error (don't wait for WS close)
            console.log('[OAI] Server error → scheduling reconnect');
            try { oaiWs.close(); } catch (_) {}
          } else if (isRateLimit) {
            userMessage = 'OpenAI 請求太頻繁，請稍等幾秒再說話';
          } else if (isAuthError) {
            userMessage = 'OpenAI API Key 無效或額度不足，請聯繫管理員';
          } else {
            userMessage = 'OpenAI 錯誤: ' + errMsg;
          }
          send(ws, { type: 'error', message: userMessage });
          break;
        }

        default:
          break;
      }
    } catch (e) { console.error('[OAI] Parse:', e.message); }
  });

  oaiWs.on('error', (err) => console.error('[OAI] WS error:', err.message));

  oaiWs.on('close', (code) => {
    console.log('[OAI] Closed:', code, '(attempt', attempt, ')');
    oaiWsMap.delete(ws);
    const MAX_RECONNECT = 5;
    // Auth/protocol errors → don't retry
    const fatalCodes = new Set([4000, 4001, 4003, 4004]);
    if (ws.readyState === 1 && !fatalCodes.has(code) && attempt < MAX_RECONNECT) {
      const delay = Math.min(1000 * Math.pow(2, attempt), 16000);
      console.log(`[OAI] Reconnecting in ${delay}ms (attempt ${attempt + 1}/${MAX_RECONNECT})...`);
      setTimeout(() => {
        if (ws.readyState === 1) createOaiConnection(ws, callerNames.get(ws), attempt + 1);
      }, delay);
    } else if (attempt >= MAX_RECONNECT) {
      console.error('[OAI] Max reconnects reached, giving up');
      send(ws, { type: 'error', message: 'OpenAI 連線失敗，請掛斷重撥' });
    }
  });
}

function closeOai(ws) {
  const oaiWs = oaiWsMap.get(ws);
  if (oaiWs) {
    try { oaiWs.close(); } catch (_) {}
    oaiWsMap.delete(ws);
  }
}

// ── HTTP server ────────────────────────────────────────────────────────────
const mime = {
  '.html': 'text/html; charset=utf-8', '.css':  'text/css',
  '.js':   'application/javascript',   '.json': 'application/json',
  '.png':  'image/png', '.jpg': 'image/jpeg',
  '.ico':  'image/x-icon', '.webmanifest': 'application/manifest+json',
};

const server = createServer((req, res) => {
  const url = req.url === '/' ? '/index.html' : req.url;

  if (url === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      llm: !!OPENAI_API_KEY, tts: !!MINIMAX_API_KEY,
      stt: !!OPENAI_API_KEY,
      connections: wss.clients.size,
    }));
    return;
  }

  if (url === '/api/auth/login' && req.method === 'POST') {
    let body = ''; req.on('data', c => body += c);
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      try {
        const { username, password } = JSON.parse(body || '{}');
        if ((username||'').trim().toUpperCase() === FIXED_USERNAME && String(password) === FIXED_PASSWORD) {
          res.writeHead(200); res.end(JSON.stringify({ token: AUTH_TOKEN, username: FIXED_USERNAME }));
        } else {
          res.writeHead(401); res.end(JSON.stringify({ message: '帳號或密碼錯誤' }));
        }
      } catch { res.writeHead(400); res.end('{}'); }
    });
    return;
  }

  const filePath = join(PUBLIC_DIR, url.split('?')[0]);
  if (!filePath.startsWith(PUBLIC_DIR) || !existsSync(filePath)) {
    res.writeHead(404); res.end('Not Found'); return;
  }
  try {
    const data = readFileSync(filePath);
    res.setHeader('Content-Type', mime[extname(filePath)] || 'application/octet-stream');
    res.writeHead(200); res.end(data);
  } catch { res.writeHead(500); res.end('Error'); }
});

// ── WebSocket server ───────────────────────────────────────────────────────
const wss = new WebSocketServer({ server, path: '/voice' });

wss.on('connection', (ws, req) => {
  const u     = new URL(req.url || '', 'http://localhost');
  const token = u.searchParams.get('token');
  if (token !== AUTH_TOKEN) { ws.close(4001, 'Unauthorized'); return; }

  const callerName = (u.searchParams.get('name') || FIXED_USERNAME).toUpperCase();
  callerNames.set(ws, callerName);
  isTtsPlaying.set(ws, false);
  audioLogMap.set(ws, 0);
  silenceNudgeMap.set(ws, 0);
  console.log('[WS] Connected:', callerName);

  // Railway proxy keepalive
  const pingInterval = setInterval(() => { if (ws.readyState === 1) ws.ping(); }, WS_PING_MS);
  wsPingMap.set(ws, pingInterval);

  // Connect to OpenAI Realtime
  createOaiConnection(ws, callerName);

  ws.on('message', (raw, isBinary) => {
    if (isBinary) {
      // PCM audio from client (48kHz) → resample → OpenAI (24kHz)
      const oaiWs = oaiWsMap.get(ws);
      if (!oaiWs || oaiWs.readyState !== WS.OPEN) return;

      const count = (audioLogMap.get(ws) || 0) + 1; audioLogMap.set(ws, count);
      if (count <= 3 || count % 100 === 0) {
        console.log(`[Audio] frame #${count} ${raw.length}b → OpenAI`);
      }

      try {
        const fromRate = clientSrMap.get(ws) || 48000;
        const pcm24 = resampleTo24k(raw, fromRate);

        // ★ TTS 進行中 → 送靜音給 OpenAI，避免回音觸發 VAD
        // Server-side 攔截，零延遲（不靠 client tts_start 訊息）
        let audioToSend = pcm24;
        if (ttsActiveMap.get(ws)) {
          audioToSend = Buffer.alloc(pcm24.length); // 全零靜音
        }

        oaiWs.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: audioToSend.toString('base64'),
        }));
      } catch (e) { console.error('[Audio] Resample error:', e.message); }
      return;
    }

    // JSON messages
    try {
      const msg = JSON.parse(raw.toString('utf8'));

      if (msg.type === 'audio_config') {
        const sr = Number(msg.sampleRate) || 48000;
        clientSrMap.set(ws, sr);  // 儲存實際 sampleRate
        console.log('[WS] audio_config:', sr, 'Hz');
        return;
      }

      if (msg.type === 'playback_done') {
        // Client finished playing a TTS chunk
        // (silence timer will be started after all TTS drains)
        console.log('[WS] playback_done');
        return;
      }

      if (msg.type === 'text' && msg.text?.trim()) {
        // Manual text input (testing fallback)
        const oaiWs = oaiWsMap.get(ws);
        if (!oaiWs || oaiWs.readyState !== WS.OPEN) return;
        const txt = msg.text.trim();
        send(ws, { type: 'user_transcript', text: txt });
        oaiWs.send(JSON.stringify({
          type: 'conversation.item.create',
          item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: txt }] },
        }));
        oaiWs.send(JSON.stringify({ type: 'response.create' }));
        return;
      }
    } catch (e) { console.error('[WS] Parse error:', e.message); }
  });

  ws.on('close', () => {
    console.log('[WS] Disconnected:', callerName);
    clearSilenceTimer(ws);
    silenceNudgeMap.delete(ws);
    const ping = wsPingMap.get(ws); if (ping) { clearInterval(ping); wsPingMap.delete(ws); }
    const abort = ttsAbortMap.get(ws); if (abort) abort.abort();
    closeOai(ws);
    callerNames.delete(ws);
    clientSrMap.delete(ws);
    isTtsPlaying.delete(ws);
    ttsAbortMap.delete(ws);
    audioLogMap.delete(ws);
    fullTextMap.delete(ws);
    ttsActiveMap.delete(ws);
    // conversationHistory persists by callerName for next call
  });

  ws.on('error', (err) => console.error('[WS] Error:', err.message));
});

// ── Start ──────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n🎙 河北彩花 語音通話 v14 → http://localhost:${PORT}`);
  console.log('OPENAI:',  OPENAI_API_KEY  ? '✅' : '❌  ← 必填！');
  console.log('MINIMAX:', MINIMAX_API_KEY ? '✅' : '❌  ← 必填！');
});
