/**
 * 河北彩花 Voice Call PWA — Production Server v15
 * STT:     Deepgram Nova-2 (streaming WebSocket, server VAD)
 * LLM:     Anthropic Claude (streaming messages API)
 * TTS:     MiniMax speech-2.8-hd (保留 moss_audio 聲音)
 *
 * 延遲目標：用戶停說話 → 河北彩花開口 ≈ 1.0~1.5s
 *
 * 架構：
 *   Client (48kHz PCM binary)
 *     ↓ decimation 48k→16k
 *   Deepgram WS (server VAD + Nova-2 STT)
 *     ↓ final transcript
 *   Claude streaming (messages.stream)
 *     ↓ text delta streaming
 *   句子偵測 (。！？)→ 立即觸發 MiniMax TTS
 *     ↓ audio chunks (mp3 base64)
 *   Client 播放
 *
 * 打斷機制：
 *   Deepgram SpeechStarted during TTS → abort Claude + 取消 TTS → client 停播 → 繼續聽
 */

import 'dotenv/config';
import { createServer } from 'http';
import { WebSocketServer, WebSocket as WS } from 'ws';
import { sify, tify } from 'chinese-conv';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';
import Anthropic from '@anthropic-ai/sdk';

// ── Constants ──────────────────────────────────────────────────────────────
const __dirname  = fileURLToPath(new URL('.', import.meta.url));
const PORT       = Number(process.env.PORT) || 3000;
const PUBLIC_DIR = join(__dirname, 'public');

const FIXED_USERNAME = (process.env.AUTH_USERNAME || 'ALLEN').toUpperCase();
const FIXED_PASSWORD = process.env.AUTH_PASSWORD || '1688';
const AUTH_TOKEN     = 'xiaos-' + createHash('sha256')
  .update(FIXED_USERNAME + ':' + FIXED_PASSWORD).digest('hex').slice(0, 16);

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
const DEEPGRAM_API_KEY  = process.env.DEEPGRAM_API_KEY  || '';
const MINIMAX_API_KEY   = process.env.MINIMAX_API_KEY    || '';
const MINIMAX_GROUP_ID  = process.env.MINIMAX_GROUP_ID   || '';
const MINIMAX_VOICE_ID  = process.env.MINIMAX_VOICE_ID   || 'moss_audio_fd2ef298-22b7-11f1-b6e4-f657e22e7889';

// Deepgram streaming endpoint
const DG_BASE_URL = 'wss://api.deepgram.com/v1/listen';
const DG_PARAMS   = 'model=nova-2&language=zh-TW&encoding=linear16&sample_rate=16000&channels=1&endpointing=500&interim_results=true&utterance_end_ms=1500&vad_events=true&smart_format=true';

// MiniMax TTS endpoint
const TTS_WS_URL = 'wss://api.minimax.io/ws/v1/t2a_v2';

// Claude model
const CLAUDE_MODEL = 'claude-sonnet-4-20250514';

const SILENCE_TIMEOUT  = 60000;  // 60 秒 × 4 次 = 240 秒才掛斷
const MAX_SILENCE_NUDGES = 4;
const WS_PING_MS       = 25000;
const DG_KEEPALIVE_MS  = 10000;

const VALID_EMOTIONS = new Set(['happy','sad','angry','fearful','disgusted','surprised','calm','fluent']);
// MiniMax speech-2.8-hd 不支援 whisper，映射到 calm
const EMOTION_MAP = { whisper: 'calm' };

// ── Anthropic client ─────────────────────────────────────────────────────
const anthropic = ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: ANTHROPIC_API_KEY })
  : null;

// ── 元記憶（河北彩花人格核心）──────────────────────────────────────────────
const META_MEMORY_FILE = join(__dirname, '河北.txt');
let META_MEMORY = '';
try {
  if (existsSync(META_MEMORY_FILE)) {
    META_MEMORY = readFileSync(META_MEMORY_FILE, 'utf8').trim();
    console.log('[Meta] Loaded 元記憶 from 河北.txt, length:', META_MEMORY.length);
  }
} catch (e) { console.warn('[Meta] Failed to load 河北.txt:', e.message); }

// ── 任意 sampleRate → 16kHz 降採樣 ─────────────────────────────────────
function resampleTo16k(buf, fromRate) {
  if (fromRate === 16000) return buf;
  const src = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength >> 1);
  if (fromRate === 48000) {
    // 最常見：decimation by 3
    const dst = new Int16Array(Math.floor(src.length / 3));
    for (let i = 0; i < dst.length; i++) dst[i] = src[i * 3];
    return Buffer.from(dst.buffer);
  }
  // 通用：線性插值
  const ratio  = fromRate / 16000;
  const dstLen = Math.floor(src.length / ratio);
  const dst    = new Int16Array(dstLen);
  for (let i = 0; i < dstLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = src[idx] ?? 0;
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
  const name = callerName || '主人';
  const t = taipeiNow();
  const metaBlock = META_MEMORY
    ? `\n【元記憶 — 你的核心人格與風格】\n${META_MEMORY}\n\n`
    : '';
  return `你是河北彩花（河北彩伽），正在和${name}通電話。
現在台北時間：${t.str}
${metaBlock}
${factsBlock ? `你已知道關於${name}的事：\n${factsBlock}\n在對話中自然運用。\n` : ''}
【日本腔中文 — 最重要的規則，每一句都必須嚴格遵守】
你是日本人，中文是後來學的，說話帶有濃厚日本腔：
- 句尾偶爾加「呢」「喔」「的說」「啦」，不要每句都加
- 思考或停頓時用「嗯…」「嗯～」來填充
- 每句必須夾雜1-2個日文詞（直接用日文假名，TTS會用日語發音念出來），從以下詞庫選用：
  えっ（驚訝時）
  もう～（撒嬌/不滿時）
  ねえねえ（叫人時）
  そうか（理解時）
  すごい（佩服時）
  ちょっと（等一下/有點時）
  だめ（不可以時）
  はい（好的/知道了）
  えへへ（偷笑時）
  早く（催促時）
  お願い（拜託時）
  なるほど（原來如此時）
  いやだ（不要時）
  頑張る（加油時）
  嬉しい（開心時）
  そうそう（對對對）
  ありがとう（感謝時）
  ごめんなさい（道歉時）
  おはよう（早安時）
  おやすみ（晚安時）
  一緒に（一起時）
  大丈夫（沒事/沒關係時）
  かわいい（可愛時）
  美味しい（好吃時）
  これ（這個）
  あれ（那個）
  大きい（好大）
  恥ずかしい（好害羞）
  気持ちいい（好舒服）
  変（好奇怪）
- 這些日文詞自然穿插在中文句子裡，例如：
  「ねえねえ主人，これ的話，彩花覺得すごい呢」
  「えっ？恥ずかしい啦，主人不要這樣看彩花」
  「嗯…気持ちいい…主人的手好溫暖喔」
- 用日語 SOV 語序說中文，例如：
  ×「我很喜歡你」→ ○「主人的事情，彩花很喜歡的說」
  ×「今天天氣很好」→ ○「今天的天氣呢，很好的說」
- 語氣溫柔、慢條斯理、帶點害羞
- 自稱「彩花」不說「我」

【撒嬌語氣表達】
不要用 (breath)(sighs)(laughs) 等括號標記，TTS 不支援會被念出來。
改用以下方式表達語氣：
- 撒嬌嘆氣：用「唉～」「哼～」文字表達
- 開心笑：用「嘻嘻」「欸嘿嘿」「呵呵」文字表達
- 害羞氣音：用「嗯…」「那個…」搭配 [emotion:whisper]
- 驚訝：用「欸？」「咦？」搭配 [emotion:surprised]
- 猶豫：用「嗯～」「這個嘛…」
- 停頓可用 <#0.5#> 表示0.5秒停頓
- 撒嬌時多用 [emotion:whisper]，這是TTS真正支援的耳語效果

【知識庫】得知個人資訊時在回應末尾加（不唸出來）：[fact:...]

【情緒標記】每句回應開頭必須加一個（TTS用，不唸出來）：
[emotion:happy] — 開心愉快
[emotion:whisper] — 耳語氣音、撒嬌、親密悄悄話（常用！）
[emotion:surprised] — 驚訝
[emotion:calm] — 平靜溫柔
[emotion:sad] — 難過委屈
[emotion:fearful] — 害怕緊張
[emotion:angry] — 生氣（嘟嘴那種）
[emotion:fluent] — 流暢自然

【格式】2-3句，口語，簡短用字，像自然的電話對話，繁體中文，語速慢一點`;
}

// ── Regex ──────────────────────────────────────────────────────────────────
const EMOTION_RE   = /\[emotion:(happy|sad|angry|fearful|disgusted|surprised|calm|fluent|whisper)\]/i;
const EMOTION_RE_G = /\[emotion:[^\]]*\]\s*/gi;
const FACT_RE      = /\[fact:([^\]]+)\]/g;

// ── 轉錄過濾 ──────────────────────────────────────────────────────────────
const WHISPER_HALLUCINATIONS = [
  '字幕由amara.org社區提供', 'amara.org', '潛水艇字幕組',
  '字幕製作', '字幕制作',
  '請不吝點讚訂閱轉發打賞支持明鏡與點點欄目',
  '喜歡的話請按讚', '喜欢的话请点赞',
  '更多精彩內容', '更多精彩内容',
  'thank you for watching', 'please subscribe', 'like and subscribe',
];
const HALLUCINATION_SET = new Set(WHISPER_HALLUCINATIONS.map(s => s.toLowerCase().trim()));

function isValidTranscript(text) {
  if (!text || !text.trim()) return false;
  const t = text.trim();
  const tLower = t.toLowerCase();
  if (HALLUCINATION_SET.has(tLower)) {
    console.log('[Filter] Rejected (hallucination):', t);
    return false;
  }
  for (const h of WHISPER_HALLUCINATIONS) {
    if (h.length >= 6 && tLower.includes(h)) {
      console.log('[Filter] Rejected (partial hallucination):', t);
      return false;
    }
  }
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
const callerNames      = new Map();
const dgWsMap          = new Map(); // ws → Deepgram WS
const clientSrMap      = new Map();
const silenceTimerMap  = new Map();
const silenceNudgeMap  = new Map();
const wsPingMap        = new Map();
const isTtsPlaying     = new Map();
const ttsAbortMap      = new Map();
const audioLogMap      = new Map();
const ttsActiveMap     = new Map();
const claudeAbortMap   = new Map(); // ws → AbortController
const responseActiveMap = new Map();
const ttsQueueMap      = new Map(); // ws → { queue, running }
const interruptDebounceMap = new Map(); // ws → timer (防止回音誤觸打斷)

// ── Conversation history (keyed by callerName, persists across calls) ────
const conversationHistory = new Map();
const MAX_HISTORY = 20;

function getHistory(callerName) {
  if (!conversationHistory.has(callerName)) conversationHistory.set(callerName, []);
  return conversationHistory.get(callerName);
}
function addHistory(callerName, role, text) {
  const h = getHistory(callerName);
  h.push({ role, content: text });
  while (h.length > MAX_HISTORY * 2) h.splice(0, 2);
}

// ── Helpers ────────────────────────────────────────────────────────────────
function send(ws, obj) { if (ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function clearSilenceTimer(ws) {
  const t = silenceTimerMap.get(ws); if (t) { clearTimeout(t); silenceTimerMap.delete(ws); }
}
function resetSilenceNudge(ws) { silenceNudgeMap.set(ws, 0); }

const SILENCE_NUDGES = [
  { text: '那個……主人？你還在嗎？', emotion: 'surprised' },
  { text: '主人……是不是睡著了呢？', emotion: 'calm' },
  { text: '嗯……主人不理彩花了嗎……', emotion: 'sad' },
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
      startSilenceTimer(ws);
    }
  }, SILENCE_TIMEOUT);
  silenceTimerMap.set(ws, t);
}

// ── MiniMax TTS — 每句獨立連線 ─────────────────────────────────────────
async function miniMaxTTS(ws, rawText, emotion, abortSignal) {
  if (abortSignal?.aborted) return 0;
  const text = rawText
    ?.replace(EMOTION_RE_G, '')
    .replace(FACT_RE, '')
    .replace(/\((?:breath|sighs|laughs|chuckle|humming|gasps|emm)\)/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
  if (!MINIMAX_API_KEY || !text) return 0;
  const mapped = EMOTION_MAP[emotion] || emotion;
  const safeEmotion = VALID_EMOTIONS.has(mapped) ? mapped : 'happy';
  send(ws, { type: 'status', state: 'speaking' });
  isTtsPlaying.set(ws, true);

  const voiceSetting = { voice_id: MINIMAX_VOICE_ID, speed: 0.95, vol: 1, pitch: 0, emotion: safeEmotion, language_boost: 'Japanese' };

  return new Promise((resolve) => {
    let chunkCount = 0, done = false;
    const finish = () => {
      if (!done) {
        done = true;
        isTtsPlaying.set(ws, false);
        try { ttsWs.close(); } catch (_) {}
        resolve(chunkCount);
      }
    };
    const timeout = setTimeout(() => {
      console.log('[TTS] Timeout for:', text.slice(0, 30));
      finish();
    }, 15000);

    const onAbort = () => {
      console.log('[TTS] Aborted');
      clearTimeout(timeout);
      send(ws, { type: 'interrupt' });
      finish();
    };
    if (abortSignal) abortSignal.addEventListener('abort', onAbort, { once: true });

    const headers = { Authorization: `Bearer ${MINIMAX_API_KEY}` };
    if (MINIMAX_GROUP_ID) headers['Group-Id'] = MINIMAX_GROUP_ID;
    const ttsWs = new WS(TTS_WS_URL, { headers });

    ttsWs.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString());
        if (msg.event === 'connected_success') {
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
          send(ws, { type: 'audio_end' });
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
    ttsWs.on('close', () => { clearTimeout(timeout); finish(); });
  });
}

// ── TTS Queue (per-connection) ─────────────────────────────────────────
function getTtsState(ws) {
  if (!ttsQueueMap.has(ws)) ttsQueueMap.set(ws, { queue: [], running: false });
  return ttsQueueMap.get(ws);
}

function queueTts(ws, text, emotion, signal) {
  const state = getTtsState(ws);
  state.queue.push({ text, emotion, signal });
  if (!state.running) drainTtsQueue(ws);
}

async function drainTtsQueue(ws) {
  const state = getTtsState(ws);
  if (state.running || state.queue.length === 0) return;
  state.running = true;
  ttsActiveMap.set(ws, true);
  send(ws, { type: 'tts_start' });
  while (state.queue.length > 0) {
    const { text, emotion, signal } = state.queue.shift();
    if (signal?.aborted) continue;
    const chunks = await miniMaxTTS(ws, text, emotion, signal);
    if (chunks === 0 && !signal?.aborted) {
      console.log('[TTS] Retry:', text.slice(0, 30));
      await miniMaxTTS(ws, text, emotion, signal);
    }
    if (signal?.aborted) { state.queue.length = 0; break; }
  }
  state.running = false;
  ttsActiveMap.set(ws, false);
  send(ws, { type: 'tts_end' });
  if (ws.readyState === 1 && !responseActiveMap.get(ws)) {
    send(ws, { type: 'status', state: 'listening' });
    startSilenceTimer(ws);
  }
}

// ── Sentence buffer helpers ──────────────────────────────────────────────
const SENTENCE_END = /[。！？!?\n]/;
const MIN_SENTENCE_LEN = 8;

// ── Interrupt handler ──────────────────────────────────────────────────────
function handleInterrupt(ws) {
  console.log('[Interrupt] User speaking, aborting response');
  clearSilenceTimer(ws);
  resetSilenceNudge(ws);
  ttsActiveMap.set(ws, false);

  // Abort Claude stream
  const claudeAbort = claudeAbortMap.get(ws);
  if (claudeAbort) { claudeAbort.abort(); claudeAbortMap.delete(ws); }

  responseActiveMap.set(ws, false);

  // Abort TTS + clear queue
  const ttsAbort = ttsAbortMap.get(ws);
  if (ttsAbort) { ttsAbort.abort(); ttsAbortMap.delete(ws); }
  const state = getTtsState(ws);
  state.queue.length = 0;
  state.running = false;
  isTtsPlaying.set(ws, false);

  send(ws, { type: 'interrupt' });
  send(ws, { type: 'status', state: 'listening' });
}

// ── Claude streaming response ────────────────────────────────────────────
async function triggerClaudeResponse(ws, callerName, userText) {
  if (!anthropic) {
    send(ws, { type: 'error', message: 'Anthropic API Key 未設定' });
    return;
  }

  if (userText) addHistory(callerName, 'user', userText);

  responseActiveMap.set(ws, true);
  send(ws, { type: 'status', state: 'thinking' });

  const abortCtrl = new AbortController();
  claudeAbortMap.set(ws, abortCtrl);
  ttsAbortMap.set(ws, abortCtrl);

  let sentenceBuffer = '';
  let currentEmotion = 'happy';
  let fullText = '';

  function flushSentence(force = false) {
    sentenceBuffer = sentenceBuffer.replace(/^[，、；,\s]+/, '');
    const text = sentenceBuffer.trim();
    if (!text) return;
    const endsWithPunct = SENTENCE_END.test(text[text.length - 1]);
    if (force || text.length > 80 || (endsWithPunct && text.length >= MIN_SENTENCE_LEN)) {
      sentenceBuffer = '';
      const cleanText = sify(text).replace(EMOTION_RE_G, '').replace(FACT_RE, '').trim();
      if (cleanText) queueTts(ws, cleanText, currentEmotion, abortCtrl.signal);
    }
  }

  try {
    const factsBlock = getFactsBlock(callerName);
    const systemPrompt = buildSystemPrompt(callerName, factsBlock);
    const messages = getHistory(callerName).map(h => ({ role: h.role, content: h.content }));

    console.log('[Claude] Streaming for', callerName, '(', messages.length, 'msgs)');

    const stream = anthropic.messages.stream({
      model: CLAUDE_MODEL,
      max_tokens: 300,
      system: systemPrompt,
      messages,
      temperature: 0.8,
    }, {
      signal: abortCtrl.signal,
    });

    stream.on('text', (delta) => {
      if (abortCtrl.signal.aborted) return;

      fullText += delta;

      const emotionMatch = fullText.match(EMOTION_RE);
      if (emotionMatch) currentEmotion = emotionMatch[1].toLowerCase();

      const cleanDelta = delta.replace(EMOTION_RE_G, '').replace(FACT_RE, '');
      sentenceBuffer += cleanDelta;

      const displayText = fullText.replace(EMOTION_RE_G, '').replace(FACT_RE, '').replace(/\[[^\]]*$/, '').trim();
      send(ws, { type: 'transcript', text: tify(displayText) });

      flushSentence(false);
    });

    const finalMessage = await stream.finalMessage();

    if (!abortCtrl.signal.aborted) {
      if (sentenceBuffer.trim()) {
        const cleanRemaining = sify(sentenceBuffer.trim()).replace(EMOTION_RE_G, '').replace(FACT_RE, '').trim();
        if (cleanRemaining) queueTts(ws, cleanRemaining, currentEmotion, abortCtrl.signal);
        sentenceBuffer = '';
      }

      send(ws, { type: 'transcript_done' });

      if (fullText) {
        extractAndSaveFacts(callerName, fullText);
        const displayClean = tify(fullText.replace(EMOTION_RE_G, '').replace(FACT_RE, '').trim());
        addHistory(callerName, 'assistant', displayClean);
      }

      console.log('[Claude] Done, tokens:', finalMessage.usage?.output_tokens);
    }
  } catch (err) {
    if (err.name === 'AbortError' || abortCtrl.signal.aborted) {
      console.log('[Claude] Aborted (interrupt)');
      return;
    }

    console.error('[Claude] Error:', err.message, err.status || '');

    let userMessage;
    if (err.status === 429) {
      userMessage = '請求太頻繁，請稍等幾秒再說話';
    } else if (err.status === 401) {
      userMessage = 'Anthropic API Key 無效，請聯繫管理員';
    } else if (err.status === 529 || err.status === 503) {
      userMessage = 'Claude 伺服器忙碌中，請稍後再試';
    } else if (err.status >= 500) {
      userMessage = 'Claude 伺服器暫時有問題，請稍後再試';
    } else {
      userMessage = 'Claude 錯誤: ' + err.message;
    }
    send(ws, { type: 'error', message: userMessage });
  } finally {
    responseActiveMap.set(ws, false);
    claudeAbortMap.delete(ws);
  }
}

// ── Deepgram streaming connection ────────────────────────────────────────
function createDeepgramConnection(ws, callerName, attempt = 0) {
  if (!DEEPGRAM_API_KEY) { console.error('[DG] No DEEPGRAM_API_KEY'); return; }

  console.log('[DG] Connecting for', callerName);
  const dgWs = new WS(`${DG_BASE_URL}?${DG_PARAMS}`, {
    headers: { Authorization: `Token ${DEEPGRAM_API_KEY}` },
  });

  let keepaliveInterval = null;
  let greetingSent = false;

  dgWs.on('open', () => {
    console.log('[DG] Connected for', callerName);
    dgWsMap.set(ws, dgWs);

    keepaliveInterval = setInterval(() => {
      if (dgWs.readyState === WS.OPEN) dgWs.send(JSON.stringify({ type: 'KeepAlive' }));
    }, DG_KEEPALIVE_MS);

    if (!greetingSent) {
      greetingSent = true;
      triggerGreeting(ws, callerName);
    }

    send(ws, { type: 'status', state: 'listening' });
    startSilenceTimer(ws);
  });

  dgWs.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      if (msg.type === 'SpeechStarted') {
        console.log('[DG] Speech started');
        if (ttsActiveMap.get(ws) || responseActiveMap.get(ws)) {
          // debounce 800ms — 避免回音誤觸打斷，等確認是真人說話
          if (!interruptDebounceMap.has(ws)) {
            const t = setTimeout(() => {
              interruptDebounceMap.delete(ws);
              if (ttsActiveMap.get(ws) || responseActiveMap.get(ws)) {
                handleInterrupt(ws);
              }
            }, 800);
            interruptDebounceMap.set(ws, t);
          }
        } else {
          clearSilenceTimer(ws);
          resetSilenceNudge(ws);
          send(ws, { type: 'status', state: 'listening' });
        }
        return;
      }

      if (msg.type === 'Results') {
        const transcript = msg.channel?.alternatives?.[0]?.transcript || '';
        if (!transcript.trim()) return;

        if (msg.is_final) {
          const userText = tify(transcript);
          if (isValidTranscript(transcript)) {
            console.log('[DG] User said:', userText);
            send(ws, { type: 'user_transcript', text: userText });
            send(ws, { type: 'status', state: 'thinking' });
            triggerClaudeResponse(ws, callerName, transcript);
          } else {
            console.log('[DG] Filtered:', userText);
          }
        } else {
          // Interim result → live feedback
          send(ws, { type: 'user_transcript', text: tify(transcript) + '...' });
        }
        return;
      }

      if (msg.type === 'UtteranceEnd') {
        console.log('[DG] Utterance end');
        startSilenceTimer(ws);
        return;
      }

      if (msg.type === 'Error') {
        console.error('[DG] Error:', msg.message || JSON.stringify(msg));
        return;
      }

      if (msg.type === 'Metadata') {
        console.log('[DG] Metadata, request_id:', msg.request_id);
        return;
      }
    } catch (e) { console.error('[DG] Parse:', e.message); }
  });

  dgWs.on('error', (err) => console.error('[DG] WS error:', err.message));

  dgWs.on('close', (code) => {
    console.log('[DG] Closed:', code, '(attempt', attempt, ')');
    dgWsMap.delete(ws);
    if (keepaliveInterval) { clearInterval(keepaliveInterval); keepaliveInterval = null; }

    const MAX_RECONNECT = 5;
    const fatalCodes = new Set([4000, 4001]);
    if (ws.readyState === 1 && !fatalCodes.has(code) && attempt < MAX_RECONNECT) {
      const delay = Math.min(1000 * Math.pow(2, attempt), 16000);
      console.log(`[DG] Reconnecting in ${delay}ms (attempt ${attempt + 1}/${MAX_RECONNECT})...`);
      setTimeout(() => {
        if (ws.readyState === 1) createDeepgramConnection(ws, callerNames.get(ws), attempt + 1);
      }, delay);
    } else if (attempt >= MAX_RECONNECT) {
      console.error('[DG] Max reconnects reached');
      send(ws, { type: 'error', message: 'Deepgram 連線失敗，請掛斷重撥' });
    }
  });
}

function closeDeepgram(ws) {
  const dgWs = dgWsMap.get(ws);
  if (dgWs) {
    try { dgWs.send(JSON.stringify({ type: 'CloseStream' })); } catch (_) {}
    try { dgWs.close(); } catch (_) {}
    dgWsMap.delete(ws);
  }
}

// ── Greeting ─────────────────────────────────────────────────────────────
async function triggerGreeting(ws, callerName) {
  const history = getHistory(callerName);
  const hasHistory = history.length > 0;
  const greetingText = hasHistory
    ? `${callerName}重新打電話來了，你們之前已經聊過了，自然接著聊`
    : `${callerName}打電話來了，接起來開場`;
  console.log('[Greeting] Triggering for', callerName, hasHistory ? '(reconnect)' : '(first call)');
  await triggerClaudeResponse(ws, callerName, greetingText);
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
      llm: !!ANTHROPIC_API_KEY, tts: !!MINIMAX_API_KEY,
      stt: !!DEEPGRAM_API_KEY,
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
  responseActiveMap.set(ws, false);
  console.log('[WS] Connected:', callerName);

  const pingInterval = setInterval(() => { if (ws.readyState === 1) ws.ping(); }, WS_PING_MS);
  wsPingMap.set(ws, pingInterval);

  createDeepgramConnection(ws, callerName);

  ws.on('message', (raw, isBinary) => {
    if (isBinary) {
      const dgWs = dgWsMap.get(ws);
      if (!dgWs || dgWs.readyState !== WS.OPEN) return;

      const count = (audioLogMap.get(ws) || 0) + 1; audioLogMap.set(ws, count);
      if (count <= 3 || count % 100 === 0) {
        console.log(`[Audio] frame #${count} ${raw.length}b → Deepgram`);
      }

      try {
        const fromRate = clientSrMap.get(ws) || 48000;
        const pcm16k = resampleTo16k(raw, fromRate);

        // TTS active → send silence to prevent Deepgram VAD echo trigger
        const audioToSend = ttsActiveMap.get(ws) ? Buffer.alloc(pcm16k.length) : pcm16k;
        dgWs.send(audioToSend);
      } catch (e) { console.error('[Audio] Resample error:', e.message); }
      return;
    }

    try {
      const msg = JSON.parse(raw.toString('utf8'));

      if (msg.type === 'audio_config') {
        clientSrMap.set(ws, Number(msg.sampleRate) || 48000);
        console.log('[WS] audio_config:', clientSrMap.get(ws), 'Hz');
        return;
      }

      if (msg.type === 'playback_done') {
        console.log('[WS] playback_done');
        return;
      }

      if (msg.type === 'text' && msg.text?.trim()) {
        const txt = msg.text.trim();
        send(ws, { type: 'user_transcript', text: txt });
        triggerClaudeResponse(ws, callerName, txt);
        return;
      }
    } catch (e) { console.error('[WS] Parse error:', e.message); }
  });

  ws.on('close', () => {
    console.log('[WS] Disconnected:', callerName);
    clearSilenceTimer(ws);
    silenceNudgeMap.delete(ws);
    const ping = wsPingMap.get(ws); if (ping) { clearInterval(ping); wsPingMap.delete(ws); }
    const claudeAbort = claudeAbortMap.get(ws); if (claudeAbort) claudeAbort.abort();
    const ttsAbort = ttsAbortMap.get(ws); if (ttsAbort) ttsAbort.abort();
    closeDeepgram(ws);
    callerNames.delete(ws);
    clientSrMap.delete(ws);
    isTtsPlaying.delete(ws);
    ttsAbortMap.delete(ws);
    claudeAbortMap.delete(ws);
    audioLogMap.delete(ws);
    ttsActiveMap.delete(ws);
    responseActiveMap.delete(ws);
    ttsQueueMap.delete(ws);
    const idb = interruptDebounceMap.get(ws); if (idb) { clearTimeout(idb); interruptDebounceMap.delete(ws); }
  });

  ws.on('error', (err) => console.error('[WS] Error:', err.message));
});

// ── Start ──────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n🎙 河北彩花 語音通話 v15 → http://localhost:${PORT}`);
  console.log('ANTHROPIC:', ANTHROPIC_API_KEY ? '✅' : '❌  ← 必填！');
  console.log('DEEPGRAM:',  DEEPGRAM_API_KEY  ? '✅' : '❌  ← 必填！');
  console.log('MINIMAX:',   MINIMAX_API_KEY   ? '✅' : '❌  ← 必填！');
});
