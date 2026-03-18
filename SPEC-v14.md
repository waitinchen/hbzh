# 河北彩花 語音通話 PWA — 系統規格書 v14

## Context
完整的系統架構與基礎設施規格書，作為版本紀錄留存。本文件記錄截至 2026-03-14 的系統現況。

---

## 一、系統概覽

| 項目 | 值 |
|------|---|
| 專案名 | xiaos-call-pwa |
| 版本 | Server v14 / Frontend v12 / SW v20260314b |
| 部署 | Railway (GitHub auto-deploy) |
| 域名 | ssd.tonetown.ai |
| 架構 | OpenAI Realtime (STT+LLM) → MiniMax TTS → PWA Client |

### 延遲實測
用戶停說話 → 河北彩花開口 ≈ 2.8~3.1s

### 流程圖
```
用戶手機 (PWA)
  ↓ 48kHz PCM binary (WebSocket)
Server (Node.js on Railway)
  ↓ decimation 48k→24k
OpenAI Realtime WS (server VAD + GPT-4o, text-only)
  ↓ response.text.delta streaming
句子偵測 (。！？ / 80字上限)
  ↓ 立即觸發 MiniMax TTS (預連線複用)
MiniMax WS (speech-2.8-hd, 1.15x 語速)
  ↓ MP3 base64 chunks streaming
Client (Web Audio API 精確排程, 零間隙播放)
```

---

## 二、檔案結構

```
xiaos-call-pwa/
├── server.js              # 主伺服器 (v14)
├── package.json           # 依賴管理
├── Dockerfile             # Docker 建置 (node:20-slim)
├── railway.toml           # Railway 部署設定
├── CLAUDE.md              # 開發規範
├── .env / .env.example    # 環境變數
├── scripts/
│   └── check-minimax-key.js  # MiniMax key 驗證工具
└── public/
    ├── index.html         # 三頁式 SPA
    ├── app.js             # 前端邏輯 (v12)
    ├── style.css          # 霓 Né 設計系統
    ├── sw.js              # Service Worker (v20260314b)
    ├── manifest.json      # PWA manifest
    ├── avatar.png         # 河北彩花 頭像
    ├── favicon.ico        # 網站圖標
    └── icons/
        ├── icon-192.png   # PWA 圖標
        └── icon-512.png   # PWA 圖標
```

---

## 三、環境變數

| 變數 | 必填 | 說明 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key (需 Realtime API 權限) |
| `MINIMAX_API_KEY` | ✅ | MiniMax TTS API key |
| `MINIMAX_GROUP_ID` | ✅ | MiniMax Group ID |
| `MINIMAX_VOICE_ID` | ❌ | 聲音 ID (預設 moss_audio) |
| `AUTH_USERNAME` | ❌ | 登入帳號 (預設 ALLEN) |
| `AUTH_PASSWORD` | ❌ | 登入密碼 (預設 1688) |
| `PORT` | ❌ | 伺服器埠 (預設 3000) |

---

## 四、Server 架構 (server.js)

### 4.1 依賴
| 套件 | 版本 | 用途 |
|------|------|------|
| ws | 8.18.0 | WebSocket 伺服器 + 客戶端 |
| chinese-conv | 4.0.0 | sify(簡體) / tify(繁體) 轉換 |
| dotenv | 16.4.5 | .env 載入 |

### 4.2 API 端點
| Method | Path | 功能 |
|--------|------|------|
| GET | `/` | 靜態首頁 (index.html) |
| GET | `/api/health` | 健康檢查 {status, llm, tts, stt, connections} |
| POST | `/api/auth/login` | 帳號密碼驗證，回傳 SHA256 token |
| WS | `/voice` | 語音通話 WebSocket (需 token query param) |

### 4.3 OpenAI Realtime 設定
| 參數 | 值 |
|------|---|
| 模型 | `gpt-4o-realtime-preview` |
| modalities | `['text']` (不要 OpenAI 音訊輸出) |
| input_audio_format | `pcm16` (24kHz mono) |
| transcription | `whisper-1`, language: `zh` |
| VAD type | `server_vad` |
| VAD threshold | `0.45` (第二道防線，client 有動態噪音底板) |
| prefix_padding_ms | `300` |
| silence_duration_ms | `500` |
| temperature | `0.8` |

### 4.4 MiniMax TTS 設定
| 參數 | 值 |
|------|---|
| 模型 | `speech-2.8-hd` |
| 聲音 | `moss_audio_d739901e-1d39-11f1-9b14-6299e7260fda` |
| 語速 | `1.15` (比預設快 15%) |
| 取樣率 | `32000 Hz` |
| 位元率 | `128000 bps` |
| 格式 | `mp3` mono |
| 連線策略 | **預連線複用** — 通話開始建立一條 WS，所有句子重複使用 |

### 4.5 句子偵測與 TTS 排程
- 句尾偵測：`/[。！？!?\n]/`
- 最小句長：`8` 字（太短先累積）
- 最大累積：`80` 字（強制 flush）
- 句首清理：去掉殘留 `，、；,` 等標點
- 文字轉換：TTS 用 `sify()` (簡體)，前端顯示用 `tify()` (繁體)
- 情緒標記：從 GPT 回應提取 `[emotion:xxx]`，傳給 MiniMax
- TTS queue：sequential drain，保證句子順序

### 4.6 打斷機制 (speech_started)
```
OpenAI 偵測到用戶說話
  → response.cancel (停止 GPT 繼續生成)
  → responseActive = false
  → abortCtrl.abort() (中止當前 TTS)
  → ttsQueue 清空, ttsRunning = false
  → send 'interrupt' (client 停播所有排程音訊)
  → send status: 'listening' (UI 切回聆聽)
```

### 4.7 連線管理
| 項目 | 值 |
|------|---|
| Railway keepalive ping | 每 25s |
| 靜音超時 | 20s → 河北彩花提醒 → 自動掛斷 |
| OpenAI 重連 | 指數退避 1s→2s→4s→8s→16s，最多 5 次 |
| 致命錯誤碼 (不重連) | 4000, 4001, 4003, 4004 |

### 4.8 記憶系統
| 層級 | 機制 | 儲存 | 跨部署 |
|------|------|------|--------|
| 短期 (對話歷史) | conversationHistory Map | 記憶體 | ❌ 重啟即失 |
| 長期 (知識庫) | user_facts.json | 容器磁碟 | ❌ 重建即失 |

- 對話歷史：最多保留 20 輪 (MAX_HISTORY)
- 知識庫：GPT 輸出 `[fact:...]` 自動提取儲存
- 重撥注入：assistant 用 `type:'text'`、user 用 `type:'input_text'`

### 4.9 System Prompt 關鍵要素
- 角色：河北彩花（河北彩伽）
- 暱稱：ALLEN → 「A冷」
- 風格：台灣口語、短句快節奏、語氣詞（吼、欸、哎唷、拜託）
- 毒舌但有愛，不說教、不端架子
- 台北時間感知 (`taipeiNow()`)：自然融入時間相關對話
- 情緒標記 `[emotion:xxx]`：8 種情緒供 TTS 使用
- 知識庫 `[fact:...]`：記住用戶個人資訊
- 格式：2-4 句，口語，不條列，繁體中文

---

## 五、Frontend 架構 (app.js)

### 5.1 頁面結構 (三頁式 SPA)
1. **登入頁** (#page-login) — 帳號密碼表單
2. **首頁** (#page-home) — 健康狀態指示 + 撥打按鈕 + PWA 安裝提示
3. **通話頁** (#page-call) — 頭像 + 狀態 + 波形 + 通話紀錄 HUD + 操作按鈕

### 5.2 音訊輸入管線
```
getUserMedia({
  echoCancellation: true,    // 瀏覽器原生回音消除
  noiseSuppression: true,    // 背景噪音抑制
  autoGainControl: true      // 自動增益
})
  ↓ ScriptProcessor (bufferSize: 4096, mono)
  ↓ 動態噪音底板 VAD (只影響 UI，不擋音訊)
  ↓ Float32 → Int16 PCM
  ↓ WebSocket binary → Server (永遠送，server VAD 需要靜音段)
```

### 5.3 動態噪音底板 VAD
| 參數 | 值 |
|------|---|
| 絕對最低門檻 | `0.004` |
| 初始噪音底板 | `0.008` |
| 更新公式 | `noiseFloor = noiseFloor × 0.95 + rms × 0.05` |
| 有效門檻 | `max(0.004, noiseFloor × 2.0)` |
| 校準條件 | 靜音幀連續 > 10 才開始學習 |
| 作用範圍 | 只控制 UI 狀態切換 (transcribing ↔ thinking) |
| 本地語音 timer | 1.5s 靜音 → transcribing → thinking |

### 5.4 音訊播放 (Web Audio API 零間隙)
```
收到 base64 chunk
  ↓ atob → Uint8Array
  ↓ decodeAudioData() (立即解碼)
  ↓ push to pendingBuffers[]
  ↓ scheduleBuffers()
     ↓ AudioBufferSourceNode.start(nextPlayTime)
     ↓ connect → playbackAnalyser → destination
     ↓ nextPlayTime += buffer.duration (精確累加)
```
- 句間間隙：~0ms（vs 舊版 HTMLAudioElement 100-200ms）
- 打斷：`source.stop()` 停止所有排程中的 source

### 5.5 波形視覺化
- 雙通道：綠色 (playback output) + 紫色 (mic input)
- ECG 心電圖風格：`Math.pow(Math.abs(Math.sin(...)), 3)` 尖峰波形
- Peak detection（比 RMS 更敏感）
- 敏感度倍數：mic x3, playback x2.5
- fftSize: mic=512, playback=256

### 5.6 通話狀態機
```
idle → ringing (鈴聲 440Hz) → connected (通話中) → ended → idle
```

### 5.7 通話紀錄 HUD
- 浮動定位 (absolute, 不擋按鈕)
- CSS mask-image 上下淡出漸層
- 串流文字：`streamingLine` 同一行即時更新（不重複）
- `transcript_done` 事件凍結當前行，下次新開一行

---

## 六、UI 設計系統 (style.css)

### 6.1 色彩系統
| Token | 值 | 用途 |
|-------|---|------|
| `--neon-green` | #00D26A | 主色 (LINE green) |
| `--neon-violet` | #8B5CF6 | 輔色 (mic 波形) |
| `--danger` | #FF3B30 | 掛斷 / 錯誤 |
| `--black` | #000000 | 全黑背景 |
| `--text-primary` | #F5F5F7 | 主文字 |
| `--text-secondary` | #86868B | 次要文字 |

### 6.2 間距 (黃金比例)
`4px → 8px → 16px → 24px → 40px → 64px`

### 6.3 動畫
| 名稱 | 週期 | 效果 |
|------|------|------|
| ringPulse | 2s | 來電脈衝光環 |
| breathe | 4s | 背景呼吸漸層 |
| textPulse | 1.5s | 狀態文字閃爍 |
| fadeIn | 400ms | 頁面切換滑入 |
| neonGlow | 3s | 霓虹 box-shadow |

### 6.4 響應式
- 手機 (<375px)：按鈕 56x56
- 手機 (>=375px)：按鈕 64x64
- 矮螢幕 (<600px)：緊湊佈局

---

## 七、PWA 配置

### 7.1 manifest.json
| 欄位 | 值 |
|------|---|
| name | 打電話給河北彩花 |
| short_name | 河北彩花來電 |
| display | standalone |
| orientation | portrait-primary |
| theme_color | #000000 |
| icons | 192x192 + 512x512 (maskable) |

### 7.2 Service Worker (sw.js)
- 快取策略：Cache-first (靜態資源)
- API / WS 請求：直通網路，不快取
- 版本控制：日期版本號 (`20260314b`)
- 更新機制：`skipWaiting()` + `clients.claim()` 立即接管
- 快取資源：`/, /index.html, /style.css, /app.js, /manifest.json, /favicon.ico, /avatar.png`

---

## 八、部署基礎設施

### 8.1 Railway
| 項目 | 值 |
|------|---|
| 平台 | Railway.app |
| 建置方式 | Dockerfile (node:20-slim) |
| root_dir | xiaos-call-pwa |
| 健康檢查 | `GET /api/health` |
| 重啟策略 | on_failure |
| 部署觸發 | GitHub push main → auto-deploy |
| 域名 | ssd.tonetown.ai (custom domain) |

### 8.2 Dockerfile
```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . ./
EXPOSE 3000
CMD ["node", "server.js"]
```

---

## 九、延遲拆解 (實測數據)

| 階段 | 耗時 | 佔比 |
|------|------|------|
| VAD 靜音偵測 | 500ms | 16% |
| GPT 生成第一句 | ~700ms | 23% |
| MiniMax TTS 合成 | ~1600ms | 52% |
| 網路傳輸 + 解碼 | ~300ms | 9% |
| **總計** | **~3.1s** | 100% |

> 瓶頸在 MiniMax TTS。預連線複用已省 ~300ms。

---

## 十、已知限制

1. **記憶不跨部署** — user_facts.json 和 conversationHistory 在容器重建後消失
2. **iOS 無 PWA 安裝按鈕** — Apple 不支援 `beforeinstallprompt`，需手動「分享→加入主畫面」
3. **TTS 延遲瓶頸** — MiniMax 合成 ≈ 1.6s，佔總延遲 52%
4. **ScriptProcessor 已棄用** — 應遷移至 AudioWorklet（但目前相容性更好）
5. **單一帳號** — 固定帳密，無多用戶系統
6. **無 HTTPS 本地開發** — 麥克風需要 HTTPS 或 localhost

---

*文件產出日期：2026-03-14*
*基於 commit: 63ac801*
