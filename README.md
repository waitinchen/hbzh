# 打電話給河北彩花（PWA）

固定帳密登入後，即可在手機或電腦上「打電話給河北彩花」進行即時語音對話。

- **帳號**：`WAITIN`
- **密碼**：`1688`

---

## 快速開始

### 1. 安裝依賴

```bash
cd xiaos-call-pwa
npm install
```

### 2. 環境變數（選填，語音回覆需 MiniMax）

複製 `realtime-voice-app/backend/.env.example` 的 `MINIMAX_API_KEY` 到本目錄，或新增：

```bash
# .env（選填）
MINIMAX_API_KEY=你的_MiniMax_API_Key
# MINIMAX_GROUP_ID=  # 若官方要求再填
```

### 3. 啟動

```bash
npm start
```

瀏覽器打開 **http://localhost:3000**：

1. 輸入帳號 `WAITIN`、密碼 `1688` → 登入  
2. 點「📞 打電話給河北彩花」→ 進入通話頁  
3. 點「📝 用文字說」輸入文字，即可收到河北彩花語音回覆（需設定 `MINIMAX_API_KEY`）

---

## 手機 PWA 安裝

1. 用 **HTTPS** 部署本服務（或本機用 `localhost` 測試）。  
2. 手機瀏覽器打開該網址 → 登入後可「加到主畫面」當 App 使用。  
3. 圖示：將 192x192、512x512 的 PNG 放到 `public/icons/` 並命名為 `icon-192.png`、`icon-512.png`（可選）。

---

## 目錄

- `public/` — PWA 前端（登入、首頁、通話頁）
- `server.js` — 靜態檔 + 固定帳密登入 API + WebSocket 語音 Gateway
