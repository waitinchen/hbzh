# 開發規範

## 嚴格禁止
- 禁止翻譯任何 console.log 內容
- 禁止修改 package.json 以外的設定檔
- 禁止 commit 二進位檔案（圖片、音頻）
- 禁止修改 railway.toml
- 禁止自動 git push，每次 push 前必須確認

## 部署方式
- 統一用 `railway up` 部署，不走 GitHub
- 每次改完只改指定的檔案

## 語言規範
- 所有 console.log 保持英文
- 註解可以中文

## Railway LOG 存取
- 查 log：`railway logs --tail 200`
- 即時：`railway logs --follow`
