# -*- coding: utf-8 -*-
"""
Telegram AI Agent Bot + Gradio 控制面板（Cyberpunk 主題）
========================================
功能整合自原 code01 / code02：
- Hermes AIAgent（OpenRouter / gpt-4o-mini）
- Brave Search + Firecrawl 網頁爬取（寫入 ~/.hermes/config.yaml）
- Telegram Bot（polling 模式）
- 對話紀錄自動儲存為 JSON 與 CSV（utf-8-sig，Excel 開啟中文不會亂碼）
- Gradio 網頁介面，讓使用者可直接在瀏覽器輸入 API Key 並啟動/停止 Bot
- UI 採賽博龐克（Cyberpunk）風格：霓虹青 / 洋紅配色、掃描線與終端機字體

安裝需求：
    pip install -q git+https://github.com/NousResearch/hermes-agent.git
    pip install -q python-telegram-bot gradio pyyaml

注意：本版本不使用 nest_asyncio。Bot 是在獨立的背景執行緒中啟動，
該執行緒本身沒有既存的事件迴圈，因此不需要（也不能）用 nest_asyncio
修補 asyncio —— 否則會與 Gradio 內部使用的 uvicorn 產生衝突
（Python 3.12 的 asyncio.run 多了 loop_factory 參數，nest_asyncio
覆寫後的版本不支援，會導致 uvicorn 執行緒全部拋出
TypeError: got an unexpected keyword argument 'loop_factory'）。

執行：
    python telegram_bot_gradio.py
"""

import os
import csv
import json
import yaml
import asyncio
import threading
import datetime
from pathlib import Path

import gradio as gr

from run_agent import AIAgent
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters


# =========================================================
# 偵測 asyncio 是否已被 nest_asyncio 永久修補過
# =========================================================
_run_qualname = getattr(asyncio.run, "__qualname__", "")
_run_module = getattr(asyncio.run, "__module__", "")
if "nest_asyncio" in _run_module or "_patch_asyncio" in _run_qualname:
    raise RuntimeError(
        "偵測到 asyncio.run 已被 nest_asyncio 永久修補過。\n"
        "這通常是因為同一個 Jupyter / Colab Kernel 曾經執行過含有\n"
        "nest_asyncio.apply() 的（舊版）程式碼，而這個修補無法單靠重新\n"
        "執行程式碼還原，必須「重新啟動」Kernel / Runtime 才會清除：\n"
        "  - Google Colab：上方選單「執行階段」→「重新啟動工作階段」\n"
        "  - Jupyter Notebook / Lab：上方選單「Kernel」→「Restart Kernel」\n"
        "  - 一般終端機執行 python 檔案不會有此問題。\n"
        "重新啟動後，本程式本身已不再使用 nest_asyncio，直接重新執行即可。"
    )
del _run_qualname, _run_module


# =========================================================
# 全域狀態
# =========================================================
class BotState:
    def __init__(self):
        self.agent = None
        self.app = None
        self.thread = None
        self.loop = None
        self.running = False
        self.log_lines = []

    def log(self, msg: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-200:]
        print(line)

    def get_log_text(self) -> str:
        return "\n".join(self.log_lines) if self.log_lines else ">_ 等待指令..."


STATE = BotState()

HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)


# =========================================================
# 對話紀錄：JSON + CSV（utf-8-sig）
# =========================================================
def get_json_path(chat_id: int) -> Path:
    return HISTORY_DIR / f"{chat_id}.json"


def get_csv_path(chat_id: int) -> Path:
    return HISTORY_DIR / f"{chat_id}.csv"


def load_history(chat_id: int) -> list:
    path = get_json_path(chat_id)
    if path.exists():
        with open(path, "r", encoding="utf-8-sig") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_history(chat_id: int, history: list):
    with open(get_json_path(chat_id), "w", encoding="utf-8-sig") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(get_csv_path(chat_id), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "role", "text"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def append_history(chat_id: int, role: str, text: str):
    history = load_history(chat_id)
    history.append(
        {
            "role": role,
            "text": text,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )
    save_history(chat_id, history)


# =========================================================
# Hermes 設定檔（網路搜尋 + 網頁爬取）
# =========================================================
def write_hermes_config():
    hermes_dir = os.path.expanduser("~/.hermes")
    os.makedirs(hermes_dir, exist_ok=True)
    config = {
        "web": {
            "search_backend": "brave-free",
            "extract_backend": "firecrawl",
        }
    }
    with open(os.path.join(hermes_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)


# =========================================================
# Telegram 訊息處理
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_text = update.message.text

    append_history(chat_id, "user", user_text)
    STATE.log(f"收到訊息 (chat_id={chat_id}): {user_text[:50]}")

    try:
        reply = await asyncio.to_thread(STATE.agent.chat, user_text)
    except Exception as e:
        reply = f"⚠️ Agent 發生錯誤：{e}"
        STATE.log(f"Agent 錯誤: {e}")

    append_history(chat_id, "assistant", reply)
    await update.message.reply_text(reply[:4096])


# =========================================================
# 啟動 / 停止 Bot（背景執行緒）
# =========================================================
def _run_bot_blocking():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    STATE.loop = loop
    try:
        STATE.app.run_polling(stop_signals=None)
    except Exception as e:
        STATE.log(f"Bot 執行時發生錯誤：{e}")
    finally:
        STATE.running = False
        STATE.loop = None
        STATE.log("Bot 已停止。")


def start_bot(openrouter_key, brave_key, firecrawl_key, telegram_token, model_name):
    if STATE.running:
        return "⚠️ Bot 已經在執行中，請先停止再重新啟動。", STATE.get_log_text()

    if not telegram_token:
        return "❌ 請輸入 Telegram Bot Token。", STATE.get_log_text()
    if not openrouter_key:
        return "❌ 請輸入 OpenRouter API Key。", STATE.get_log_text()

    os.environ["OPENROUTER_API_KEY"] = openrouter_key
    if brave_key:
        os.environ["BRAVE_SEARCH_API_KEY"] = brave_key
    if firecrawl_key:
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_key

    write_hermes_config()

    try:
        STATE.agent = AIAgent(
            model=model_name or "openai/gpt-4o-mini",
            quiet_mode=True,
        )
    except Exception as e:
        return f"❌ 建立 Agent 失敗：{e}", STATE.get_log_text()

    try:
        STATE.app = ApplicationBuilder().token(telegram_token).build()
        STATE.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )
    except Exception as e:
        return f"❌ 建立 Telegram Bot 失敗：{e}", STATE.get_log_text()

    STATE.running = True
    STATE.thread = threading.Thread(target=_run_bot_blocking, daemon=True)
    STATE.thread.start()

    STATE.log("Bot 已啟動，開始輪詢 Telegram 訊息...")
    return "✅ ONLINE — Bot 已啟動，可以在 Telegram 開始對話。", STATE.get_log_text()


def stop_bot():
    if not STATE.running or STATE.app is None or STATE.loop is None:
        return "⚠️ Bot 目前未在執行。", STATE.get_log_text()

    try:
        STATE.loop.call_soon_threadsafe(STATE.app.stop_running)
        STATE.log("已送出停止指令，Bot 即將關閉...")
    except Exception as e:
        STATE.log(f"停止 Bot 時發生錯誤：{e}")
        return f"❌ 停止失敗：{e}", STATE.get_log_text()

    return "🛑 OFFLINE — 已送出停止指令，請稍候幾秒。", STATE.get_log_text()


def refresh_log():
    return STATE.get_log_text()


# =========================================================
# 匯出對話紀錄（依 chat_id）
# =========================================================
def export_history(chat_id_str: str):
    if not chat_id_str.strip():
        return None, None, "❌ 請輸入 chat_id。"

    try:
        chat_id = int(chat_id_str.strip())
    except ValueError:
        return None, None, "❌ chat_id 必須是數字。"

    json_path = get_json_path(chat_id)
    csv_path = get_csv_path(chat_id)

    if not json_path.exists():
        return None, None, f"⚠️ 找不到 chat_id={chat_id} 的對話紀錄。"

    return str(json_path), str(csv_path), f"✅ 已找到 chat_id={chat_id} 的紀錄。"


def list_available_chats():
    files = sorted(HISTORY_DIR.glob("*.json"))
    if not files:
        return "目前尚無任何對話紀錄。"
    ids = [f.stem for f in files]
    return "可用的 chat_id：\n" + "\n".join(ids)


# =========================================================
# Cyberpunk 主題：色彩 / 字體 / 元件樣式
# =========================================================
CYBERPUNK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

:root {
    --cp-void:      #0a0e17;
    --cp-panel:     #0f1622;
    --cp-line:      #1c2534;
    --cp-cyan:      #00fff2;
    --cp-magenta:   #ff2d95;
    --cp-yellow:    #f4ff00;
    --cp-text:      #d6f7f5;
    --cp-text-dim:  #5b7480;
}

.gradio-container {
    background: radial-gradient(circle at 15% 0%, #101c2b 0%, var(--cp-void) 45%) !important;
    background-attachment: fixed !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--cp-text) !important;
    position: relative;
}

/* 掃描線疊加效果 */
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9999;
    background: repeating-linear-gradient(
        to bottom,
        rgba(0, 255, 242, 0.025) 0px,
        rgba(0, 255, 242, 0.025) 1px,
        transparent 2px,
        transparent 4px
    );
    mix-blend-mode: screen;
}

/* 標題橫幅 */
#cp-header {
    border: 1px solid var(--cp-cyan);
    background: linear-gradient(180deg, rgba(0,255,242,0.08), rgba(10,14,23,0.4));
    box-shadow: 0 0 18px rgba(0,255,242,0.25), inset 0 0 22px rgba(0,255,242,0.05);
    border-radius: 2px;
    padding: 22px 26px 18px 26px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

#cp-header::after {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--cp-magenta);
    box-shadow: 0 0 12px var(--cp-magenta);
    animation: cp-blink 2.4s ease-in-out infinite;
}

@keyframes cp-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

#cp-title h1, #cp-title h2 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    letter-spacing: 2px;
    color: var(--cp-cyan) !important;
    text-shadow: 0 0 8px rgba(0,255,242,0.7), 0 0 22px rgba(0,255,242,0.35);
    margin: 0 !important;
}

#cp-subtitle p {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--cp-text-dim) !important;
    letter-spacing: 1px;
    font-size: 13px !important;
}

#cp-subtitle p::before {
    content: "// ";
    color: var(--cp-magenta);
}

/* 區塊卡片 */
.gr-block, .gr-box, .block, .gr-panel, .form {
    background: var(--cp-panel) !important;
    border: 1px solid var(--cp-line) !important;
    border-radius: 2px !important;
}

label span, .gr-form label, label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--cp-cyan) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 12px !important;
}

/* 輸入框 */
input, textarea, select {
    background: #060a10 !important;
    border: 1px solid var(--cp-line) !important;
    color: var(--cp-text) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--cp-cyan) !important;
    box-shadow: 0 0 10px rgba(0,255,242,0.4) !important;
}

/* 按鈕 */
button {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1px;
    border-radius: 2px !important;
    transition: all 0.15s ease-in-out !important;
}

.gr-button, button.primary, button.secondary {
    background: transparent !important;
    border: 1px solid var(--cp-cyan) !important;
    color: var(--cp-cyan) !important;
}

.gr-button:hover, button.primary:hover, button.secondary:hover {
    background: rgba(0,255,242,0.12) !important;
    box-shadow: 0 0 14px rgba(0,255,242,0.6) !important;
    color: #ffffff !important;
}

button.primary {
    border-color: var(--cp-magenta) !important;
    color: var(--cp-magenta) !important;
}

button.primary:hover {
    background: rgba(255,45,149,0.12) !important;
    box-shadow: 0 0 14px rgba(255,45,149,0.6) !important;
}

/* 終端機風格 Log 視窗 */
#cp-log textarea {
    background: #04070b !important;
    color: var(--cp-cyan) !important;
    border: 1px solid rgba(0,255,242,0.3) !important;
    box-shadow: inset 0 0 16px rgba(0,255,242,0.08) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Accordion 標題 */
.gr-accordion .label-wrap, .accordion .label-wrap {
    color: var(--cp-yellow) !important;
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1px;
}

/* 分隔標題文字 (Markdown h3) */
h3 {
    color: var(--cp-magenta) !important;
    font-family: 'Orbitron', sans-serif !important;
    text-shadow: 0 0 6px rgba(255,45,149,0.5);
    border-bottom: 1px dashed var(--cp-line);
    padding-bottom: 6px;
}
"""

CYBERPUNK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Share Tech Mono"),
    font_mono=gr.themes.GoogleFont("Share Tech Mono"),
).set(
    body_background_fill="#0a0e17",
    body_text_color="#d6f7f5",
    block_background_fill="#0f1622",
    block_border_color="#1c2534",
    border_color_primary="#1c2534",
)


# =========================================================
# Gradio 介面
# =========================================================
with gr.Blocks(
    title="Hermes agent with Telegram",
    theme=CYBERPUNK_THEME,
    css=CYBERPUNK_CSS,
) as demo:

    with gr.Column(elem_id="cp-header"):
        with gr.Column(elem_id="cp-title"):
            gr.Markdown("# ⌁ HERMES AGENT WITH TELEGRAM")
        with gr.Column(elem_id="cp-subtitle"):
            gr.Markdown("NEURAL LINK CONTROL PANEL — 輸入金鑰、啟動 Bot、監控對話紀錄")

    with gr.Row():
        with gr.Column():
            openrouter_key = gr.Textbox(label="OpenRouter API Key", type="password")
            brave_key = gr.Textbox(label="Brave Search API Key（選填，網路搜尋用）", type="password")
            firecrawl_key = gr.Textbox(label="Firecrawl API Key（選填，網頁全文爬取用）", type="password")
            telegram_token = gr.Textbox(label="Telegram Bot Token", type="password")
            model_name = gr.Textbox(label="模型名稱", value="openai/gpt-4o-mini")

            with gr.Row():
                start_btn = gr.Button("▶ 啟動 BOT", variant="primary")
                stop_btn = gr.Button("■ 停止 BOT")

            status_box = gr.Textbox(label="狀態", interactive=False)

        with gr.Column():
            log_box = gr.Textbox(
                label="執行紀錄 // SYSTEM LOG",
                lines=18,
                interactive=False,
                elem_id="cp-log",
            )
            refresh_btn = gr.Button("⟳ 重新整理 LOG")

    gr.Markdown("### ▚ 對話紀錄匯出 // DATA ARCHIVE")
    with gr.Row():
        with gr.Column():
            list_btn = gr.Button("列出所有 CHAT_ID")
            chat_list_box = gr.Textbox(label="可用 chat_id 清單", lines=6, interactive=False)
        with gr.Column():
            chat_id_input = gr.Textbox(label="輸入要匯出的 chat_id")
            export_btn = gr.Button("匯出 JSON / CSV")
            export_status = gr.Textbox(label="匯出狀態", interactive=False)
            json_file = gr.File(label="下載 JSON")
            csv_file = gr.File(label="下載 CSV")

    start_btn.click(
        fn=start_bot,
        inputs=[openrouter_key, brave_key, firecrawl_key, telegram_token, model_name],
        outputs=[status_box, log_box],
    )
    stop_btn.click(fn=stop_bot, outputs=[status_box, log_box])
    refresh_btn.click(fn=refresh_log, outputs=[log_box])
    list_btn.click(fn=list_available_chats, outputs=[chat_list_box])
    export_btn.click(
        fn=export_history,
        inputs=[chat_id_input],
        outputs=[json_file, csv_file, export_status],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
