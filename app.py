import os
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="給湯器サポートBot", page_icon="♨️")
st.title("♨️ 給湯器サポートBot")
st.caption("Gemini API で動くシンプルなサポートBot / 会社スタッフとして回答")

# --- API キーの取得（Streamlit Secrets または 環境変数 → 未設定なら入力欄） ---
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

# --- Gemini 初期化 ---
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

SYSTEM_PROMPT = (
    "あなたは給湯器メーカーのカスタマーサポート担当スタッフです。方針:\n"
    "- まず『設定の見直し』と『再起動』を基本の対処として案内する。\n"
    "- ユーザーが『もうやった/効果なし』と言ったら、簡単な確認チェックリスト → 追加の軽い対処（リモコンの電池交換/取扱説明書の参照/給湯器のリセット操作）→ サービス窓口の案内を行う。\n"
    "- 回答は3〜6行で簡潔に、手順は番号付き。専門用語は噛み砕く。危険な作業や分解は絶対に勧めない。\n"
    "- 不明点は1問だけ確認質問を返す（例: どの型番か、どんな症状か）。個人情報は収集しない。\n"
)

# --- 簡易チャットUI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) 

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("弊社の商品をご利用いただきありがとうございます。給湯器に関するお困りごとがあれば教えてください。AIスタッフがサポートいたします。")

user_msg = st.chat_input("メッセージを入力…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    history_text = "\n\n".join(
        [
            ("ユーザー: " + m["content"]) if m["role"] == "user" else ("スタッフ: " + m["content"]) 
            for m in st.session_state.messages[-8:]
        ]
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\nこれまでの会話:\n{history_text}\n\n"
        "最新のユーザーの質問に、上記方針に厳密に従って日本語で回答してください。"
    )

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "") if hasattr(resp, "text") else ""
        if not text:
            text = "（応答を取得できませんでした。時間をおいて再度お試しください）"
    except Exception as e:
        text = f"エラーが発生しました: {e}"

    st.session_state.messages.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.markdown(text)

st.divider()
st.caption("このBotは給湯器メーカーのサポートスタッフとして一般的な案内を行います。危険な操作や修理はせず、必要に応じて公式窓口をご利用ください。")