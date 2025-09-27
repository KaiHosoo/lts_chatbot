# app.py
import os
import json
import re
from io import BytesIO

import streamlit as st
import google.generativeai as genai
from PIL import Image

import joblib
import numpy as np
import pandas as pd

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="給湯器サポートBot", page_icon="♨️")
st.title("♨️ 給湯器サポートBot")
st.caption("Gemini API で動くシンプルなサポートBot / 会社スタッフとして回答")

# --- APIキー取得（Secrets→環境変数→入力欄）
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if not api_key:
    with st.expander("初回設定: GEMINI_API_KEY を入力（セッション内のみ保持）", expanded=True):
        api_key = st.text_input("GEMINI_API_KEY", type="password")
if not api_key:
    st.warning("Gemini APIキーを設定してください。")
    st.stop()

# --- Gemini 初期化（画像対応モデル）
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

SYSTEM_PROMPT = (
    "あなたは給湯器メーカーのカスタマーサポート担当スタッフです。方針:\n"
    "- まず『設定の見直し』と『再起動』を基本の対処として案内する。\n"
    "- ユーザーが『もうやった/効果なし』と言ったら、簡単な確認チェックリスト → 追加の軽い対処（リモコンの電池交換/取扱説明書の参照/給湯器のリセット操作）→ サービス窓口の案内を行う。\n"
    "- 回答は3〜6行で簡潔に、手順は番号付き。専門用語は噛み砕く。危険な作業や分解は絶対に勧めない。\n"
    "- 不明点は1問だけ確認質問を返す（例: どの型番か、どんな症状か）。個人情報は収集しない。\n"
)

# =========================
# 予測モデル（LightGBM + F1最適しきい値）
# =========================
MODEL_PATH = "artifacts/lgbm_f1_model.pkl"
META_PATH  = "artifacts/lgbm_f1_meta.json"

class FaultPredictor:
    """LightGBM + F1最適しきい値で Yes/No を返す予測器"""
    def __init__(self, model_path: str, meta_path: str):
        self.model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]           # 学習時の列順
        self.threshold = float(meta["threshold_f1"])         # F1最適しきい値

    def predict_one(self, payload: dict) -> dict:
        # dict -> DataFrame（列揃え & 数値化）
        X = pd.DataFrame([payload]).apply(pd.to_numeric, errors="coerce")
        X = X.reindex(columns=self.feature_names)
        prob = float(self.model.predict_proba(X)[:, 1][0])
        label_bin = 1 if prob > self.threshold else 0
        label = "Yes" if label_bin == 1 else "No"
        return {
            "label": label,            # "Yes" / "No"
            "prob": prob,              # 故障(Yes)確率
            "threshold": self.threshold,
        }

predictor = None
try:
    predictor = FaultPredictor(MODEL_PATH, META_PATH)
except Exception as e:
    st.warning(f"予測モデルの読み込みに失敗しました: {e}")

# =========================
# 画像→数値抽出（Gemini Vision OCR）
# =========================
FEATURE_KEYS = [
    "temparature1","humidity1","valtage1","usage1","rotation1",
    "temparature2","humidity2","valtage2","usage2","rotation2",
    "temparature3","humidity3","valtage3","usage3","rotation3"
]

EXTRACTION_PROMPT = (
    "次の画像に書かれたセンサー値を読み取り、以下の15キーのJSONのみを厳密に出力してください。\n"
    "数値以外の文字は無視し、単位があれば数値に正規化して小数で出力。\n"
    "キーのスペルは以下を厳守（temparature/valtage の綴りもそのまま）。\n"
    f"{FEATURE_KEYS}\n\n"
    "出力は **コードブロックなし・説明文なし** の純粋なJSON一発のみ。"
)

def _extract_json_from_text(text: str) -> dict:
    """Geminiの返答から最初のJSONオブジェクトを抽出してdict化"""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("JSONが見つかりませんでした。")
    return json.loads(m.group(0))

def ocr_sensors_with_gemini(model, pil_image: Image.Image) -> dict:
    resp = model.generate_content([EXTRACTION_PROMPT, pil_image])
    text = (resp.text or "") if hasattr(resp, "text") else ""
    data = _extract_json_from_text(text)

    # 必須キーを揃える（欠落は None、数値化に挑戦）
    fixed = {}
    for k in FEATURE_KEYS:
        v = data.get(k, None)
        try:
            fixed[k] = float(v) if v is not None else None
        except Exception:
            fixed[k] = None
    return fixed

# =========================
# チャット UI
# =========================
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
        [("ユーザー: " + m["content"]) if m["role"] == "user" else ("スタッフ: " + m["content"])
         for m in st.session_state.messages[-8:]]
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\nこれまでの会話:\n{history_text}\n\n"
        "最新のユーザーの質問に、上記方針に厳密に従って日本語で回答してください。"
    )

    try:
        resp = gemini_model.generate_content(prompt)
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

# =========================
# 画像アップロード → 数値抽出 → 故障判定
# =========================
st.divider()
st.subheader("画像からセンサー値を読み取って故障判定する")

col1, col2 = st.columns([1, 1])
with col1:
    img_file = st.file_uploader("センサー値が記載された画像（スクショ可）", type=["png", "jpg", "jpeg"])
with col2:
    st.caption("画像内の表やテキストから、以下15項目の値をOCRで抽出します：")
    st.code(", ".join(FEATURE_KEYS), language="text")

if img_file is not None:
    # 画像プレビュー
    image = Image.open(BytesIO(img_file.read()))
    st.image(image, caption="入力画像のプレビュー", use_container_width=True)

    # OCR & 構造化抽出
    with st.spinner("画像から数値を抽出しています…"):
        try:
            payload = ocr_sensors_with_gemini(gemini_model, image)
        except Exception as e:
            st.error(f"OCR抽出に失敗しました: {e}")
            payload = None

    if payload:
        st.markdown("**抽出結果（必要に応じて修正できます）**")
        # ユーザーが微修正できるように数値入力フォーム表示
        edited = {}
        cols = st.columns(3)
        for i, k in enumerate(FEATURE_KEYS):
            with cols[i % 3]:
                val = payload.get(k, None)
                edited[k] = st.number_input(
                    k,
                    value=(float(val) if isinstance(val, (int, float)) else 0.0),
                    format="%.6f",
                )

        # 予測
        if predictor is None:
            st.error("予測モデルが読み込まれていないため、判定できません。artifacts の配置を確認してください。")
        else:
            if st.button("この値で故障判定する"):
                try:
                    res = predictor.predict_one(edited)
                    label = "⚠️ 故障の可能性あり (Yes)" if res["label"] == "Yes" else "✅ 故障の可能性は低い (No)"
                    st.success(
                        f"{label}\n\n確率: **{res['prob']:.3f}** / しきい値(F1最適): **{res['threshold']:.3f}**"
                    )
                    with st.expander("送信値（モデル入力の最終形）を確認"):
                        st.json(edited)
                except Exception as e:
                    st.error(f"予測に失敗しました: {e}")