# -*- coding: utf-8 -*-
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
import zipfile
import os
import re
import gc

# =========================
# セキュリティ/堅牢化のポイント
# - 入力検証：拡張子/PDF暗号化/ページ数/ピクセル数
# - サニタイズ：Zip内のファイル名を安全化
# - リソース解放：doc.close(), del, gc.collect()
# - 例外：詳細を出さず要点のみ表示（情報漏えい抑止）
# - 0倍レンダリング防止（最小スケールを適用）
# - Zipは1本化して /common /protanopia 等に振り分け
# =========================

# ---- 安全側の上限/制限（必要に応じて調整）----
MAX_FILES = 200               # 一度に処理するPDF数の上限
MAX_PAGES_PER_FILE = 200      # 1ファイルの最大ページ数
MAX_PIXELS_PER_PAGE = 1000_000_000  # 1ページの最大ピクセル数（約12MP）
ALLOWED_EXT = {".pdf"}       # 受け付ける拡張子

# ---- 表示スケールの下限/上限 ----
MIN_SCALE = 0.1              # 0指定で0ピクセルになる事故を防止
MAX_SCALE = 1.0              # 必要に応じて 1.5 などへ

# ============ 色覚シミュレーション ============
def gamma_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def linear_to_gamma(rgb):
    gamma = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (np.power(rgb, 1/2.4)) - 0.055)
    return np.clip(gamma * 255.0, 0, 255).astype(np.uint8)

def sRGB_to_XYZ(rgb):
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    return np.dot(rgb, M.T)

def XYZ_to_LMS(xyz):
    M = np.array([[0.4002, 0.7075, -0.0808],
                  [-0.2263, 1.1653, 0.0457],
                  [0.0,    0.0,     0.9182]], dtype=np.float32)
    return np.dot(xyz, M.T)

def LMS_to_XYZ(lms):
    M = np.array([[ 1.8599, -1.1294, 0.2199],
                  [ 0.3612,  0.6388, 0.0   ],
                  [ 0.0,     0.0,    1.0891]], dtype=np.float32)
    return np.dot(lms, M.T)

def XYZ_to_sRGB(xyz):
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]], dtype=np.float32)
    return np.dot(xyz, M.T)

def simulate_deficiency(arr_rgb, deficiency_type):
    linear_rgb = gamma_to_linear(arr_rgb)
    xyz = sRGB_to_XYZ(linear_rgb)
    lms = XYZ_to_LMS(xyz)

    if deficiency_type == 'protanopia':
        M = np.array([[0, 1.208, -0.208],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float32)
    elif deficiency_type == 'deuteranopia':
        M = np.array([[1, 0, 0],
                      [0.8278, 0, 0.1722],
                      [0, 0, 1]], dtype=np.float32)
    elif deficiency_type == 'tritanopia':
        M = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [-0.5254, 1.5254, 0]], dtype=np.float32)
    else:
        return arr_rgb  # unknown -> no change

    simulated_lms = np.dot(lms, M.T)
    xyz_sim = LMS_to_XYZ(simulated_lms)
    rgb_sim = XYZ_to_sRGB(xyz_sim)
    return linear_to_gamma(rgb_sim)

def convert_image(img: Image.Image, mode: str):
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    if mode == 'Acromat':
        gray = np.mean(arr, axis=2, keepdims=True)
        new_arr = np.repeat(gray, 3, axis=2).astype(np.uint8)
    else:
        new_arr = simulate_deficiency(arr, mode)
    return Image.fromarray(new_arr)

# ============ ユーティリティ ============
def sanitize_filename(name: str) -> str:
    """Zip内のファイル名を安全化（パストラバーサル対策・制御文字除去）"""
    base = os.path.basename(name or "file")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base

def clamp_scale(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 1.0
    return max(MIN_SCALE, min(MAX_SCALE, x))

# ============ Streamlit UI ============
st.set_page_config(page_title="Color Vision Simulation Viewer", layout="wide")
st.markdown(
    """
    <style>
      h1 { text-align:center; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color:#C00000; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1>Color Vision Simulation Viewer</h1>", unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
scale_in = st.sidebar.slider("画質スケール（0.1〜1.0）", MIN_SCALE, MAX_SCALE, 1.0)
scale = clamp_scale(scale_in)
st.sidebar.write(f"画質：{scale:.2f} 倍で処理")

process_btn = st.sidebar.button("Process PDF Files", disabled=not uploaded_files)

# タブ（閲覧プレビュー用）
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Common(正常)", "Protanopia(赤機能不全)", "Deuteranopia(緑機能不全)",
    "Tritanopia(青機能不全)", "Achromat(全色盲)"
])

def render_page_to_image(page, scale: float) -> Image.Image:
    """PyMuPDFでページを画像化（解像度制御）"""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # 透過なし（容量軽減）
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

if process_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
        st.stop()

    if len(uploaded_files) > MAX_FILES:
        st.warning(f"❗ 一度に処理できるPDFは最大 {MAX_FILES} 件です。")
        uploaded_files = uploaded_files[:MAX_FILES]

    # 出力Zip（1本）
    out_zip_buf = io.BytesIO()
    with zipfile.ZipFile(out_zip_buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as outzip:
        # 進捗管理
        total_steps = 0
        # 事前に総ページ数（制限後）をカウント
        for uf in uploaded_files:
            try:
                doc = fitz.open(stream=uf.read(), filetype="pdf")
                uf.seek(0)
                if doc.needs_password():
                    doc.close()
                    continue
                total_steps += min(doc.page_count, MAX_PAGES_PER_FILE) * 5  # 5種のモード
                doc.close()
            except Exception:
                # 読み込み失敗 → スキップ
                continue
        if total_steps == 0:
            st.warning("処理可能なPDFがありません（暗号化/破損/0ページ等）。")
            st.stop()

        step = 0
        progress = st.progress(0.0, text="Processing... Please wait")
        st.markdown("<br><br>", unsafe_allow_html=True)

        for uploaded_file in uploaded_files:
            # 拡張子検証
            ext = os.path.splitext(uploaded_file.name or "")[1].lower()
            if ext not in ALLOWED_EXT:
                st.warning(f"❗ 未対応の拡張子です: {uploaded_file.name}")
                continue

            safe_pdf_name = sanitize_filename(uploaded_file.name)

            # PDFを開く（暗号化や破損の扱い）
            try:
                data = uploaded_file.read()
                uploaded_file.seek(0)
                doc = fitz.open(stream=data, filetype="pdf")
            except Exception:
                st.warning(f"❗ PDFを開けませんでした（破損の可能性）: {uploaded_file.name}")
                continue

            try:
                if doc.needs_password():
                    st.info(f"🔒 暗号化PDFのためスキップ: {uploaded_file.name}")
                    doc.close()
                    continue

                page_count = min(doc.page_count, MAX_PAGES_PER_FILE)
                per_file_note = ""
                if doc.page_count > MAX_PAGES_PER_FILE:
                    per_file_note = f"（先頭 {MAX_PAGES_PER_FILE} ページのみ処理）"

                with tab1:
                    st.subheader(f"{safe_pdf_name} {per_file_note}")

                # 各ページ処理
                for page_num in range(page_count):
                    try:
                        page = doc.load_page(page_num)
                        # 粗大なDoSを防ぐため、ピクセル総数をチェック
                        # （scale適用後の概算、PDFのメディアボックスから計算）
                        rect = page.rect
                        est_w = max(1, int(rect.width * scale))
                        est_h = max(1, int(rect.height * scale))
                        if est_w * est_h > MAX_PIXELS_PER_PAGE:
                            # 大きすぎるページはスケールを落としてレンダリング
                            adj_scale = scale * (MAX_PIXELS_PER_PAGE / (est_w * est_h)) ** 0.5
                            adj_scale = max(MIN_SCALE, min(adj_scale, scale))
                        else:
                            adj_scale = scale

                        img = render_page_to_image(page, adj_scale)

                        # プレビュー表示（Normal）
                        with tab1:
                            st.subheader(f"{safe_pdf_name} - Page {page_num + 1} (Normal)")
                            st.image(img, caption="Original", use_container_width=True)

                        # Zip書き込み（各モード）
                        # 共通：/common 等のディレクトリ配下に格納
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        outzip.writestr(f"common/{safe_pdf_name}_p{page_num+1}.png", buf.getvalue())
                        step += 1; progress.progress(step / total_steps, text="Processing... Please wait")

                        # Protanopia
                        with tab2:
                            st.subheader(f"{safe_pdf_name} - Page {page_num + 1} (Protanopia)")
                            converted = convert_image(img, 'protanopia')
                            st.image(converted, caption="Protanopia", use_container_width=True)
                        buf = io.BytesIO(); converted.save(buf, format="PNG")
                        outzip.writestr(f"protanopia/{safe_pdf_name}_p{page_num+1}.png", buf.getvalue())
                        step += 1; progress.progress(step / total_steps, text="Processing... Please wait")

                        # Deuteranopia
                        with tab3:
                            st.subheader(f"{safe_pdf_name} - Page {page_num + 1} (Deuteranopia)")
                            converted = convert_image(img, 'deuteranopia')
                            st.image(converted, caption="Deuteranopia", use_container_width=True)
                        buf = io.BytesIO(); converted.save(buf, format="PNG")
                        outzip.writestr(f"deuteranopia/{safe_pdf_name}_p{page_num+1}.png", buf.getvalue())
                        step += 1; progress.progress(step / total_steps, text="Processing... Please wait")

                        # Tritanopia
                        with tab4:
                            st.subheader(f"{safe_pdf_name} - Page {page_num + 1} (Tritanopia)")
                            converted = convert_image(img, 'tritanopia')
                            st.image(converted, caption="Tritanopia", use_container_width=True)
                        buf = io.BytesIO(); converted.save(buf, format="PNG")
                        outzip.writestr(f"tritanopia/{safe_pdf_name}_p{page_num+1}.png", buf.getvalue())
                        step += 1; progress.progress(step / total_steps, text="Processing... Please wait")

                        # Achromat
                        with tab5:
                            st.subheader(f"{safe_pdf_name} - Page {page_num + 1} (Achromatopsia)")
                            converted = convert_image(img, 'Acromat')
                            st.image(converted, caption="Achromatopsia", use_container_width=True)
                        buf = io.BytesIO(); converted.save(buf, format="PNG")
                        outzip.writestr(f"achromat/{safe_pdf_name}_p{page_num+1}.png", buf.getvalue())
                        step += 1; progress.progress(step / total_steps, text="Processing... Please wait")

                        # メモリ解放
                        del img, converted, buf
                        gc.collect()

                    except Exception:
                        st.warning(f"❗ ページ {page_num+1} の処理に失敗しました（{safe_pdf_name}）。")
                        continue

            except Exception:
                st.warning(f"❗ 処理中に問題が発生しました: {uploaded_file.name}")
            finally:
                try:
                    doc.close()
                except Exception:
                    pass

        progress.empty()
        st.balloons()
        st.toast("All images processed successfully!")

    # Zipダウンロード
    out_zip_buf.seek(0)
    st.download_button(
        "Download all processed images as ZIP",
        data=out_zip_buf.getvalue(),
        file_name="processed_images.zip",
        mime="application/zip"
    )
    del out_zip_buf
    gc.collect()
else:
    # 静的な説明セクション（ユーザー入力を混ぜない）
    st.markdown(
        """<style>
    .display_1 {
        top: 0;
        left: 0;
    }

    .table {
        color: #000000;
        background-color: #FBF4E5;
        text-align: center;
        border: 2px #360000 solid;
        width: 100%;
        height: 200px;
        margin: auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-collapse: collapse;
    }

    td {
        border: 1px #360000 solid;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.25vw;
        vertical-align: middle;
        line-height: 2.0;
    }

    html,
    body {
        background-color: #f0f0f0;
    }

    .header_row {
        border-bottom: 2px double #360000;
    }

    h2 {
        margin-bottom: 5px;
    }

    .hosoku {
        matgin: 1px;
    }
</style>
<div class="table_wrap">
    <h2>型別の割合</h2>
    <div id="display-element" class="display_1">
        <table border="1" class="table">
            <tbody>
                <tr>
                    <td rowspan="2">型</td>
                    <td colspan="3">錐体細胞</td>
                    <td rowspan="2">割合<br>(男性)</td>
                </tr>
                <tr class="header_row">
                    <td class="red">L</td>
                    <td class="magenta">M</td>
                    <td class="cyan">S</td>
                </tr>
                <tr>
                    <td>C型</td>
                    <td>○</td>
                    <td>○</td>
                    <td>○</td>
                    <td>約９５％</td>
                </tr>
                <tr>
                    <td>P型</td>
                    <td>-</td>
                    <td>○</td>
                    <td>○</td>
                    <td>約１.５％</td>
                </tr>
                <tr>
                    <td>D型</td>
                    <td>○</td>
                    <td>-</td>
                    <td>○</td>
                    <td>約３.５％</td>
                </tr>
                <tr>
                    <td>T型</td>
                    <td>○</td>
                    <td>○</td>
                    <td>-</td>
                    <td>約０.００１％</td>
                </tr>
                <tr>
                    <td>A型</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>約０.００１％</td>
                </tr>
            </tbody>
        </table>
        <div class="pyramidal_cells">
            <p class="hosoku">※Ｌ（赤）錐体：主に黄緑～赤の光を感じる</p>
            <p class="hosoku">※Ｍ（緑）錐体：主に緑～橙の光を感じる</p>
            <p class="hosoku">※Ｓ（青）錐体：主に紫～青の光を感じる</p>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

