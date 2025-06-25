import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
import zipfile

# Color vision deficiency simulation functions
def gamma_to_linear(rgb):
    rgb = np.array(rgb) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

def linear_to_gamma(rgb):
    gamma = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1 / 2.4)) - 0.055)
    return np.clip(gamma * 255, 0, 255).astype(np.uint8)

def sRGB_to_XYZ(rgb):
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    return np.dot(M, rgb)

def XYZ_to_LMS(xyz):
    M = np.array([[0.4002, 0.7075, -0.0808],
                  [-0.2263, 1.1653, 0.0457],
                  [0.0, 0.0, 0.9182]])
    return np.dot(M, xyz)

def LMS_to_XYZ(lms):
    M = np.array([[1.8599, -1.1294, 0.2199],
                  [0.3612, 0.6388, 0.0],
                  [0.0, 0.0, 1.0891]])
    return np.dot(M, lms)

def XYZ_to_sRGB(xyz):
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    return np.dot(M, xyz)

def simulate_deficiency(rgb, deficiency_type):
    linear_rgb = gamma_to_linear(rgb)
    xyz = sRGB_to_XYZ(linear_rgb)
    lms = XYZ_to_LMS(xyz)

    if deficiency_type == 'protanopia':
        M = np.array([[0, 1.208, -0.208],
                      [0, 1, 0],
                      [0, 0, 1]])
    elif deficiency_type == 'deuteranopia':
        M = np.array([[1, 0, 0],
                      [0.8278, 0, 0.1722],
                      [0, 0, 1]])
    elif deficiency_type == 'tritanopia':
        M = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [-0.5254, 1.5254, 0]])
    else:
        return rgb

    simulated_lms = np.dot(M, lms)
    xyz_sim = LMS_to_XYZ(simulated_lms)
    rgb_sim = XYZ_to_sRGB(xyz_sim)
    return linear_to_gamma(rgb_sim)

def convert_image(img, mode, multiple):
    img = img.resize(((img.width * multiple) , (img.height * multiple)), Image.LANCZOS)

    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    new_arr = np.zeros_like(arr)

    for y in range(h):
        for x in range(w):
            pixel = arr[y, x]
            if mode == 'Acromat':
                gray = int(np.mean(pixel))
                new_arr[y, x] = [gray, gray, gray]
            else:
                new_arr[y, x] = simulate_deficiency(pixel, mode)

    return Image.fromarray(new_arr)

def main():
    # Streamlit UI
    st.html("""
            <style>
                h1 {
                    text-align: center;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 1.8vw;
                    color: #C00000;
                }
            </style>
            <h1>Color Vision Simulation Viewer</h1>
            """)

    uploaded_files = st.sidebar.file_uploader("Upload a PDF file", accept_multiple_files=True, type="pdf")
    multiple = st.sidebar.slider("画質を調節してください。", 0.0, 1.0, 0.1)
    st.sidebar.write(f"画質：{multiple}倍で処理")
    
    if st.sidebar.button("Process PDF Files", disabled=not uploaded_files):
        if uploaded_files:
            progress_text = "Operation in progress. Please wait..."

            zip_buffer_common = io.BytesIO()
            zip_buffer_protanopia = io.BytesIO()
            zip_buffer_deuteranopia = io.BytesIO()
            zip_buffer_tritanopia = io.BytesIO()
            zip_buffer_acromat = io.BytesIO()

            zip_file_common = zipfile.ZipFile(zip_buffer_common, "w", zipfile.ZIP_DEFLATED)
            zip_file_protanopia = zipfile.ZipFile(zip_buffer_protanopia, "w", zipfile.ZIP_DEFLATED)
            zip_file_deuteranopia = zipfile.ZipFile(zip_buffer_deuteranopia, "w", zipfile.ZIP_DEFLATED)
            zip_file_tritanopia = zipfile.ZipFile(zip_buffer_tritanopia, "w", zipfile.ZIP_DEFLATED)
            zip_file_acromat = zipfile.ZipFile(zip_buffer_acromat, "w", zipfile.ZIP_DEFLATED)

            num = 0
            my_bar = st.progress(0, text=progress_text)
            st.html("<br><br><br>")  # Add some space before the tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Common(正常)", "Protanopia(赤機能不全)", "Deuteranopia(緑機能不全)", "Tritanopia(青機能不全)", "Achromat(全色盲)"])
            par_num = 100 / len(uploaded_files)

            for uploaded_file in uploaded_files:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                page_count = doc.page_count
                filename = uploaded_file.name
                par_page = (par_num / (page_count * 5)) / 100

                for page_num in range(page_count):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    with tab1:
                        st.subheader(f"{filename} - Page {page_num + 1} (Normal Vision)")
                        st.image(img, caption="Original", use_container_width=True)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    zip_file_common.writestr(f"{filename}_{page_num + 1}_common.png", img_bytes.getvalue())
                    num += par_page
                    my_bar.progress(num, text=progress_text)

                    with tab2:
                        st.subheader(f"{filename} - Page {page_num + 1} (Protanopia)")
                        converted = convert_image(img, 'protanopia', multiple)
                        st.image(converted, caption="Protanopia", use_container_width=True)
                    img_bytes = io.BytesIO()
                    converted.save(img_bytes, format="PNG")
                    zip_file_protanopia.writestr(f"{filename}_{page_num + 1}_protanopia.png", img_bytes.getvalue())
                    num += par_page
                    my_bar.progress(num, text=progress_text)

                    with tab3:
                        st.subheader(f"{filename} - Page {page_num + 1} (Deuteranopia)")
                        converted = convert_image(img, 'deuteranopia', multiple)
                        st.image(converted, caption="Deuteranopia", use_container_width=True)
                    img_bytes = io.BytesIO()
                    converted.save(img_bytes, format="PNG")
                    zip_file_deuteranopia.writestr(f"{filename}_{page_num + 1}_deuteranopia.png", img_bytes.getvalue()) 
                    num += par_page
                    my_bar.progress(num, text=progress_text)

                    with tab4:
                        st.subheader(f"{filename} - Page {page_num + 1} (Tritanopia)")
                        converted = convert_image(img, 'tritanopia', multiple)    
                        st.image(converted, caption="Tritanopia", use_container_width=True)
                    img_bytes = io.BytesIO()
                    converted.save(img_bytes, format="PNG")
                    zip_file_tritanopia.writestr(f"{filename}_{page_num + 1}_tritanopia.png", img_bytes.getvalue())
                    num += par_page
                    my_bar.progress(num, text=progress_text)

                    with tab5:
                        st.subheader(f"{filename} - Page {page_num + 1} (Achromatopsia)")
                        converted = convert_image(img, 'Acromat', multiple)
                        st.image(converted, caption="Achromatopsia", use_container_width=True)
                    img_bytes = io.BytesIO()
                    converted.save(img_bytes, format="PNG")
                    zip_file_acromat.writestr(f"{filename}_{page_num + 1}_acromat.png", img_bytes.getvalue())
                    num += par_page
                    my_bar.progress(num, text=progress_text)
                
            zip_file_common.close()
            zip_file_protanopia.close()
            zip_file_deuteranopia.close()
            zip_file_tritanopia.close()
            zip_file_acromat.close()
            my_bar.empty()
            st.balloons()
            st.toast("All images processed successfully!")
            
            zip_file_common_bytes = zip_buffer_common.getvalue()
            zip_file_protanopia_bytes = zip_buffer_protanopia.getvalue()
            zip_file_deuteranopia_bytes = zip_buffer_deuteranopia.getvalue()
            zip_file_tritanopia_bytes = zip_buffer_tritanopia.getvalue()
            zip_file_acromat_bytes = zip_buffer_acromat.getvalue()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("images_common_view.zip", zip_file_common_bytes)
                zip_file.writestr("images_protanopia_view.zip", zip_file_protanopia_bytes)
                zip_file.writestr("images_deuteranopia_view.zip", zip_file_deuteranopia_bytes)
                zip_file.writestr("images_tritanopia_view.zip", zip_file_tritanopia_bytes)
                zip_file.writestr("images_acromat_view.zip", zip_file_acromat_bytes)

            zip_buffer.seek(0)
            st.download_button("Download all processed images as ZIP", data=zip_buffer.getvalue(), file_name="processed_images.zip", mime="application/zip")
        else:
            st.error("Please upload at least one PDF file to process.")
    else:
        st.html("""
            <style>
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

                td{
                    border: 1px #360000 solid;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 1.25vw;
                    vertical-align: middle;
                    line-height: 2.0;
                }

                html, body{
                    background-color: #f0f0f0;
                }

                .header_row {
                    border-bottom: 2px double #360000;
                }

                h2{
                    margin-bottom: 5px;
                }

                .hosoku{
                    matgin: 1px;
                }

            </style>
            <div class= "table_wrap">
                <h2>型別の割合</h2>
                <div id= "display-element" class="display_1">
                    <table border= "1" class= "table">
                        <tbody>
                            <tr>
                                <td rowspan= "2">型</td>
                                <td colspan= "3">錐体細胞</td>
                                <td rowspan= "2">割合<br>(男性)</td>
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

                """)
        

if __name__ == "__main__":
    main()
