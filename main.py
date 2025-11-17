
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import plotly.express as px

# Sayfa Stil ve Config Ayarları
st.set_page_config(
    page_title="Kurumsal Sürdürülebilirlik Puan Sistemi",
    page_icon="leaf",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema Ayarları 
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background: #f8fff8;
    }
    .title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a5d1a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #2d6a2d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #4CAF50, #66BB6A);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(76, 175, 80, 0.3);
    }
    .status-good {
        background-color: #e8f5e8;
        color: #1b5e20;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-bad {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .table-header {
        background: #1a5d1a;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 0.5rem 0;
        text-align: center;
    }
    .table-subheader {
        font-size: 0.9rem;
        margin: 0;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Hesaplama Fonksiyonlarının Tanımlanması
def ahp_weighting_detailed(df: pd.DataFrame, RI: float):
    df = df.copy()
    df.columns = df.index
    if df.shape[0] != df.shape[1]:
        raise ValueError("Karşılaştırma matrisi kare değil.")
    A = df.to_numpy()
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(eigvals.real)
    lambda_max = eigvals.real[max_idx]
    w = eigvecs[:, max_idx].real
    w = w / w.sum()
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI
    status = "Kabul Edilebilir" if CR < 0.10 else "TUTARSIZ"
    return w, {
        "n": n, "λ_max": round(lambda_max, 4), "CI": round(CI, 4),
        "RI": RI, "CR": round(CR, 4), "Durum": status
    }

def topsis(df_real: pd.DataFrame, weights: np.ndarray) -> float:
    real = df_real["KPI Değeri"].values.astype(float)
    n = len(real)
    best = np.full(n, 3)
    worst = np.full(n, 1)
    data = pd.DataFrame([best, real, worst], index=["Best", "Real", "Worst"])
    norm = data / np.sqrt((data**2).sum())
    weighted = norm * weights
    ideal, anti = weighted.max(), weighted.min()
    d_pos = np.sqrt(((weighted - ideal)**2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti)**2).sum(axis=1))
    return (d_neg / (d_pos + d_neg)).loc["Real"]

def gra(df_real: pd.DataFrame, weights: np.ndarray) -> float:
    real = df_real["KPI Değeri"].values.astype(float)
    n = len(real)
    best = np.full(n, 3)
    worst = np.full(n, 1)
    data = pd.DataFrame([best, real, worst], index=["Best", "Real", "Worst"])
    min_val, max_val = data.min().min(), data.max().max()
    norm = (data - min_val) / (max_val - min_val)
    ref = norm.loc["Best"]
    abs_diff = np.abs(norm.sub(ref, axis=1))
    min_diff, max_diff = abs_diff.min().min(), abs_diff.max().max()
    xi = 0.5
    grc = (min_diff + xi * max_diff) / (abs_diff + xi * max_diff)
    return (grc * weights).sum(axis=1).loc["Real"]

# Arayüz
st.markdown('<div class="title">Kurumsal Sürdürülebilirlik Puan Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Çok Kriterli Karar Verme Yöntemlerine Dayalı Sürdürülebilirlik Performans Ölçümü</div>', unsafe_allow_html=True)

# Dosya Yükleme İçin Sidebar oluşturulması
with st.sidebar:
    st.header("Excel Dosyalarını Yükleyin")
    ekonomik_pair = st.file_uploader("Ekonomik İkili Karşılaştırma Matrisi", type="xlsx")
    cevresel_pair = st.file_uploader("Çevresel İkili Karşılaştırma Matrisi", type="xlsx")
    sosyal_pair = st.file_uploader("Sosyal İkili Karşılaştırma Matrisi", type="xlsx")
    ekonomik_veri = st.file_uploader("Ekonomik Veriler", type="xlsx")
    cevresel_veri = st.file_uploader("Çevresel Veriler", type="xlsx")
    sosyal_veri = st.file_uploader("Sosyal Veriler", type="xlsx")

# Ana Hesaplama Süreci
if st.button("Hesaplamaları Başlat", type="primary", use_container_width=True):
    if all([ekonomik_pair, cevresel_pair, sosyal_pair, ekonomik_veri, cevresel_veri, sosyal_veri]):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = {}
                for name, file in zip(
                    ["ek_p", "ce_p", "so_p", "ek_v", "ce_v", "so_v"],
                    [ekonomik_pair, cevresel_pair, sosyal_pair, ekonomik_veri, cevresel_veri, sosyal_veri]
                ):
                    path = f"{tmp}/{name}.xlsx"
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    paths[name] = path

                # Veri Okuma İşlemi
                ek_pair_df = pd.read_excel(paths["ek_p"], index_col=0)
                ce_pair_df = pd.read_excel(paths["ce_p"], index_col=0)
                so_pair_df = pd.read_excel(paths["so_p"], index_col=0)
                ek_veri = pd.read_excel(paths["ek_v"])
                ce_veri = pd.read_excel(paths["ce_v"])
                so_veri = pd.read_excel(paths["so_v"])

                # AHP Hesaplamalarının Gerçekleştirmeleri
                w_ek, det_ek = ahp_weighting_detailed(ek_pair_df, 1.58)
                w_ce, det_ce = ahp_weighting_detailed(ce_pair_df, 1.59)
                w_so, det_so = ahp_weighting_detailed(so_pair_df, 1.59)

                # TOPSIS & GRA Hesaplamaları
                t_ek = topsis(ek_veri, w_ek)
                t_ce = topsis(ce_veri, w_ce)
                t_so = topsis(so_veri, w_so)
                t_gen = round((t_ek + t_ce + t_so) / 3, 4)

                g_ek = gra(ek_veri, w_ek)
                g_ce = gra(ce_veri, w_ce)
                g_so = gra(so_veri, w_so)
                g_gen = round((g_ek + g_ce + g_so) / 3, 4)

                # Genel Skor Kartı Düzenlemeleri
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>TOPSIS Genel Skor</h3>
                        <h1>%{t_gen*100:.2f}</h1>
                        <p>İdeal Çözüme Yakınlık</p>
                    </div>
                    ''', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>GRA Genel Skor</h3>
                        <h1>%{g_gen*100:.2f}</h1>
                        <p>Gri İlişkisel Derece</p>
                    </div>
                    ''', unsafe_allow_html=True)

                # AHP Tutarlılık Tablosunun Oluşturulması
                st.markdown("""
                <div class="table-header">
                    <h3>AHP Kriterleri Tutarlılık Analizi </h3>
                    <p class="table-subheader">CR < 0.10 → Tutarlı | Yeşil = Kabul Edilebilir</p>
                </div>
                """, unsafe_allow_html=True)

                tutarlılık_data = [
                    {"Boyut": "Ekonomik", **det_ek},
                    {"Boyut": "Çevresel", **det_ce},
                    {"Boyut": "Sosyal", **det_so}
                ]
                tutarlılık_df = pd.DataFrame(tutarlılık_data)
                def color_status(val):
                    return 'background-color: #e8f5e8; color: #1b5e20' if val == "Kabul Edilebilir" else 'background-color: #ffebee; color: #c62828'
                st.dataframe(
                    tutarlılık_df.style.applymap(color_status, subset=['Durum']),
                    use_container_width=True
                )

                # TOPSIS Tablosunun Oluşturulması
                st.markdown("""
                <div class="table-header">
                    <h3>TOPSIS Boyut Skorları</h3>
                    <p class="table-subheader">İdeal Çözüme Göreli Yakınlık</p>
                </div>
                """, unsafe_allow_html=True)

                topsis_df = pd.DataFrame({
                    "Boyut": ["Ekonomik", "Çevresel", "Sosyal"],
                    "TOPSIS Skoru": [round(t_ek, 4), round(t_ce, 4), round(t_so, 4)],
                    "Yüzde (%)": [f"{round(t_ek*100, 2)}%", f"{round(t_ce*100, 2)}%", f"{round(t_so*100, 2)}%"]
                })
                st.dataframe(topsis_df, use_container_width=True)

                # GRA Tablosunun Oluşturulması
                st.markdown("""
                <div class="table-header">
                    <h3>GRA Boyut Skorları</h3>
                    <p class="table-subheader">Gri İlişkisel Derece</p>
                </div>
                """, unsafe_allow_html=True)

                gra_df = pd.DataFrame({
                    "Boyut": ["Ekonomik", "Çevresel", "Sosyal"],
                    "GRA Skoru": [round(g_ek, 4), round(g_ce, 4), round(g_so, 4)],
                    "Yüzde (%)": [f"{round(g_ek*100, 2)}%", f"{round(g_ce*100, 2)}%", f"{round(g_so*100, 2)}%"]
                })
                st.dataframe(gra_df, use_container_width=True)

              # Yöntemler Arası Karşılaştırma Bar Grafiklerinin Oluşturulması
                st.markdown("""
                <div class="table-header">
                    <h3>TOPSIS vs GRA Karşılaştırması</h3>
                    <p class="table-subheader">Boyut Bazında Yan Yana Skor Dağılımı</p>
                </div>
                """, unsafe_allow_html=True)

                chart_data = pd.DataFrame({
                    "Boyut": ["Ekonomik", "Çevresel", "Sosyal"],
                    "TOPSIS": [t_ek, t_ce, t_so],
                    "GRA": [g_ek, g_ce, g_so]
                })

                chart_data_melt = chart_data.melt(id_vars="Boyut", value_vars=["TOPSIS", "GRA"], 
                                                var_name="Yöntem", value_name="Skor")

                fig = px.bar(
                    chart_data_melt,
                    x="Boyut",
                    y="Skor",
                    color="Yöntem",
                    barmode="group",
                    text="Skor",
                    color_discrete_map={
                        "TOPSIS": "#1b5e20",
                        "GRA": "#66bb6a"
                    },
                    height=550,
                    labels={"Skor": "Skor", "Boyut": "Boyut", "Yöntem": "Yöntem"}
                )

                # Barların labelları
                fig.update_traces(
                    texttemplate='%{text:.3f}', 
                    textposition='outside',
                    textfont=dict(size=13, color='black')
                )

                # Layout ayarları
                fig.update_layout(
                    title={
                        'text': "TOPSIS vs GRA Skor Karşılaştırması",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': dict(size=18)
                    },
                    xaxis_title="Boyut",
                    yaxis_title="Skor (0-1)",
                    legend_title="Yöntem",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13),
                    margin=dict(l=60, r=60, t=100, b=80),
                    
                    # X Eksen Etiketleri
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 1, 2],
                        ticktext=["Ekonomik", "Çevresel", "Sosyal"],
                        tickangle=0,  # 0 derece = yatay
                        title_font=dict(size=14),
                        tickfont=dict(size=13),
                        showgrid=False
                    ),
                    yaxis=dict(
                        range=[0, 1.1],
                        showgrid=True,
                        gridcolor='lightgray'
                    ),
                    bargap=0.3,
                    bargroupgap=0.1
                )

                # Y ekseni max değer ayarları
                max_score = chart_data_melt["Skor"].max()
                fig.update_yaxes(range=[0, min(1.1, max_score * 1.3)])

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
    else:
        st.warning("Lütfen tüm 6 Excel dosyasını yükleyin!")

# Arayüz Footerı
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9rem;'>"
            "Onat ÖZÇELİK | Y225052005 | Sakarya Üniversitesi </p>", unsafe_allow_html=True)