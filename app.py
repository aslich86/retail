import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
from datetime import timedelta

# Konfigurasi Halaman (Tab Browser)
st.set_page_config(page_title="Retail BI Dashboard", page_icon="📈", layout="wide")

# Fungsi memuat data dan model dari Colab
@st.cache_resource
def load_data():
    try:
        with open('retail_bi_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Ambil paket data
data_pack = load_data()

if data_pack is None:
    st.error("⚠️ File 'retail_bi_model.pkl' tidak ditemukan! Pastikan sudah di-upload ke GitHub sejajar dengan app.py")
    st.stop()

df = data_pack['data_master']
model = data_pack['model_prediksi']
start_date = data_pack['tanggal_awal']
last_date = data_pack['tanggal_akhir']

# --- HEADER ---
st.title("📈 Smart Retail BI & Forecasting")
st.markdown(f"Update Data Terakhir: **{last_date.strftime('%d %B %Y')}**")
st.markdown("---")

# --- BAGIAN 1: KPI (Key Performance Indicators) ---
# Menghitung angka ringkasan (REVISI: Menggunakan kolom 'Sales' bukan 'Total')
total_revenue = df['Sales'].sum()
total_gross_income = df['gross income'].sum()
top_category = df.groupby('Product line')['Quantity'].sum().idxmax()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Omzet (Revenue)", f"${total_revenue:,.2f}")
with col2:
    st.metric("Total Laba Kotor (Gross)", f"${total_gross_income:,.2f}")
with col3:
    st.metric("Kategori Terlaris", top_category)

st.markdown("---")

# --- BAGIAN 2: VISUALISASI ANALITIK ---
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("🛒 Penjualan per Kategori Produk")
    # REVISI: Menggunakan kolom 'Sales'
    df_cat = df.groupby('Product line')['Sales'].sum().reset_index()
    fig1 = px.bar(df_cat, x='Sales', y='Product line', orientation='h', 
                  color='Sales', color_continuous_scale='Viridis')
    fig1.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    st.subheader("🏙️ Kontribusi Laba per Kota")
    df_city = df.groupby('City')['gross income'].sum().reset_index()
    fig2 = px.pie(df_city, values='gross income', names='City', hole=0.4,
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- BAGIAN 3: PREDIKSI MASA DEPAN (FORECASTING) ---
st.subheader("🤖 Prediksi Tren Laba (30 Hari ke Depan)")
st.info("AI menggunakan Linear Regression untuk memproyeksikan laba berdasarkan tren historis.")

# Generate data 30 hari ke depan
last_day_idx = (last_date - start_date).days
future_days = np.array([[last_day_idx + i] for i in range(1, 31)])
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

# Prediksi menggunakan model pkl
predictions = model.predict(future_days)

df_pred = pd.DataFrame({
    'Tanggal': future_dates,
    'Estimasi Laba ($)': predictions
})

col_pred1, col_pred2 = st.columns([2, 1])

with col_pred1:
    fig_pred = px.area(df_pred, x='Tanggal', y='Estimasi Laba ($)', 
                       color_discrete_sequence=['#8b5cf6'])
    fig_pred.update_layout(xaxis_title="Periode Mendatang", yaxis_title="Prediksi Laba ($)")
    st.plotly_chart(fig_pred, use_container_width=True)

with col_pred2:
    st.write("📋 Daftar Estimasi Harian")
    st.dataframe(df_pred.style.format({'Estimasi Laba ($)': '{:.2f}'}), 
                 use_container_width=True, hide_index=True)

st.markdown("---")
with st.expander("🔍 Lihat Detail Dataset Mentah"):
    st.dataframe(df.head(100), use_container_width=True)
