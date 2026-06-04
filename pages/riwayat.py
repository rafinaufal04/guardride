import streamlit as st
import pandas as pd
from streamlit_local_storage import LocalStorage

localS = LocalStorage()

st.title("Riwayat Prediksi")
st.divider()

if not st.session_state.get("riwayat"):
    st.info("Belum ada prediksi yang dilakukan. Mulai analisis di halaman '🔍 Analisis Pesan'.")
else:
    df_riwayat = pd.DataFrame(st.session_state.riwayat)
    st.dataframe(
        df_riwayat,
        use_container_width = True,
        hide_index          = True,
    )

    st.divider()
    
    st.subheader("Ringkasan Riwayat")
    col_r1, col_r2, col_r3 = st.columns(3)
    total_r   = len(df_riwayat)
    penipuan_r = (df_riwayat["Prediksi"] == "PENIPUAN").sum()
    aman_r    = total_r - penipuan_r

    col_r1.metric("Total Dianalisis", total_r)
    col_r2.metric("🟢 Aman", aman_r)
    col_r3.metric("🔴 Penipuan", penipuan_r)

    st.divider()

    if st.button("🗑️ Reset Riwayat", use_container_width=True, type="primary"):
        st.session_state.riwayat          = []
        st.session_state.jumlah_prediksi  = 0
        st.session_state.jumlah_penipuan  = 0
        localS.setItem("riwayat_sesi", [])
        st.rerun()
