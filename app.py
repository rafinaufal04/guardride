# Import Library
import streamlit as st
import joblib
import time
import preprocessing
import kamus
from datetime import datetime
from streamlit_local_storage import LocalStorage
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Set Halaman
st.set_page_config(
    page_title = "GuardRide - Deteksi Penipuan Chat Ojol",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)

# Inisialisasi Local Storage
localS = LocalStorage()

# Session State
if "riwayat" not in st.session_state:
    riwayat_tersimpan = localS.getItem("riwayat_sesi")
    if not isinstance(riwayat_tersimpan, list):
        riwayat_tersimpan = []

    st.session_state.riwayat = riwayat_tersimpan
    st.session_state.jumlah_prediksi = len(riwayat_tersimpan)
    st.session_state.jumlah_penipuan = sum(1 for r in riwayat_tersimpan if r["Prediksi"] == "PENIPUAN")

# Load Model
@st.cache_resource
def load_semua_komponen():
    MODEL_DIR = "./model_output"
    
    model_loaded      = joblib.load(f"{MODEL_DIR}/model_naive_bayes.joblib")
    vectorizer_loaded = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    encoder_loaded    = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")

    stemmer_factory   = StemmerFactory()
    stemmer_loaded    = stemmer_factory.create_stemmer()

    sw_factory        = StopWordRemoverFactory()
    stop_words_loaded = set(sw_factory.get_stop_words()) | kamus.STOP_WORDS_JAWA

    return (
        model_loaded,
        vectorizer_loaded,
        encoder_loaded,
        stemmer_loaded,
        stop_words_loaded,
    )

# Predict
def prediksi_pesan(
    teks_input : str,
    model,
    vectorizer,
    encoder,
    stop_words : set,
    stemmer,
    threshold  : float,
) -> dict:
    teks_bersih  = preprocessing.preprocess_teks_lengkap(teks_input, stop_words, stemmer)

    if not teks_bersih.strip():
        return {"error": "Teks tidak mengandung konten yang bisa dianalisis."}

    vektor       = vectorizer.transform([teks_bersih])
    proba        = model.predict_proba(vektor)[0]

    prob_aman      = float(proba[0])
    prob_penipuan  = float(proba[1])

    prediksi_idx = 1 if proba[1] >= threshold else 0
    prediksi_lbl = encoder.inverse_transform([prediksi_idx])[0]

    if prediksi_lbl == 'penipuan':
        if prob_penipuan >= 0.80:
            status = '🔴 RISIKO SANGAT TINGGI — Kemungkinan besar penipuan!'
            warna  = "error"
        elif prob_penipuan >= 0.70:
            status = '🟠 RISIKO TINGGI — Patut dicurigai sebagai penipuan.'
            warna  = "warning"
        else:
            status = '🟡 RISIKO SEDANG — Perlu perhatian lebih.'
            warna  = "warning"
    else:
        if prob_aman >= 0.80:
            status = '🟢 SANGAT AMAN — Pesan tampak tidak mencurigakan.'
            warna  = "success"
        else:
            status = '🟡 KURANG AMAN — Tetap waspada.'
            warna  = "warning"

    # Ekstraksi Kata Kunci
    kata_kunci = []
    try:
        feature_names = vectorizer.get_feature_names_out()
        input_word_indices = vektor.nonzero()[1]
        
        if len(input_word_indices) > 0:
            class_idx = 1 if proba[1] >= threshold else 0
            log_probs = model.feature_log_prob_[class_idx]
            
            word_weights = [(feature_names[i], log_probs[i]) for i in input_word_indices]
            word_weights.sort(key=lambda x: x[1], reverse=True)

            top_words = word_weights[:3]
            kata_kunci = [word for word, weight in top_words]
    except Exception as e:
        print(f"Error mengekstrak kata kunci: {e}")
        pass

    return {
        "teks_asli"    : teks_input,
        "teks_bersih"  : teks_bersih,
        "prediksi"     : prediksi_lbl,
        "prob_aman"    : round(prob_aman, 4),
        "prob_penipuan": round(prob_penipuan, 4),
        "kepercayaan"  : round(float(max(proba)), 4),
        "status"       : status,
        "warna"        : warna,
        "kata_kunci"   : kata_kunci,
        "threshold"    : 0.2,
        "waktu"        : datetime.now().strftime("%d-%m-%Y %H:%M"),
        "error"        : None,
    }

def halaman_analisis():
    # Load Komponen
    with st.spinner("⏳ Memuat halaman deteksi spam ojol..."):
        model, vectorizer, encoder, stemmer, stop_words = load_semua_komponen()
        
    # Header
    st.title("GuardRide")
    st.subheader("Deteksi Penipuan Chat Ojol")
    st.divider()
    
    # Input Teks
    with st.container(border=True):
        st.subheader("Masukkan Pesan")

        def set_teks_contoh(teks):
            st.session_state.input_teks = teks

        teks_input = st.text_area(
            label       = "Teks pesan chat customer:",
            label_visibility = "collapsed",
            placeholder = ("Masukkan teks pesan customer disini"),
            height      = 25,
            max_chars   = 500,
            help        = "Salin pesan chat customer yang ingin anda analisis.",
            key         = "input_teks",
        )

        # Counter karakter real-time
        jumlah_karakter = len(teks_input)
        st.caption(f"📝 {jumlah_karakter}/500 karakter")

        # Tombol Analisis
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

        with col_btn1:
            tombol_analisis = st.button(
                "🔍 Deteksi Sekarang",
                type             = "primary",
                use_container_width = True,
            )
        with col_btn2:
            # Contoh teks penipuan untuk demo
            st.button(
                "⚠️ Contoh Penipuan", 
                use_container_width=True,
                on_click=set_teks_contoh,
                args=("Mas kirim kode otp-nya, klo ngga akunnya diblokir",)
            )
        with col_btn3:
            # Contoh teks aman untuk demo
            st.button(
                "✅ Contoh Aman", 
                use_container_width=True,
                on_click=set_teks_contoh,
                args=("Sudah sampai mana pak, saya nanti tunggu didepan rumah",)
            )

    # Proses & Tampilkan Hasil
    st.divider()
    st.subheader("Hasil Analisis")

    if tombol_analisis:
        if not teks_input.strip():
            st.warning("⚠️ Silakan masukkan teks pesan terlebih dahulu.")
        else:
            with st.spinner("Menganalisis pesan..."):
                progress_text = "Memproses teks..."
                my_bar = st.progress(0, text=progress_text)
                
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                hasil = prediksi_pesan(
                    teks_input = teks_input,
                    model      = model,
                    vectorizer = vectorizer,
                    encoder    = encoder,
                    stop_words = stop_words,
                    stemmer    = stemmer,
                    threshold  = 0.2,
                )
                my_bar.empty()

            if hasil.get("error"):
                st.error(f"❌ {hasil['error']}")
            else:
                with st.container(border=True):
                    getattr(st, hasil["warna"])(f"### {hasil['status']}")

                    with st.expander("Lihat Detail Probabilitas & Kata Kunci"):
                        c1, c2 = st.columns(2)
                        c1.metric("🟢 Persentase Aman", f"{hasil['prob_aman']*100:.1f}%")
                        c2.metric("🔴 Persentase Penipuan", f"{hasil['prob_penipuan']*100:.1f}%")
                        
                        if hasil.get("kata_kunci"):
                            st.markdown("**Kata kunci pemicu prediksi:**")
                            kata_kunci_format = " ".join([f"` {kata} `" for kata in hasil["kata_kunci"]])
                            st.markdown(kata_kunci_format)

                # Simpan ke Riwayat
                st.session_state.riwayat.append({
                    "Waktu"      : hasil["waktu"],
                    "Pesan"      : teks_input[:100] + ("..." if len(teks_input) > 100 else ""),
                    "Prediksi"   : hasil["prediksi"].upper(),
                })
                localS.setItem("riwayat_sesi", st.session_state.riwayat)
                st.session_state.jumlah_prediksi += 1
                if hasil["prediksi"] == "penipuan":
                    st.session_state.jumlah_penipuan += 1
                
    else:
        st.info("💡 Belum ada teks yang dianalisis. Silakan masukkan teks ke dalam kotak di atas dan tekan **'🔍 Deteksi Sekarang'**.")

# Halaman
halaman_1 = st.Page(halaman_analisis, title="Analisis Pesan", icon="🔍", default=True)
halaman_2 = st.Page("pages/riwayat.py", title="Riwayat Prediksi", icon="📜")
halaman_3 = st.Page("pages/panduan.py", title="Panduan Penggunaan", icon="📖")

pg = st.navigation([halaman_1, halaman_2, halaman_3])

# Footer Sidebar
st.sidebar.markdown(
    "<div style='text-align: center; color: grey;'>GuardRide v1.0</div>", 
    unsafe_allow_html=True
)

pg.run()