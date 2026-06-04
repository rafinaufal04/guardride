import re
from kamus import KAMUS_SLANG_GABUNGAN

def normalisasi_slang(teks: str, kamus: dict) -> str:
    """Ganti kata slang dengan kata formal sesuai kamus."""
    return " ".join(kamus.get(kata, kata) for kata in teks.split())

def hapus_karakter_tidak_penting(teks: str) -> str:
    """Hapus URL, angka, tanda baca, dan spasi berlebih."""
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks)
    return teks.strip()

def hapus_stop_words(teks: str, stop_words: set) -> str:
    """Hapus stop words dari teks."""
    return " ".join(w for w in teks.split() if w not in stop_words)

def preprocess_teks_lengkap(teks: str, stop_words: set, stemmer) -> str:
    """
    Pipeline preprocessing lengkap: case folding → normalisasi slang
    → hapus karakter → hapus stop words → stemming.
    """
    if not isinstance(teks, str) or not teks.strip():
        return ""
    teks = teks.lower()
    teks = normalisasi_slang(teks, KAMUS_SLANG_GABUNGAN)
    teks = hapus_karakter_tidak_penting(teks)
    teks = hapus_stop_words(teks, stop_words)
    teks = stemmer.stem(teks)
    return teks
