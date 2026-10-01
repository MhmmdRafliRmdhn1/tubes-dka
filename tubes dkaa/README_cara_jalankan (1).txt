CARA MENJALANKAN PROJECT CREDIT RISK - FUZZY + ML + DL

1. Pastikan menggunakan Python 3.11, bukan Python 3.14.
   Cek dengan:
   py -0p

2. Masuk folder project:
   cd "C:\Users\mhmmd\Documents\tubes dkaa"

3. Hapus virtual environment lama kalau ada:
   rmdir /s /q .venv

4. Buat virtual environment baru:
   py -3.11 -m venv .venv
   .venv\Scripts\activate

5. Install library:
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

6. Simpan dataset credit_risk_dataset.csv di folder yang sama dengan app.py.

7. Jalankan aplikasi:
   python -m streamlit run app.py

8. Jika model belum muncul, buka sidebar aplikasi lalu klik tombol:
   Training / Buat Ulang Model

Catatan perbaikan:
- Path /home/claude sudah diganti menjadi path relatif folder project.
- Sidebar putih sudah diperbaiki menjadi dark theme.
- TensorFlow sekarang tidak membuat aplikasi crash jika belum tersedia.
- ML dan DL dimuat terpisah, jadi ML tetap berjalan walaupun DL belum siap.
- Dataset dicari otomatis di folder project.
- Error dataframe Arrow karena simbol "—" sudah ditangani.
- use_container_width sudah diganti menjadi width="stretch".
