## 🎮 **Film Investment Risk Predictor – Backend API**

> 📊 *“Bro, film ini bakal cuan gak sih? Worth gak kalau duit gue yang segunung ini dimasukin buat project ini?”*
> Tenang bro, kita jawabnya pake data, bukan feeling. 💡

---

### 💥 Latar Belakang

Industri film tuh ibarat taruhan besar. Modal bisa tembus jutaan dolar, tapi... gak ada jaminan balik modal.
Makanya, investor dan studio butuh pencerahan.
Nah, proyek ini adalah **alat bantu prediksi dan klasifikasi risiko investasi film** berdasarkan data historis.

---

### 🎯 Tujuan Proyek

Kita bikin backend API yang:

✅ Prediksi pendapatan film (regresi)
✅ Klasifikasi risiko investasi (tinggi/sedang/rendah)
✅ Menyediakan endpoint RESTful untuk konsumsi frontend (dashboard)
✅ Pakai model machine learning kayak Random Forest & XGBoost

---

### 🔍 Pertanyaan Kunci

* **F1**: Faktor apa aja sih yang paling ngaruh ke pendapatan film?
* **F2**: Gimana cara ngelompokin film jadi kategori risiko investasi?

---

### 🧠 Algoritma Machine Learning

#### Untuk Prediksi Revenue (Regresi):

* `RandomForestRegressor`
* `XGBoostRegressor`

#### Untuk Klasifikasi Risiko Investasi:

* `RandomForestClassifier`
* `XGBoostClassifier`

---
Jadi workflow-nya gini kira-kira:

📝 User daftar/login (bisa pakai email & password misalnya).

🔐 Dapat JWT token (biar bisa akses route yang secure).

🧠 User masukin data film (title, budget, genre, dst).

🔮 Backend jalankan model prediksi (regresi & klasifikasi).

💾 Hasil prediksi disimpan ke database (biar bisa dilihat lagi nanti).

📊 Di dashboard, user bisa lihat histori prediksi mereka.
### 🗂️ Struktur Folder

```
app/
│
├── controllers/         # Logika pengendali (handle request-response)
├── core/                # Konfigurasi dasar (DB, environment, settings)
├── data/                # Dataset mentah (CSV, dll)
├── middleware/          # Middlewares (autentikasi, logging, dll)
├── models/              # Model ML & ORM
│   ├── klasifikasi/     # Model klasifikasi (risiko)
│   └── regresi/         # Model regresi (prediksi revenue)
├── repositories/        # Akses ke database (CRUD)
├── routes/              # Routing API
├── schemas/             # Pydantic schema untuk validasi data
└── utils/               # Fungsi utilitas umum (konversi, helper, dll)
```

---

### 🧪 Teknologi & Library

* 🐍 **FastAPI** – Backend framework
* 🧠 **Scikit-Learn, XGBoost** – Model ML
* 🐘 **PostgreSQL** – Database
* 🥯 **SQLAlchemy** – ORM
* 📦 **dotenv** – Config environment
* 📊 **Pandas, NumPy** – Data wrangling

---

### 🚀 Cara Jalanin Proyek Ini

```bash
# 1. Clone proyek
git clone <repo-url>

# 2. Masuk ke direktori
cd film-invest-api

# 3. Setup environment
python -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Buat file .env
DB_TYPE=postgresql
DB_USER=youruser
DB_PASSWORD=yourpass
DB_HOST=localhost
DB_PORT=5432
DB_NAME=yourdbname


# 6. Run server
uvicorn main:app --reload
```

---

### 🌟 Catatan

Model kamu bisa di-load di dalam `models/klasifikasi/` dan `models/regresi/`, dan dipanggil lewat controller yang pas. Pastikan filenya udah di-`joblib.dump()` atau `pickle`.

---

Feel free buat nambahin apapun atau minta template file lainnya yaa\~
