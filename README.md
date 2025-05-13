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

---

### 🧪 Teknologi & Library

* 🐍 **FastAPI** – Backend framework
* 🧠 **Scikit-Learn, XGBoost** – Model ML
* 🐘 **PostgreSQL** – Database
* 🥯 **SQLAlchemy** – ORM
* 📦 **dotenv** – Config environment
* 📊 **Pandas, NumPy** – Data wrangling

---

## 🚀 Cara Jalankan Proyek Ini

### Setup Local Development
```bash
# 1. Clone proyek
git clone https://github.com/PangeranJJ4321/backend_dashboard.git

# 2. Masuk ke direktori
cd backend_dashboard

# 3. Setup environment virtual
python -m venv venv

# 4. Aktifkan environment
# Untuk Windows:
venv\Scripts\activate
# Untuk macOS/Linux:
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Setup database PostgreSQL
# - Pastikan PostgreSQL sudah terinstall dan berjalan
# - Buat database baru

# 7. Buat file .env di root project
DB_TYPE=postgresql
DB_USER=youruser
DB_PASSWORD=yourpass
DB_HOST=localhost
DB_PORT=5432
DB_NAME=film_investment_db
JWT_SECRET=your_super_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# 8. Jalankan migrasi database (jika ada)
alembic upgrade head

# 9. Run server development
uvicorn app.main:app --reload --port 8000
```

### Setup dengan Docker
```bash
# 1. Clone proyek
git clone https://github.com/PangeranJJ4321/backend_dashboard.git

# 2. Masuk ke direktori
cd backend_dashboard

# 3. Buat file .env (sama seperti di atas)

# 4. Build dan jalankan dengan Docker Compose
docker-compose up -d --build

# 5. Untuk mematikan container
docker-compose down
```

## 🤝 Cara Kolaborasi

### 1. Fork dan Clone Repository
```bash
# Fork repository melalui GitHub UI
# Kemudian clone fork kamu
git clone https://github.com/username-kamu/backend_dashboard.git
cd backend_dashboard
```

### 2. Setup Branch
```bash
# Buat branch baru untuk fitur yang akan kamu kerjakan
git checkout -b feature/nama-fitur

# Atau untuk bug fix
git checkout -b fix/nama-bug
```

### 3. Setup Development Environment
Ikuti langkah "Cara Jalankan Proyek Ini" di atas untuk setup environment development.

### 4. Coding Guidelines
- Gunakan type hints untuk semua fungsi
- Dokumentasikan fungsi dan class dengan docstrings
- Ikuti PEP 8 style guide
- Tulis unit test untuk setiap fitur baru

### 5. Membuat Pull Request
```bash
# Commit perubahan kamu
git add .
git commit -m "feat: menambahkan fitur xyz" # Gunakan conventional commits

# Push ke fork repository kamu
git push origin feature/nama-fitur

# Buat Pull Request melalui GitHub UI
```

### 6. Code Review
- Pull Request akan di-review oleh maintainer
- Mungkin akan ada request untuk perubahan
- Setelah disetujui, PR akan di-merge ke main branch

## 🧪 Testing
```bash
# Jalankan unit tests
pytest

# Dengan coverage report
pytest --cov=app
```

## 📦 Struktur Project (Detail)
### Bagian Enpoint
```
app/
│
├── controllers/                 # Logika pengendali (handle request-response)
│   ├── auth_controller.py       # Autentikasi user
│   ├── film_controller.py       # Operasi terkait film
│   └── prediction_controller.py # Handle prediksi film
│
├── core/                        # Konfigurasi dasar
│   ├── config.py                # Konfigurasi aplikasi
│   ├── database.py              # Setup database
│   └── security.py              # Fungsi keamanan (JWT, password)
│
├── data/                        # Dataset mentah (CSV, dll)
│   ├── raw/                     # Data mentah
│   ├── processed/               # Data yang sudah diproses
│   └── models/                  # Model ML yang sudah di-train
│
├── middleware/                  # Middlewares
│   ├── auth_middleware.py       # Middleware autentikasi
│   └── logging_middleware.py    # Middleware logging
│
├── models/                      # Model ML & ORM
│   ├── klasifikasi/             # Model klasifikasi (risiko)
│   │   ├── model.joblib         # Model yang disimpan
│   │   └── train.py             # Script untuk training
│   │
│   ├── regresi/                 # Model regresi (prediksi revenue)
│   │   ├── model.joblib         # Model yang disimpan
│   │   └── train.py             # Script untuk training
│   │
│   ├── user.py                  # ORM model untuk user
│   ├── film.py                  # ORM model untuk film
│   └── prediction.py            # ORM model untuk hasil prediksi
│
├── repositories/                # Akses ke database (CRUD)
│   ├── user_repository.py       # Repository untuk user
│   ├── film_repository.py       # Repository untuk film
│   └── prediction_repository.py # Repository untuk hasil prediksi
│
├── routes/                      # Routing API
│   ├── auth_routes.py           # Endpoint autentikasi
│   ├── film_routes.py           # Endpoint film
│   └── prediction_routes.py     # Endpoint prediksi
│
├── schemas/                     # Pydantic schema untuk validasi data
│   ├── user_schema.py           # Schema untuk user
│   ├── film_schema.py           # Schema untuk film
│   └── prediction_schema.py     # Schema untuk hasil prediksi
│
├── utils/                       # Fungsi utilitas umum
│   ├── feature_engineering.py   # Fungsi untuk feature engineering
│   ├── data_preprocessing.py    # Fungsi untuk preprocessing data
│   └── model_loader.py          # Fungsi untuk load model ML
│
models/
|   |__data/
|   |__models/
|   |__result/
|   |__model_dn.ipynb
├main.py                      # Entry point aplikasi
├.env                         # Environment variables (jangan commit!)
├requirements.txt             # Dependencies
```

## 📋 API Endpoints

### Autentikasi
- `POST /api/auth/register` - Daftar user baru
- `POST /api/auth/login` - Login user
- ...


---

### 🌟 Catatan > Belum yah

Model kamu bisa di-load di dalam `models/klasifikasi/` dan `models/regresi/`, dan dipanggil lewat controller yang pas. Pastikan filenya udah di-`joblib.dump()` atau `pickle`.

---

Feel free buat nambahin apapun atau minta template file lainnya yaa\~
