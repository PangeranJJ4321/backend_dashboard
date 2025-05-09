import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()

# === Choose your DB type here ===
DB_TYPE = os.getenv("DB_TYPE", "postgresql")  # Options: postgresql / mysql

# === Get credentials from .env ===
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432" if DB_TYPE == "postgresql" else "3306")
DB_NAME = os.getenv("DB_NAME")

# === Create connection URL based on DB_TYPE ===
if DB_TYPE == "postgresql":
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    metadata = MetaData(schema="public")  # PostgreSQL schema
elif DB_TYPE == "mysql":
    DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    metadata = MetaData()  # MySQL doesn't use schema the same way
else:
    raise ValueError("Unsupported DB_TYPE. Use 'postgresql' or 'mysql'.")

# === SQLAlchemy Engine and Session ===
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# === Base for ORM Models ===
Base = declarative_base(metadata=metadata)
