import uuid
from sqlalchemy import Boolean, Column, String, Text, DateTime, ForeignKey, Float, Integer, BigInteger, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

# Association Table untuk Many-to-Many genres <-> predictions
prediction_genres = Table(
    "prediction_genres",
    Base.metadata,
    Column("prediction_id", UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), primary_key=True),
    Column("genre_id", UUID(as_uuid=True), ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True)
)

class User(Base):
    __tablename__ = "users" 

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    photo = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Email verification fields
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String, nullable=True)
    reset_token = Column(String, nullable=True)
    token_expires = Column(DateTime, nullable=True)

    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.name}>"
    
class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="projects")
    predictions = relationship("Prediction", back_populates="project", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    film_title = Column(String(255), nullable=False)
    release_date = Column(DateTime, nullable=False)
    budget = Column(BigInteger, nullable=False)
    predicted_revenue = Column(BigInteger, nullable=False)
    predicted_roi = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    popularity = Column(Float, nullable=True)
    runtime = Column(Integer, nullable=True)
    vote_average = Column(Float, nullable=True)
    vote_count = Column(Integer, nullable=True)
    original_language = Column(String(10), nullable=True)
    feature_importance = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="predictions")
    genres = relationship("Genre", secondary=prediction_genres, back_populates="predictions")

class Genre(Base):
    __tablename__ = "genres"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)

    predictions = relationship("Prediction", secondary=prediction_genres, back_populates="genres")
