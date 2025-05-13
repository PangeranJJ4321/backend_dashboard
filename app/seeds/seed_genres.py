import uuid
from app.core.database import SessionLocal  
from app.models.models import Genre  
from app.core.database import engine  

def create_common_genres():
    return [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
    ]

def seed_genres():
    session = SessionLocal()
    try:

        genres = create_common_genres()

        for genre_name in genres:
            existing = session.query(Genre).filter_by(name=genre_name).first()
            if not existing:
                new_genre = Genre(id=uuid.uuid4(), name=genre_name)
                session.add(new_genre)

        session.commit()
        print("✅ Genre seeding completed.")
    except Exception as e:
        print("❌ Error during seeding:", e)
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    seed_genres()
