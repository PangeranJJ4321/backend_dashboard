import uuid
import pandas as pd
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import Base
from app.models.models import Genre  

def collect_genres_from_data(file_path):
    """Extract unique genres from the dataset"""
    try:
        df = pd.read_csv(file_path)
        all_genres = []
        
        if 'genres' in df.columns:
            for genre_str in df['genres'].dropna():
                if isinstance(genre_str, str):
                    genres = [g.strip() for g in genre_str.split(',')]
                    all_genres.extend(genres)
        
        unique_genres = sorted(list(set(all_genres)))
        return unique_genres
    
    except Exception as e:
        print(f"Error collecting genres: {e}")
        return []

def create_common_genres():
    """Create a list of common movie genres"""
    common_genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
    ]
    return common_genres

def generate_genre_seed_data(data_file=None):
    """
    Generate seed data for the Genre table
    
    Args:
        data_file: Optional path to data file to extract genres from
        
    Returns:
        List of dictionaries with genre data
    """
    all_genres = create_common_genres()
    
    if data_file:
        data_genres = collect_genres_from_data(data_file)
        for genre in data_genres:
            if genre not in all_genres:
                all_genres.append(genre)
    
    genre_seed = []
    for genre_name in all_genres:
        genre_seed.append({
            "id": str(uuid.uuid4()),
            "name": genre_name
        })
    
    return genre_seed

def save_seed_data(seed_data, output_file="genre_seed.json"):
    with open(output_file, 'w') as f:
        json.dump(seed_data, f, indent=2)
    print(f"Seed data saved to {output_file}")

def create_seed_sql(seed_data, output_file="genre_seed.sql"):
    sql_statements = []
    
    # Add header comment
    sql_statements.append("-- Genre table seed data")
    sql_statements.append("-- Generated on " + pd.Timestamp.now().strftime("%Y-%m-%d"))
    sql_statements.append("")
    
    # Create INSERT statements
    for genre in seed_data:
        sql = f"INSERT INTO genres (id, name) VALUES ('{genre['id']}', '{genre['name']}');"
        sql_statements.append(sql)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(sql_statements))
    
    print(f"SQL seed data saved to {output_file}")

def insert_seed_data_to_db(seed_data, db_url):
    """
    Insert seed data directly into the database
    
    Args:
        seed_data: List of genre dictionaries
        db_url: SQLAlchemy database URL
    """
    try:
        # Create engine and session
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Insert genres
        for genre_data in seed_data:
            genre = Genre(id=uuid.UUID(genre_data['id']), name=genre_data['name'])
            session.add(genre)
        
        # Commit changes
        session.commit()
        print(f"Successfully inserted {len(seed_data)} genres into the database")
    
    except Exception as e:
        print(f"Error inserting seed data: {e}")
        if 'session' in locals():
            session.rollback()
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    # Generate seed data
    data_file = "contoh.csv"  # Use your data file
    seed_data = generate_genre_seed_data(data_file)
    
    # Save as JSON and SQL
    save_seed_data(seed_data, "seeds/genre_seed.json")
    create_seed_sql(seed_data, "seeds/genre_seed.sql")
    
    # Print summary
    print(f"\nGenerated seed data for {len(seed_data)} genres:")
    for genre in seed_data[:5]:
        print(f"  - {genre['name']}")
    
    if len(seed_data) > 5:
        print(f"  - ... and {len(seed_data) - 5} more")
    
    # To insert into database, uncomment and configure:
    # DB_URL = "postgresql://username:password@localhost:5432/dbname"
    # insert_seed_data_to_db(seed_data, DB_URL)