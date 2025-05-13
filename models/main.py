import os
import argparse
import pandas as pd
from model.preprocessing import preprocess_data
from model.train_models import train_models, create_directories
from model.predict import predict, format_for_api
from seeds.gendre_seed import generate_genre_seed_data, save_seed_data, create_seed_sql

def main():
    parser = argparse.ArgumentParser(description="Movie Revenue Prediction and Risk Classification")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess raw data")
    preprocess_parser.add_argument("--input", required=True, help="Input data file path")
    preprocess_parser.add_argument("--output", default="processed_data.csv", help="Output file path")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--data", required=True, help="Data file path for training")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--input", required=True, help="Input data file path")
    predict_parser.add_argument("--output", default="prediction_results.csv", help="Output file path")
    
    # Seed command
    seed_parser = subparsers.add_parser("seed", help="Generate genre seed data")
    seed_parser.add_argument("--data", help="Optional data file to extract genres from")
    seed_parser.add_argument("--json", default="seeds/genre_seed.json", help="Output JSON file path")
    seed_parser.add_argument("--sql", default="seeds/genre_seed.sql", help="Output SQL file path")
    
    # All command
    all_parser = subparsers.add_parser("all", help="Run entire pipeline")
    all_parser.add_argument("--data", required=True, help="Input data file path")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models/klasifikasi", exist_ok=True)
    os.makedirs("models/regresi", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("seeds", exist_ok=True)
    
    if args.command == "preprocess":
        print(f"Preprocessing data from {args.input}...")
        df = pd.read_csv(args.input)
        processed_df = preprocess_data(df)
        processed_df.to_csv(args.output, index=False)
        print(f"Preprocessed data saved to {args.output}")
    
    elif args.command == "train":
        print(f"Training models using data from {args.data}...")
        train_models(args.data)
    
    elif args.command == "predict":
        print(f"Generating predictions for {args.input}...")
        predictions = predict(args.input, args.output)
        formatted = format_for_api(predictions)
        print(f"Predictions saved to {args.output}")
        
        # Print sample prediction
        if len(formatted) > 0:
            sample = formatted[0]
            print("\nSample prediction:")
            print(f"Title: {sample['film_title']}")
            print(f"Budget: ${sample['budget']:,}")
            print(f"Predicted Revenue: ${sample['predicted_revenue']:,.2f}")
            print(f"Predicted ROI: {sample['predicted_roi']:.2f}%")
            print(f"Risk Level: {sample['risk_level']}")
    
    elif args.command == "seed":
        print("Generating genre seed data...")
        seed_data = generate_genre_seed_data(args.data)
        save_seed_data(seed_data, args.json)
        create_seed_sql(seed_data, args.sql)
    
    elif args.command == "all":
        print(f"Running entire pipeline with data from {args.data}...")
        
        # Step 1: Preprocess
        print("\n=== Step 1: Preprocessing data ===")
        df = pd.read_csv(args.data)
        processed_df = preprocess_data(df)
        processed_df.to_csv("processed_data.csv", index=False)
        print("Preprocessed data saved to processed_data.csv")
        
        # Step 2: Train models
        print("\n=== Step 2: Training models ===")
        create_directories()
        train_models(args.data)
        
        # Step 3: Make predictions
        print("\n=== Step 3: Generate predictions ===")
        predictions = predict(args.data, "prediction_results.csv")
        
        # Step 4: Generate seed data
        print("\n=== Step 4: Generate genre seed data ===")
        seed_data = generate_genre_seed_data(args.data)
        save_seed_data(seed_data, "seeds/genre_seed.json")
        create_seed_sql(seed_data, "seeds/genre_seed.sql")
        
        print("\nPipeline completed successfully!")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()