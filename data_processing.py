"""
M7 News Data Processing Pipeline
=================================
This script processes raw news data collected from NewsAPI for the Magnificent 7 tech companies.
It performs data cleaning, preprocessing, feature engineering, and quality checks.

Author: Data Processing Team
Date: 2025-11-28
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data (run once)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    print("Warning: NLTK not installed. Text preprocessing will be limited.")
    print("Install with: pip install nltk")


class M7DataProcessor:
    """
    Data processor for M7 tech companies news dataset.
    Handles the complete pipeline from raw data to processed, analysis-ready data.
    """
    
    def __init__(self, input_file, output_dir='processed_data'):
        """
        Initialize the data processor.
        
        Args:
            input_file (str): Path to the raw CSV file
            output_dir (str): Directory to save processed data and reports
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'original_count': 0,
            'after_cleaning': 0,
            'duplicates_removed': 0,
            'invalid_removed': 0,
            'missing_removed': 0
        }
        
        # M7 company keywords for classification
        self.company_keywords = {
            'Apple': ['apple', 'iphone', 'ipad', 'mac', 'tim cook', 'ios'],
            'Microsoft': ['microsoft', 'windows', 'azure', 'satya nadella', 'office 365'],
            'Google': ['google', 'alphabet', 'android', 'youtube', 'sundar pichai'],
            'Amazon': ['amazon', 'aws', 'jeff bezos', 'andy jassy', 'prime'],
            'Meta': ['meta', 'facebook', 'instagram', 'whatsapp', 'mark zuckerberg'],
            'Tesla': ['tesla', 'elon musk', 'cybertruck', 'model 3', 'model y', 'ev'],
            'Nvidia': ['nvidia', 'jensen huang', 'gpu', 'cuda', 'geforce']
        }
        
        print("="*70)
        print("M7 News Data Processing Pipeline")
        print("="*70)
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        print()
    
    
    def load_data(self):
        """
        Step 1: Load raw data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print("Step 1: Loading raw data...")
        print("-" * 70)
        
        try:
            df = pd.read_csv(self.input_file, encoding='utf-8')
            self.stats['original_count'] = len(df)
            
            print(f"✓ Successfully loaded {len(df):,} articles")
            print(f"  Columns: {', '.join(df.columns.tolist())}")
            print()
            
            return df
        
        except FileNotFoundError:
            print(f"✗ Error: File '{self.input_file}' not found")
            return None
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    
    def validate_data(self, df):
        """
        Step 2: Validate data structure and required fields.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Validated dataframe
        """
        print("Step 2: Validating data structure...")
        print("-" * 70)
        
        required_fields = ['title', 'publishedAt']
        optional_fields = ['description', 'content', 'author', 'source', 'url']
        
        # Check for required fields
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            print(f"✗ Missing required fields: {', '.join(missing_fields)}")
            return None
        
        print("✓ All required fields present")
        
        # Report on optional fields
        present_optional = [field for field in optional_fields if field in df.columns]
        print(f"  Optional fields present: {', '.join(present_optional)}")
        
        # Check data completeness
        print("\n  Field completeness:")
        for col in df.columns:
            non_null_pct = (df[col].notna().sum() / len(df)) * 100
            print(f"    {col}: {non_null_pct:.1f}% complete")
        
        print()
        return df
    
    
    def clean_data(self, df):
        """
        Step 3: Clean data - remove duplicates, invalid entries, and handle missing values.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        print("Step 3: Cleaning data...")
        print("-" * 70)
        
        original_count = len(df)
        
        # 3.1: Remove articles with [Removed] title
        removed_mask = df['title'].str.contains('\[Removed\]', na=False, regex=True)
        removed_count = removed_mask.sum()
        df = df[~removed_mask]
        print(f"✓ Removed {removed_count} '[Removed]' articles")
        self.stats['invalid_removed'] += removed_count
        
        # 3.2: Remove duplicates based on URL
        if 'url' in df.columns:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=['url'], keep='first')
            dedup_count = before_dedup - len(df)
            print(f"✓ Removed {dedup_count} duplicate articles (by URL)")
            self.stats['duplicates_removed'] += dedup_count
        
        # 3.3: Remove duplicates based on title similarity (exact match)
        before_title_dedup = len(df)
        df = df.drop_duplicates(subset=['title'], keep='first')
        title_dedup_count = before_title_dedup - len(df)
        print(f"✓ Removed {title_dedup_count} duplicate articles (by title)")
        self.stats['duplicates_removed'] += title_dedup_count
        
        # 3.4: Handle missing values in description
        if 'description' in df.columns:
            before_missing = len(df)
            df = df.dropna(subset=['description'])
            missing_count = before_missing - len(df)
            print(f"✓ Removed {missing_count} articles with missing description")
            self.stats['missing_removed'] += missing_count
        
        # 3.5: Remove articles with very short content
        if 'description' in df.columns:
            before_short = len(df)
            df = df[df['description'].str.len() > 30]
            short_count = before_short - len(df)
            print(f"✓ Removed {short_count} articles with too short description (<30 chars)")
            self.stats['invalid_removed'] += short_count
        
        # 3.6: Fill missing author field
        if 'author' in df.columns:
            df['author'] = df['author'].fillna('Unknown')
        
        # 3.7: Validate and filter date range
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        before_date_filter = len(df)
        
        start_date = pd.Timestamp('2024-11-01')
        end_date = pd.Timestamp('2025-11-25')
        
        df = df[(df['publishedAt'] >= start_date) & (df['publishedAt'] <= end_date)]
        date_filtered = before_date_filter - len(df)
        print(f"✓ Removed {date_filtered} articles outside date range (2024-11-01 to 2025-11-25)")
        self.stats['invalid_removed'] += date_filtered
        
        self.stats['after_cleaning'] = len(df)
        
        print(f"\n  Summary: {original_count:,} → {len(df):,} articles")
        print()
        
        return df
    
    
    def preprocess_text(self, text):
        """
        Preprocess a single text string.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords and lemmatize (if NLTK is available)
        try:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            
            words = text.split()
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
            text = ' '.join(words)
        except:
            # If NLTK not available, just remove common stopwords manually
            common_stops = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
            words = text.split()
            words = [w for w in words if w not in common_stops and len(w) > 2]
            text = ' '.join(words)
        
        return text
    
    
    def process_text_fields(self, df):
        """
        Step 4: Preprocess text fields (title, description, content).
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with processed text fields
        """
        print("Step 4: Preprocessing text fields...")
        print("-" * 70)
        
        # Create combined text field for analysis
        text_components = []
        
        if 'title' in df.columns:
            text_components.append(df['title'].fillna(''))
        
        if 'description' in df.columns:
            text_components.append(df['description'].fillna(''))
        
        # Combine title and description
        df['raw_text'] = ' '.join(text_components[0] for text_components in zip(*text_components))
        
        print("  Preprocessing text (this may take a few minutes)...")
        
        # Preprocess the combined text
        df['processed_text'] = df['raw_text'].apply(self.preprocess_text)
        
        # Calculate text length statistics
        df['text_length'] = df['processed_text'].str.len()
        
        print(f"✓ Text preprocessing complete")
        print(f"  Average processed text length: {df['text_length'].mean():.0f} characters")
        print()
        
        return df
    
    
    def extract_features(self, df):
        """
        Step 5: Extract temporal and categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with extracted features
        """
        print("Step 5: Extracting features...")
        print("-" * 70)
        
        # 5.1: Extract temporal features
        df['date'] = df['publishedAt'].dt.date
        df['year'] = df['publishedAt'].dt.year
        df['month'] = df['publishedAt'].dt.month
        df['day'] = df['publishedAt'].dt.day
        df['day_of_week'] = df['publishedAt'].dt.dayofweek
        df['hour'] = df['publishedAt'].dt.hour
        df['week_of_year'] = df['publishedAt'].dt.isocalendar().week
        
        # Add readable day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_name'] = df['day_of_week'].apply(lambda x: day_names[x])
        
        print("✓ Temporal features extracted")
        
        # 5.2: Classify articles by company
        print("  Classifying articles by company...")
        
        def classify_company(text):
            """Classify article to a company based on keywords."""
            if pd.isna(text):
                return 'Unknown'
            
            text_lower = text.lower()
            scores = {}
            
            for company, keywords in self.company_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[company] = score
            
            # Return company with highest score, or 'Unknown' if no matches
            max_score = max(scores.values())
            if max_score > 0:
                return max(scores.items(), key=lambda x: x[1])[0]
            return 'Unknown'
        
        df['company'] = df['raw_text'].apply(classify_company)
        
        print(f"✓ Company classification complete")
        print(f"  Company distribution:")
        for company, count in df['company'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"    {company}: {count:,} articles ({percentage:.1f}%)")
        
        print()
        
        return df
    
    
    def quality_check(self, df):
        """
        Step 6: Perform data quality checks and generate statistics.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Quality check results
        """
        print("Step 6: Performing quality checks...")
        print("-" * 70)
        
        quality_report = {}
        
        # 6.1: Check data completeness
        quality_report['total_articles'] = len(df)
        quality_report['date_range'] = {
            'start': df['publishedAt'].min().strftime('%Y-%m-%d'),
            'end': df['publishedAt'].max().strftime('%Y-%m-%d'),
            'days_covered': (df['publishedAt'].max() - df['publishedAt'].min()).days
        }
        
        # 6.2: Check company distribution
        quality_report['company_distribution'] = df['company'].value_counts().to_dict()
        
        # 6.3: Check temporal distribution
        quality_report['daily_avg'] = len(df) / quality_report['date_range']['days_covered']
        quality_report['articles_by_day'] = df.groupby('date').size().to_dict()
        
        # 6.4: Text quality metrics
        quality_report['text_stats'] = {
            'avg_length': float(df['text_length'].mean()),
            'min_length': int(df['text_length'].min()),
            'max_length': int(df['text_length'].max()),
            'median_length': float(df['text_length'].median())
        }
        
        # 6.5: Check for potential issues
        issues = []
        
        # Check for companies with too few articles
        min_articles = 100
        for company, count in quality_report['company_distribution'].items():
            if company != 'Unknown' and count < min_articles:
                issues.append(f"{company} has only {count} articles (< {min_articles})")
        
        # Check for date gaps
        date_counts = df.groupby('date').size()
        if date_counts.min() < 10:
            issues.append(f"Some dates have very few articles (min: {date_counts.min()})")
        
        quality_report['issues'] = issues
        
        # Print summary
        print("✓ Quality check complete")
        print(f"\n  Dataset Summary:")
        print(f"    Total articles: {quality_report['total_articles']:,}")
        print(f"    Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
        print(f"    Days covered: {quality_report['date_range']['days_covered']}")
        print(f"    Daily average: {quality_report['daily_avg']:.1f} articles/day")
        print(f"    Avg text length: {quality_report['text_stats']['avg_length']:.0f} chars")
        
        if issues:
            print(f"\n  ⚠ Potential Issues Found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  ✓ No major issues detected")
        
        print()
        
        return quality_report
    
    
    def save_processed_data(self, df, quality_report):
        """
        Step 7: Save processed data in multiple formats.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            quality_report (dict): Quality check results
        """
        print("Step 7: Saving processed data...")
        print("-" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 7.1: Save full processed dataset as CSV
        csv_file = self.output_dir / f"m7_processed_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ Saved processed CSV: {csv_file}")
        
        # 7.2: Save as JSON for MongoDB import
        json_file = self.output_dir / f"m7_processed_{timestamp}.json"
        
        # Convert dataframe to list of dictionaries
        records = df.to_dict('records')
        
        # Convert date objects to strings for JSON serialization
        for record in records:
            if 'date' in record:
                record['date'] = str(record['date'])
            if 'publishedAt' in record:
                record['publishedAt'] = record['publishedAt'].isoformat()
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved processed JSON: {json_file}")
        
        # 7.3: Save quality report
        report_file = self.output_dir / f"quality_report_{timestamp}.json"
        
        # Convert date objects in quality report
        report_to_save = quality_report.copy()
        if 'articles_by_day' in report_to_save:
            report_to_save['articles_by_day'] = {
                str(k): v for k, v in report_to_save['articles_by_day'].items()
            }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_to_save, f, indent=2, default=str)
        print(f"✓ Saved quality report: {report_file}")
        
        # 7.4: Save processing statistics
        stats_file = self.output_dir / f"processing_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        print(f"✓ Saved processing statistics: {stats_file}")
        
        print()
    
    
    def generate_visualizations(self, df, quality_report):
        """
        Step 8: Generate visualization charts for the processed data.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            quality_report (dict): Quality check results
        """
        print("Step 8: Generating visualizations...")
        print("-" * 70)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('M7 News Data Processing - Quality Report', fontsize=16, fontweight='bold')
        
        # 8.1: Company distribution
        company_counts = df['company'].value_counts()
        axes[0, 0].bar(company_counts.index, company_counts.values, color='steelblue')
        axes[0, 0].set_title('Articles by Company', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Company')
        axes[0, 0].set_ylabel('Number of Articles')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, (company, count) in enumerate(company_counts.items()):
            axes[0, 0].text(i, count, f'{count:,}', ha='center', va='bottom')
        
        # 8.2: Daily article count over time
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        axes[0, 1].plot(daily_counts['date'], daily_counts['count'], color='darkgreen', linewidth=1.5)
        axes[0, 1].set_title('Daily Article Volume Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Articles')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 8.3: Articles by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['day_name'].value_counts().reindex(day_order)
        
        axes[1, 0].bar(day_counts.index, day_counts.values, color='coral')
        axes[1, 0].set_title('Articles by Day of Week', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Articles')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 8.4: Text length distribution
        axes[1, 1].hist(df['text_length'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Processed Text Length', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Text Length (characters)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(df['text_length'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["text_length"].mean():.0f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = self.output_dir / f"data_quality_report_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {viz_file}")
        
        # Don't show plot in non-interactive environments
        # plt.show()
        plt.close()
        
        print()
    
    
    def print_summary(self):
        """
        Print final processing summary.
        """
        print("="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print("\nProcessing Statistics:")
        print(f"  Original articles: {self.stats['original_count']:,}")
        print(f"  Duplicates removed: {self.stats['duplicates_removed']:,}")
        print(f"  Invalid entries removed: {self.stats['invalid_removed']:,}")
        print(f"  Missing data removed: {self.stats['missing_removed']:,}")
        print(f"  Final dataset size: {self.stats['after_cleaning']:,}")
        
        retention_rate = (self.stats['after_cleaning'] / self.stats['original_count']) * 100
        print(f"\n  Data retention rate: {retention_rate:.1f}%")
        
        print("\nOutput files saved in:", self.output_dir)
        print("\nNext steps:")
        print("  1. Review the quality report and visualizations")
        print("  2. Import processed data into MongoDB")
        print("  3. Proceed with sentiment analysis")
        print("="*70)
    
    
    def run(self):
        """
        Execute the complete data processing pipeline.
        
        Returns:
            pd.DataFrame: Processed dataframe (or None if failed)
        """
        # Step 1: Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Step 2: Validate data
        df = self.validate_data(df)
        if df is None:
            return None
        
        # Step 3: Clean data
        df = self.clean_data(df)
        
        # Step 4: Preprocess text
        df = self.process_text_fields(df)
        
        # Step 5: Extract features
        df = self.extract_features(df)
        
        # Step 6: Quality check
        quality_report = self.quality_check(df)
        
        # Step 7: Save processed data
        self.save_processed_data(df, quality_report)
        
        # Step 8: Generate visualizations
        self.generate_visualizations(df, quality_report)
        
        # Print summary
        self.print_summary()
        
        return df


def main():
    """
    Main function to run the data processing pipeline.
    """
    # Configuration
    INPUT_FILE = "m7_data.csv"  # Input CSV file from NewsAPI
    OUTPUT_DIR = "processed_data"  # Output directory
    
    # Create processor instance
    processor = M7DataProcessor(INPUT_FILE, OUTPUT_DIR)
    
    # Run the complete pipeline
    processed_df = processor.run()
    
    # Return processed dataframe for further use
    return processed_df


if __name__ == "__main__":
    # Run the pipeline
    df = main()