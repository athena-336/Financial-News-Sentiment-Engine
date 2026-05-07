# Financial News Sentiment Engine

A full-stack sentiment analysis platform that captures, processes, and 
visualizes financial news sentiment for the Magnificent 7 stocks 
(Apple, Microsoft, Amazon, Alphabet, Meta, Tesla, NVIDIA).

Built for Columbia APAN 5400 · Group 5 · December 2025

## Overview

| Layer | Tech |
|---|---|
| Data Collection | NewsAPI + Scrapy |
| Sentiment Analysis | VADER (NLTK) |
| Storage | MongoDB |
| Backend API | Flask + PyMongo |
| Frontend Dashboard | JavaScript ES6, Chart.js, Bootstrap 5.3 |

## Pipeline
1. **Ingest** — Collect English-language financial news via NewsAPI 
   (Jan 2024 – Nov 2025)
2. **Clean** — Remove duplicates, filter low-quality articles, 
   standardize fields; 50% noise reduction → 250K high-quality articles
3. **Analyze** — VADER sentiment scoring per article 
   (compound score: –1 to +1)
4. **Store** — Raw JSON in MongoDB; aggregated daily sentiment metrics 
   exported to CSV/JSON
5. **Serve** — Flask REST API with time-decay weighted scoring
6. **Visualize** — Interactive dashboard with trend lines and 
   sentiment breakdowns

## Key API Endpoints

| Endpoint | Description |
|---|---|
| `/api/company_summary` | Overall sentiment + date range |
| `/api/company_stats` | Time-series sentiment + breakdown |
| `/api/articles` | Paginated news with scores and metadata |

## Dataset

- **Source:** NewsAPI (Magnificent 7 companies)
- **Time range:** Jan 1, 2024 – Nov 24, 2025
- **Volume:** ~250,000 articles after cleaning
- **Fields:** title, description, content, published_at, source, company

## Business Use Cases

- **Investors & Analysts** — Track sentiment trends to inform buy/sell 
  decisions; identify early risk signals
- **Corporate Teams** — Monitor public perception and competitor 
  sentiment over time

## Course

Columbia University · APAN 5400 · Fall 2025
