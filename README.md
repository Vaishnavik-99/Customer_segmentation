# Customer Segmentation using Machine Learning

## Project Overview
This project segments customers into groups based on age, income, and spending behavior using K-Means clustering.

## Algorithm
- K-Means
- StandardScaler
- Elbow Method

## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn, FastAPI

## Folder Structure
- data/ → dataset
- notebooks/ → analysis
- src/ → ML scripts
- api/ → REST API
- models/ → saved models

## How to Run

Install dependencies:
pip install -r requirements.txt

Train model:
python src/train.py

Run notebook:
jupyter notebook

Run API:
uvicorn api.app:app --reload

## Output
Customers grouped into 4 meaningful clusters.

## Author
Vaishnavi
