"""
LingoMap column clusterer (v4 - Final fusion version)
- Core idea: Use manual descriptions combined with data samples to enhance semantic richness.
- Stability: Use a fixed number of clusters k, specified by experts, to ensure predictable and accurate results.
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
import sys
import re


# Ensure that modules in the src folder can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_hybrid_semantic_document(column_name: str, column_descriptions: dict, df: pd.DataFrame) -> str:
    """
    Create a hybrid semantic document that combines manual descriptions and data samples for a column.
    """
    # 1. Get the core manual description
    manual_description = column_descriptions.get(column_name, "No specific description.")

    # 2. Get data samples as content evidence
    if df[column_name].isnull().all():
        samples_str = "all values are empty"
    else:
        # Randomly sample up to 10 non-empty samples
        samples = df[column_name].dropna().sample(n=min(10, len(df[column_name].dropna()))).tolist()
        samples_str = ", ".join(map(str, samples))

    # 3. Combine the final, rich semantic document
    document = (
        f"Column Name: '{column_name}'. "
        f"Expert Description: '{manual_description}'. "
        f"Data Samples: [{samples_str}]."
    )
    return document

def generate_intuitive_cluster_name(columns: List[str], descriptions: dict) -> str:
    """
    Generate a more intuitive cluster name based on the columns and their manual descriptions within the group.
    """
    words = []
    for col in columns:
        desc = descriptions.get(col, '')
        match = re.search(r'\((.*?)\)', desc)
        if match:
            words.extend(match.group(1).lower().split())

    if not words:
        # If there is no English description, use the column name itself to determine
        lower_columns = [c.lower() for c in columns]
        if any(w in ' '.join(lower_columns) for w in ['address', 'city', 'state', 'zip']):
            return "Branch Physical Address"
        return f"{columns[0]} & related"

    word_counts = pd.Series(words).value_counts()
    top_word = word_counts.index[0]

    # Based on expert knowledge naming rules
    if top_word in ['address', 'city', 'state', 'zip', 'county']:
        return "Branch Physical Address"
    if top_word in ['statistical', 'area']:
        return "Statistical Areas"
    if top_word in ['date']:
        return "Key Dates"
    if top_word in ['number', 'certificate', 'code', 'id', 'uninum', 'fdic']:
        return "Identifiers & Codes"
    if top_word in ['name', 'class', 'status', 'office', 'institution', 'type']:
         return "Institution & Branch Descriptors"
    if top_word in ['coordinate', 'latitude', 'longitude']:
         return "Geographic Coordinates"
    
    return f"{top_word.title()} Related Fields"


def cluster_columns(df: pd.DataFrame = None, csv_file_path: str = None) -> List[Dict[str, Any]]:
    """Cluster the columns of a DataFrame."""
    if df is None:
        if csv_file_path and os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path, low_memory=False)
        else:
            raise FileNotFoundError("Must provide a DataFrame or a valid CSV file path.")

    # --- [Core] The most valuable manual description data you provided ---
    column_descriptions = {
        "ACQDATE": "Acquisition Date", "ADDRESS": "Branch Address",
        "ADDRESS2": "Street Address Line 2", "BKCLASS": "Institution Class",
        "CBSA": "Core Based Statistical Area Name", "CBSA_DIV": "Metropolitan Divisions Name",
        "CBSA_DIV_FLG": "Metropolitan Divisions Flag", "CBSA_DIV_NO": "Metropolitan Divisions Number",
        "CBSA_METRO": "Metropolitan Division Number", "CBSA_METRO_FLG": "Metropolitan Division Flag",
        "CBSA_METRO_NAME": "Metropolitan Division Name", "CBSA_MICRO_FLG": "Micropolitan Division Flag",
        "CBSA_NO": "Core Based Statistical Areas", "CERT": "Institution FDIC Certificate #",
        "CITY": "Branch City", "COUNTY": "Branch County", "CSA": "Combined Statistical Area Name",
        "CSA_FLG": "Combined Statistical Area Flag", "CSA_NO": "Combined Statistical Area Number",
        "ESTYMD": "Branch Established Date", "FI_UNINUM": "FDIC UNINUM of the Owner Institution",
        "ID": "ID", "LATITUDE": "Latitude", "LONGITUDE": "Longitude", "MAINOFF": "Main Office",
        "MDI_STATUS_CODE": "Minority Status Code", "MDI_STATUS_DESC": "Minority Status Description",
        "NAME": "Institution Name", "OFFNAME": "Office Name", "OFFNUM": "Branch Number",
        "RUNDATE": "Run Date", "SERVTYPE": "Service Type Code",
        "SERVTYPE_DESC": "Service Type Description", "STALP": "Branch State Abbreviation",
        "STCNTY": "State and County Number", "STNAME": "Branch State",
        "UNINUM": "Unique Identification Number for a Branch Office", "ZIP": "Branch Zip Code",
        "X": "X Coordinate, likely Longitude", "Y": "Y Coordinate, likely Latitude"
    }
    
    # Ensure that all columns in the DataFrame are in the description dictionary
    for col in df.columns:
        if col not in column_descriptions:
            column_descriptions[col] = f"No description available for {col}"

    column_names = df.columns.tolist()
    
    print("📝 Creating hybrid semantic documents for each column...")
    semantic_documents = [create_hybrid_semantic_document(col, column_descriptions, df) for col in column_names]
    
    print("🔄 Loading embedding model and performing vectorization...")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(semantic_documents)
    
    # --- Use a fixed k value based on expert knowledge to ensure accuracy and stability ---
    n_clusters = 8
    print(f"🎯 Using expert-specified k={n_clusters} for K-means clustering...")
    # ------------------------------------------------------------------

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)
    
    clusters_by_id = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(cluster_labels):
        clusters_by_id[label].append(column_names[i])
    
    print("🏷️ Generating names for clusters...")
    final_clusters = []
    for cluster_id, columns in clusters_by_id.items():
        if not columns: continue
        cluster_name = generate_intuitive_cluster_name(columns, column_descriptions)
        final_clusters.append({'name': cluster_name, 'columns': sorted(columns)})

    print(f"✅ Clustering completed! {len(final_clusters)} clusters")
    return sorted(final_clusters, key=lambda x: x['name'])


if __name__ == "__main__":
    print("🧪 Testing column clustering functionality (v4 Final)...")
    try:
        test_csv_path = 'FDIC_Insured_Banks.csv'
        clusters = cluster_columns(csv_file_path=test_csv_path)
        
        print("\n📋 Clustering results:")
        for i, cluster in enumerate(clusters):
            print(f"\n--- Cluster {i+1}: {cluster['name']} ---")
            print(f"   {', '.join(cluster['columns'])}")
        
    except Exception as e:
        import traceback
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

