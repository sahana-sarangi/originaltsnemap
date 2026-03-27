'''

import pandas as pd
import numpy as np
import os

DATA_PATH = '.' 

def clean_and_deduplicate(df):
    """Standardizes column names and removes duplicates."""
    df.columns = [c.lower().replace("_", "").replace(" ", "").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# 1. LOAD DATA
data = pd.read_csv(os.path.join(DATA_PATH, "updated_astro_dataset60.csv"), index_col=0, low_memory=False)
data = clean_and_deduplicate(data)

df_tsne = pd.read_csv(os.path.join(DATA_PATH, "updated_fine_tuned_tsne100.csv"), encoding="utf-8-sig", on_bad_lines='skip')
df_tsne = clean_and_deduplicate(df_tsne)
df_tsne = df_tsne.rename(columns={"title": "abstracttitle"})

# 2. MERGE & FIX COLUMNS
df_final = pd.merge(df_tsne, data, on='abstracttitle', how='left')
df_final = df_final.drop_duplicates(subset=['abstracttitle'])

# Dynamic Rename to standardize year, sessiontype, and sessiontitle
for base in ['year', 'sessiontype', 'sessiontitle']:
    matches = [c for c in df_final.columns if base in c]
    if matches:
        df_final = df_final.rename(columns={matches[0]: base})

# 3. CALCULATE METRICS
summary = df_final.groupby('year').size().reset_index(name='No. Abstracts')
sessions_per_year = df_final.groupby('year')['sessiontitle'].nunique().reset_index(name='Total No. Sessions')

session_counts = df_final.groupby(['year', 'sessiontitle']).size().reset_index(name='count')
agg_stats = session_counts.groupby('year')['count'].agg(['mean', 'std']).reset_index()
agg_stats['Avg. session +/- Std Dev'] = agg_stats.apply(
    lambda row: f"{row['mean']:.1f} +/- {row['std']:.1f}" if pd.notnull(row['std']) else f"{row['mean']:.1f} +/- 0.0", axis=1
)

# 4. PIVOT SESSION TYPES
# Note: pivot_table uses the values INSIDE the 'sessiontype' column, which aren't affected by our header cleaning function
session_pivot = df_final.pivot_table(index='year', columns='sessiontype', aggfunc='size', fill_value=0).reset_index()

# UPDATED: Mapping based on your terminal output "Detected Session Types"
column_mapping = {
    'Scientific': 'No. Scientific sessions',
    'Quick Pitch': 'No. Quick Pitch',
    'Clinical Trials': 'No. Clinical Trials',
    'Plenary': 'No. Plenary',
    'Poster Q & A': 'No. Poster Q&A',
    'Digital XP Poster Q & A': 'No. Digital XP poster Q&A',
    'Mini Oral': 'No. Mini Oral'
}
session_pivot = session_pivot.rename(columns=column_mapping)

# 5. ASSEMBLE & ORDER
final_table = summary.merge(sessions_per_year, on='year')
final_table = final_table.merge(agg_stats[['year', 'Avg. session +/- Std Dev']], on='year')
final_table = final_table.merge(session_pivot, on='year')

# Columns required by your table layout
expected_order = [
    'year', 'No. Abstracts', 'Total No. Sessions', 'Avg. session +/- Std Dev',
    'No. Scientific sessions', 'No. Quick Pitch', 'No. Clinical Trials',
    'No. Plenary', 'No. Poster Q&A', 'No. Digital XP poster Q&A', 'No. Mini Oral'
]

# Select only what exists, but print a warning if a main one is missing
existing_cols = [col for col in expected_order if col in final_table.columns]
final_table = final_table[existing_cols]

# Add Total Row
numeric_cols = final_table.select_dtypes(include=[np.number]).columns
total_vals = final_table[numeric_cols].sum().to_dict()
total_vals['year'] = 'Total'
total_vals['Avg. session +/- Std Dev'] = ""
final_table = pd.concat([final_table, pd.DataFrame([total_vals])], ignore_index=True)

# 6. OUTPUT
print("\n--- Research Summary Table ---")
print(final_table.to_string(index=False))
final_table.to_csv("research_summary_table.csv", index=False)
'''

import pandas as pd
import numpy as np
import os

# 1. SETUP PATHS
DATA_PATH = '.' 

def clean_and_deduplicate(df):
    df.columns = [c.lower().replace("_", "").replace(" ", "").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# 2. LOAD DATA
data = pd.read_csv(os.path.join(DATA_PATH, "updated_astro_dataset60.csv"), index_col=0, low_memory=False)
data = clean_and_deduplicate(data)

df_tsne = pd.read_csv(os.path.join(DATA_PATH, "updated_fine_tuned_tsne100.csv"), encoding="utf-8-sig", on_bad_lines='skip')
df_tsne = clean_and_deduplicate(df_tsne)
df_tsne = df_tsne.rename(columns={"title": "abstracttitle"})

# 3. MERGE & STANDARDIZE HEADERS
df_final = pd.merge(df_tsne, data, on='abstracttitle', how='left')
df_final = df_final.drop_duplicates(subset=['abstracttitle'])

# Fix any mangled names from the merge (e.g., yearx, sessiontypex)
for base in ['year', 'sessiontype', 'sessiontitle']:
    matches = [c for c in df_final.columns if base in c]
    if matches:
        df_final = df_final.rename(columns={matches[0]: base})

# 4. CALCULATE METRICS
summary = df_final.groupby('year').size().reset_index(name='No. Abstracts')
sessions_per_year = df_final.groupby('year')['sessiontitle'].nunique().reset_index(name='Total No. Sessions')

# Mean/Std Dev calculation for the "~6 +/- stdev" column
session_counts = df_final.groupby(['year', 'sessiontitle']).size().reset_index(name='count')
agg_stats = session_counts.groupby('year')['count'].agg(['mean', 'std']).reset_index()
agg_stats['Avg. session +/- Std Dev'] = agg_stats.apply(
    lambda row: f"{row['mean']:.1f} +/- {row['std']:.1f}" if pd.notnull(row['std']) else f"{row['mean']:.1f} +/- 0.0", axis=1
)

# 5. PIVOT AND MAP SPECIFIC SESSION COLUMNS
session_pivot = df_final.pivot_table(index='year', columns='sessiontype', aggfunc='size', fill_value=0).reset_index()

# Exact mapping based on your requirements
column_mapping = {
    'Scientific': 'No. Scientific sessions',
    'Quick Pitch': 'No. Quick Pitch',
    'Clinical Trials': 'No. Clinical Trials',
    'Plenary': 'No. Plenary',
    'Poster Q & A': 'No. Poster Q&A',
    'Digital XP Poster Q & A': 'No. Digital XP poster Q&A',
    'Mini Oral': 'No. Mini Oral'
}
session_pivot = session_pivot.rename(columns=column_mapping)

# 6. ASSEMBLE IN EXACT ORDER
final_table = summary.merge(sessions_per_year, on='year')
final_table = final_table.merge(agg_stats[['year', 'Avg. session +/- Std Dev']], on='year')
final_table = final_table.merge(session_pivot, on='year')

# The exact order shown in your images
expected_order = [
    'year', 'No. Abstracts', 'Total No. Sessions', 'Avg. session +/- Std Dev',
    'No. Scientific sessions', 'No. Quick Pitch', 'No. Clinical Trials',
    'No. Plenary', 'No. Poster Q&A', 'No. Digital XP poster Q&A', 'No. Mini Oral'
]

# Filter for columns that exist and reorder
existing_cols = [col for col in expected_order if col in final_table.columns]
final_table = final_table[existing_cols]

# 7. ADD TOTAL ROW
numeric_cols = final_table.select_dtypes(include=[np.number]).columns
total_vals = final_table[numeric_cols].sum().to_dict()
total_vals['year'] = 'Total'
total_vals['Avg. session +/- Std Dev'] = "" # Blank for the summary row

final_table = pd.concat([final_table, pd.DataFrame([total_vals])], ignore_index=True)

# 8. OUTPUT
print("\n--- Finalized Research Table ---")
print(final_table.to_string(index=False))
final_table.to_csv("final_summary_table.csv", index=False)