
'''
import pandas as pd
import numpy as np
import altair as alt
import os

DATA_PATH = '.'
alt.data_transformers.disable_max_rows()

def add_leading_zeroes(x):
    if pd.isna(x):
        return "00"
    return "{:02d}".format(int(x))

data = pd.read_csv(os.path.join(DATA_PATH, "updated_astro_dataset60.csv"), index_col=0, low_memory=False)
data['years'] = data['years'].fillna(0)
data.years = data.years.astype(int)
data = data.rename(columns={"years": "Year"})

file_path_tsne = os.path.join(DATA_PATH, "updated_fine_tuned_tsne60.csv")
with open(file_path_tsne, encoding="utf8", errors='ignore') as temp_f:
    df = pd.read_csv(temp_f)
    df = df.rename(columns={
        "Topic Name (Post Forced)": "Cluster_ID_Full",
        "x": "TSNE-x",
        "y": "TSNE-y",
        "title": "AbstractTitle",
        "abstract": "Abstract"
    })
    df["Topic (Post Forced)"] = df["Topic (Post Forced)"].fillna(0).astype(int)
    df = pd.merge(df, data, on=['AbstractTitle'], suffixes=("_df", None))
    df = df.drop(columns=df.filter(regex="_df$").columns)
    df["Cluster_Num"] = df["Topic (Post Forced)"].apply(add_leading_zeroes)

file_path_names = os.path.join(DATA_PATH, "updated_fine_tuned_tnse60_w_names_final_ver.csv")
bt60_names = pd.read_csv(file_path_names)
bt60_names = bt60_names.rename(columns={"title": "AbstractTitle"})
bt60_names["Topic (Post Forced)"] = bt60_names["Topic (Post Forced)"].fillna(0).astype(int)

df = pd.merge(
    df,
    bt60_names[["AbstractTitle", "GPT_Names", "Topic (Post Forced)"]],
    on=["AbstractTitle", "Topic (Post Forced)"],
    how="left"
)

df = df.rename(columns={"GPT_Names": "TopicName"})
df["TopicName"] = df["TopicName"].fillna("Topic " + df["Topic (Post Forced)"].astype(str))

for col in df.columns:
    clean_col = col.lower().replace("_", "").replace(" ", "").strip()
    if clean_col == "sessiontitle":
        df = df.rename(columns={col: "session_title"})
    elif clean_col == "sessiontype":
        df = df.rename(columns={col: "session_type"})

df = df.loc[:, ~df.columns.duplicated()]

print("Unique Columns used for tooltips:", [c for c in df.columns if "session" in c.lower()])

keep_cols = [
    'TSNE-x', 'TSNE-y', 'Cluster_Num', 'Cluster_ID_Full', 'Year', 
    'TopicName', 'AbstractTitle', 'Abstract'
]

if 'session_title' in df.columns: keep_cols.append('session_title')
if 'session_type' in df.columns: keep_cols.append('session_type')

keep_cols = [c for c in keep_cols if c in df.columns]
df_final = df[keep_cols].copy()

selection = alt.selection_point(fields=['Cluster_Num'], bind='legend')

final_chart = (
    alt.Chart(df_final)
    .mark_circle(size=25)
    .encode(
        x=alt.X('TSNE-x:Q', title='TSNE-x'),
        y=alt.Y('TSNE-y:Q', title='TSNE-y'),
        color=alt.Color(
            'Cluster_Num:N',
            title='Cluster ID',
            scale=alt.Scale(scheme='tableau20'),
            legend=alt.Legend(
                orient='right',
                columns=2,
                symbolSize=80,
                labelFontSize=10,
                titleFontSize=12,
                symbolLimit=100
            )
        ),
        opacity=alt.condition(selection, alt.value(0.9), alt.value(0.05)),
        tooltip=[
            alt.Tooltip('Cluster_ID_Full:N', title='Cluster'),
            alt.Tooltip('Year:O', title='Year'),
            alt.Tooltip('session_title:N', title='Session Title'),
            alt.Tooltip('TopicName:N', title='Cluster Name'),
            alt.Tooltip('session_type:N', title='Session Type'),
            alt.Tooltip('AbstractTitle:N', title='Abstract Title'),
            alt.Tooltip('Abstract:N', title='Abstract')
        ]
    )
    .properties(
        width=800,
        height=700
    )
    .add_params(selection)
    .interactive()
    .configure_title(fontSize=18, anchor='start')
    .configure_axis(labelFontSize=12, titleFontSize=14, grid=True)
    .configure_view(strokeWidth=0)
)

final_chart.save('tsne_medical_clusters.html')
'''

import pandas as pd
import numpy as np
import altair as alt
import os

DATA_PATH = '.'
alt.data_transformers.disable_max_rows()

def add_leading_zeroes(x):
    if pd.isna(x):
        return "00"
    return "{:02d}".format(int(x))

data = pd.read_csv(os.path.join(DATA_PATH, "updated_astro_dataset60.csv"), index_col=0, low_memory=False)
data['years'] = data['years'].fillna(0)
data.years = data.years.astype(int)
data = data.rename(columns={"years": "Year"})

file_path_tsne = os.path.join(DATA_PATH, "updated_fine_tuned_tsne60.csv")
with open(file_path_tsne, encoding="utf8", errors='ignore') as temp_f:
    df = pd.read_csv(temp_f)
    df = df.rename(columns={
        "Topic Name (Post Forced)": "Cluster_ID_Full",
        "x": "TSNE-x",
        "y": "TSNE-y",
        "title": "AbstractTitle",
        "abstract": "Abstract"
    })
    df["Topic (Post Forced)"] = df["Topic (Post Forced)"].fillna(0).astype(int)
    df = pd.merge(df, data, on=['AbstractTitle'], suffixes=("_df", None))
    df = df.drop(columns=df.filter(regex="_df$").columns)
    df["Cluster_Num"] = df["Topic (Post Forced)"].apply(add_leading_zeroes)

file_path_names = os.path.join(DATA_PATH, "updated_fine_tuned_tnse60_w_names_final_ver.csv")
bt60_names = pd.read_csv(file_path_names)
bt60_names = bt60_names.rename(columns={"title": "AbstractTitle"})
bt60_names["Topic (Post Forced)"] = bt60_names["Topic (Post Forced)"].fillna(0).astype(int)

df = pd.merge(
    df,
    bt60_names[["AbstractTitle", "GPT_Names", "Topic (Post Forced)"]],
    on=["AbstractTitle", "Topic (Post Forced)"],
    how="left"
)

df = df.rename(columns={"GPT_Names": "TopicName"})
df["TopicName"] = df["TopicName"].fillna("Topic " + df["Topic (Post Forced)"].astype(str))

for col in df.columns:
    clean_col = col.lower().replace("_", "").replace(" ", "").strip()
    if clean_col == "sessiontitle":
        df = df.rename(columns={col: "session_title"})
    elif clean_col == "sessiontype":
        df = df.rename(columns={col: "session_type"})

df = df.loc[:, ~df.columns.duplicated()]

keep_cols = [
    'TSNE-x', 'TSNE-y', 'Cluster_Num', 'Cluster_ID_Full', 'Year', 
    'TopicName', 'AbstractTitle', 'Abstract', 'session_title', 'session_type'
]
keep_cols = [c for c in keep_cols if c in df.columns]
df_final = df[keep_cols].copy()

# --- INTERACTIVE WIDGETS ---

years = sorted(df_final['Year'].unique().tolist())
year_dropdown = alt.binding_select(options=[None] + years, labels=['All'] + [str(y) for y in years], name='Year: ')
year_selection = alt.selection_point(fields=['Year'], bind=year_dropdown)

session_types = sorted(df_final['session_type'].dropna().unique().tolist())
session_dropdown = alt.binding_select(options=[None] + session_types, labels=['All'] + session_types, name='Session Type: ')
session_selection = alt.selection_point(fields=['session_type'], bind=session_dropdown)

# FIX: Using alt.binding with input='text' instead of alt.binding_text
search_param = alt.param(
    value='',
    bind=alt.binding(input='text', placeholder='Keyword Search', name='Search: ')
)

legend_selection = alt.selection_point(fields=['Cluster_Num'], bind='legend')

# --- PLOT ---

final_chart = (
    alt.Chart(df_final)
    .mark_circle(size=25)
    .encode(
        x=alt.X('TSNE-x:Q', title='TSNE-x'),
        y=alt.Y('TSNE-y:Q', title='TSNE-y'),
        color=alt.Color(
            'Cluster_Num:N',
            title='Cluster ID',
            scale=alt.Scale(scheme='tableau20'),
            legend=alt.Legend(orient='right', columns=2, symbolLimit=100)
        ),
        opacity=alt.condition(
            year_selection & session_selection & legend_selection, 
            alt.value(0.9), 
            alt.value(0.01)
        ),
        tooltip=[
            alt.Tooltip('Cluster_ID_Full:N', title='Cluster'),
            alt.Tooltip('Year:O', title='Year'),
            alt.Tooltip('session_title:N', title='Session Title'),
            alt.Tooltip('TopicName:N', title='Cluster Name'),
            alt.Tooltip('session_type:N', title='Session Type'),
            alt.Tooltip('AbstractTitle:N', title='Abstract Title'),
            alt.Tooltip('Abstract:N', title='Abstract')
        ]
    )
    .add_params(
        year_selection, 
        session_selection, 
        legend_selection,
        search_param
    )
    # This filter handles the keyword search dynamically
    .transform_filter(
        alt.expr.test(alt.expr.regexp(search_param, 'i'), alt.datum.AbstractTitle)
    )
    .properties(
        width=800,
        height=700
    )
    .interactive()
    .configure_title(fontSize=18, anchor='start')
    .configure_axis(labelFontSize=12, titleFontSize=14, grid=True)
    .configure_view(strokeWidth=0)
)

final_chart.save('tsne_original.html')