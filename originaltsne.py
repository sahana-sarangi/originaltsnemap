

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

years = sorted(df_final['Year'].unique().tolist())
year_dropdown = alt.binding_select(options=[None] + years, labels=['All'] + [str(y) for y in years], name='Year: ')
year_selection = alt.selection_point(fields=['Year'], bind=year_dropdown)

session_types = sorted(df_final['session_type'].dropna().unique().tolist())
session_dropdown = alt.binding_select(options=[None] + session_types, labels=['All'] + session_types, name='Session Type: ')
session_selection = alt.selection_point(fields=['session_type'], bind=session_dropdown)

search_param = alt.param(
    value='',
    bind=alt.binding(input='text', placeholder='Keyword Search', name='Search: ')
)

legend_selection = alt.selection_point(fields=['Cluster_Num'], bind='legend')

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
    .add_params(year_selection, session_selection, legend_selection, search_param)
    .transform_filter(alt.expr.test(alt.expr.regexp(search_param, 'i'), alt.datum.AbstractTitle))
    .properties(width=950, height=750)
    .interactive()
    .configure_title(fontSize=18, anchor='start')
    .configure_axis(labelFontSize=12, titleFontSize=14, grid=True)
    .configure_view(strokeWidth=0)
)

chart_json = final_chart.to_json()

html_template = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    body {{
      background-color: #f4f7f9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }}
    .header-bar {{
      width: 100%;
      background-color: #2c3e50;
      color: white;
      padding: 40px 0;
      text-align: center;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      margin-bottom: 50px;
    }}
    .header-bar h1 {{ margin: 0; font-size: 32px; font-weight: 700; letter-spacing: 0.5px; }}
    .header-bar p {{ margin: 12px 0 0 0; opacity: 0.85; font-size: 18px; }}
    
    .card {{
      background: white;
      border-radius: 16px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.1);
      padding: 60px; /* Increased padding inside the card */
      width: fit-content;
      min-width: 1100px; /* Wider card to accommodate legend + graph */
      margin-bottom: 80px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}

    #vis {{ 
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 100%;
    }}

    .vega-embed {{
      display: flex !important;
      flex-direction: column;
      align-items: center;
    }}

    canvas.vega-canvas {{ order: 1; }}

    .vega-binds {{
      order: 2; 
      margin-top: 50px; /* More space between graph and filters */
      display: flex;
      justify-content: center;
      align-items: center;
      flex-wrap: wrap;
      gap: 40px;
      width: 100%;
    }}

    .vega-bind {{
      font-size: 16px;
      display: flex;
      align-items: center;
    }}

    .vega-bind-name {{ 
      font-weight: 600; 
      margin-right: 12px;
      color: #2c3e50;
    }}

    select, input {{
      padding: 10px 15px;
      border-radius: 8px;
      border: 1.5px solid #dcdde1;
      background-color: #fff;
      font-size: 14px;
      transition: border-color 0.2s;
    }}

    select:focus, input:focus {{
      outline: none;
      border-color: #3498db;
    }}
  </style>
</head>
<body>
  <div class="header-bar">
    <h1>t-SNE Visualization</h1>
    <p>subtitle</p>
  </div>
  <div class="card">
    <div id="vis"></div>
  </div>
  <script>
    const spec = {chart_json};
    vegaEmbed('#vis', spec, {{"actions": false}}).catch(console.error);
  </script>
</body>
</html>
"""

with open('index.html', 'w') as f:
    f.write(html_template)