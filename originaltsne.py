

import pandas as pd
import numpy as np
import altair as alt
import os

DATA_PATH = '.'
alt.data_transformers.disable_max_rows()

def clean_and_deduplicate(df):
    df.columns = [c.lower().replace("_", "").replace(" ", "").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

data = pd.read_csv(os.path.join(DATA_PATH, "updated_astro_dataset60.csv"), index_col=0, low_memory=False)
data = clean_and_deduplicate(data)
data = data.rename(columns={"years": "year", "abstracttitle": "abstracttitle"})

file_path_tsne = os.path.join(DATA_PATH, "updated_fine_tuned_tsne100.csv")
with open(file_path_tsne, encoding="utf8", errors='ignore') as temp_f:
    df_tsne = pd.read_csv(temp_f)
    df_tsne = clean_and_deduplicate(df_tsne)
    df_tsne = df_tsne.rename(columns={
        "x": "tsne_x", "y": "tsne_y",
        "title": "abstracttitle", "abstract": "abstract",
        "topic(postforced)": "topic_id"
    })

growth_data = pd.read_csv(os.path.join(DATA_PATH, "growth_df100.csv"))
growth_data = clean_and_deduplicate(growth_data)
growth_data = growth_data.rename(columns={
    "topic(postforced)": "topic_id",
    "presentcount(2023)": "count_2023",
    "pastcount(2019)": "count_2019"
})
growth_data['growth_val'] = growth_data['count_2023'] - growth_data['count_2019']

file_path_reindexed = os.path.join(DATA_PATH, "Document-specific-reindexed-with-gpt-names.csv")
reindexed_df = pd.read_csv(file_path_reindexed)
reindexed_df = clean_and_deduplicate(reindexed_df)
reindexed_df = reindexed_df.rename(columns={
    "title": "abstracttitle", 
    "reindexedclustername": "cluster_label"
})

df = pd.merge(df_tsne, data, on='abstracttitle', how='left', suffixes=('', '_drop'))
df = pd.merge(df, reindexed_df[['abstracttitle', 'cluster_label']], on='abstracttitle', how='left', suffixes=('', '_drop'))
df = pd.merge(df, growth_data[['topic_id', 'growth_val']], on='topic_id', how='left', suffixes=('', '_drop'))

df = df.loc[:, ~df.columns.duplicated()]
df = df.drop(columns=[c for c in df.columns if c.endswith('_drop')])
df['growth_val'] = df['growth_val'].fillna(0)
df['cluster_label'] = df['cluster_label'].fillna("Unknown Cluster")
df_final = df.drop_duplicates(subset=['abstracttitle']).copy()

year_col = 'year' if 'year' in df_final.columns else None
if year_col:
    years = sorted(df_final[year_col].dropna().unique().tolist())
    year_selection = alt.selection_point(fields=[year_col], bind=alt.binding_select(options=[None] + years, labels=['All'] + [str(y) for y in years], name='Year: '))
else:
    year_selection = alt.selection_point()

session_col = 'sessiontype' if 'sessiontype' in df_final.columns else None
if session_col:
    session_types = sorted(df_final[session_col].dropna().unique().tolist())
    session_selection = alt.selection_point(fields=[session_col], bind=alt.binding_select(options=[None] + session_types, labels=['All'] + session_types, name='Session Type: '))
else:
    session_selection = alt.selection_point()

search_param = alt.param(value='', bind=alt.binding(input='text', placeholder='Keyword Search', name='Search: '))
legend_selection = alt.selection_point(fields=['cluster_label'], bind='legend')
click_selection = alt.selection_point(fields=['abstracttitle'], on='click', toggle=True, empty='all')

final_chart = (
    alt.Chart(df_final)
    .mark_circle(size=25)
    .encode(
        x=alt.X('tsne_x:Q', title='TSNE-x'),
        y=alt.Y('tsne_y:Q', title='TSNE-y'),
        color=alt.Color('cluster_label:N', 
                        title='Cluster Name',
                        sort=alt.EncodingSortField(field='growth_val', order='descending'),
                        scale=alt.Scale(scheme='redblue', reverse=True, domainMid=0),
                        legend=alt.Legend(
                            orient='right', 
                            columns=1,
                            symbolLimit=0,
                            labelLimit=180, 
                            offset=20, 
                            labelFontSize=10
                        )),
        opacity=alt.condition(
            (year_selection & session_selection & legend_selection & click_selection),
            alt.value(1.0),
            alt.value(0.05)
        )
    )
    .add_params(year_selection, session_selection, legend_selection, search_param, click_selection)
    .transform_filter(alt.expr.test(alt.expr.regexp(search_param, 'i'), alt.datum.abstracttitle))
    .properties(width=1000, height=800) 
    .interactive()
)

html_template = """
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;800&display=swap" rel="stylesheet">
  <style>
    :root { --primary-color: #2c3e50; --bg-color: #f0f2f5; --card-bg: #ffffff; --text-muted: #94a3b8; }
    body { background-color: var(--bg-color); font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
    .header-bar { width: 100%; background-color: var(--primary-color); color: white; padding: 1.5rem 0; text-align: center; }
    .app-wrapper { padding: 30px; display: flex; justify-content: center; }
    #vis-container { 
        background: var(--card-bg); 
        padding: 40px; 
        border-radius: 12px; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        width: fit-content;
        max-width: 95vw;
        overflow: auto;
    }
    .vega-bindings {
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    .vega-bind { display: inline-flex !important; align-items: center !important; font-size: 11px !important; margin: 0 !important; }
    .vega-bind-name { margin-right: 10px !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; font-size: 10px !important; color: var(--text-muted) !important; font-weight: 800 !important; }
    .vega-bind select, .vega-bind input { 
        appearance: none !important; 
        background-color: #fff !important; 
        border: 1px solid #e2e8f0 !important; 
        border-radius: 20px !important; 
        padding: 6px 16px !important; 
        outline: none !important; 
        color: var(--primary-color) !important; 
        font-family: inherit !important; 
        font-size: 12px !important;
    }
    #sidebar { width: 450px; height: 100vh; background: white; position: fixed; right: -500px; top: 0; box-shadow: -10px 0 30px rgba(0,0,0,0.1); transition: right 0.4s ease; padding: 40px 30px 100px 30px; overflow-y: auto; z-index: 2000; box-sizing: border-box; }
    #sidebar.open { right: 0; }
    .close-btn { position: absolute; top: 20px; right: 25px; font-size: 24px; cursor: pointer; color: #bdc3c7; }
    .field-label { font-weight: 800; color: var(--text-muted); text-transform: uppercase; font-size: 0.65rem; display: block; margin-bottom: 4px; letter-spacing: 1px; }
    .field-value { color: var(--primary-color); font-size: 0.95rem; line-height: 1.4; font-weight: 500; margin-bottom: 25px; }
    .meta-row { display: flex; gap: 40px; margin-bottom: 25px; border-top: 1px solid #f1f5f9; border-bottom: 1px solid #f1f5f9; padding: 15px 0; }
    .abstract-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #edf2f7; font-size: 0.92rem; line-height: 1.6; white-space: pre-wrap; color: #4a5568; }
  </style>
</head>
<body>
  <div class="header-bar"><h1 style="margin:0; font-weight: 700; font-size: 1.8rem;">t-SNE Visualization</h1></div>
  <div class="app-wrapper">
    <div id="vis-container">
      <div id="vis"></div>
    </div>
  </div>
  <div id="sidebar">
    <span class="close-btn" onclick="document.getElementById('sidebar').classList.remove('open')">✕</span>
    <div id="sidebar-data"></div>
  </div>
  <script>
    const spec = REPLACE_ME_WITH_JSON;
    vegaEmbed('#vis', spec, { "actions": false }).then(result => {
        result.view.addEventListener('click', (event, item) => {
            if (item && item.mark && item.mark.marktype === 'symbol' && item.datum) {
                const d = item.datum;
                const growthArrow = d.growth_val >= 0 ? '↑' : '↓';
                document.getElementById('sidebar-data').innerHTML = `
                    <h2 style="color:var(--primary-color); font-size: 1.35rem; margin-top: 0; margin-bottom: 20px; line-height: 1.4; font-weight: 800;">` + (d.abstracttitle || "Untitled") + `</h2>
                    <span class="field-label">Cluster Name</span><div class="field-value">${d.cluster_label}</div>
                    <span class="field-label">Session Title</span><div class="field-value">${d.sessiontitle || "N/A"}</div>
                    <div class="meta-row">
                        <div><span class="field-label">Session Type</span><div class="field-value">${d.sessiontype || "N/A"}</div></div>
                        <div><span class="field-label">Year</span><div class="field-value">${d.year}</div></div>
                        <div><span class="field-label">Growth</span><div class="field-value">${growthArrow} ${Math.abs(d.growth_val)}</div></div>
                    </div>
                    <span class="field-label">Abstract Content</span><div class="abstract-box">${d.abstract}</div>
                `;
                document.getElementById('sidebar').classList.add('open');
            } else {
                document.getElementById('sidebar').classList.remove('open');
            }
        });
    });
  </script>
</body>
</html>
"""

with open('index.html', 'w') as f:
    f.write(html_template.replace("REPLACE_ME_WITH_JSON", final_chart.to_json()))