'''

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
click_selection = alt.selection_point(fields=['abstracttitle'], on='click', toggle=True, empty='all')

zoom_transform = alt.selection_interval(bind='scales', name="zoom_transform")

essential_cols = ['tsne_x', 'tsne_y', 'cluster_label', 'abstracttitle', 'sessiontitle', 'sessiontype', 'growth_val', 'year', 'abstract']
df_final = df_final[essential_cols].copy()
df_final['tsne_x'] = df_final['tsne_x'].round(4)
df_final['tsne_y'] = df_final['tsne_y'].round(4)
df_final['abstract'] = df_final['abstract'].str.replace(r'\s+', ' ', regex=True).str.strip()

final_chart = (
    alt.Chart(df_final)
    .mark_circle(size=25)
    .encode(
        x=alt.X('tsne_x:Q', title='TSNE-x'),
        y=alt.Y('tsne_y:Q', title='TSNE-y'),
        color=alt.Color('growth_val:Q', 
                        title='Growth Scale',
                        scale=alt.Scale(
                            scheme='redblue', 
                            reverse=True, 
                            domain=[-25, 25],
                            clamp=True
                        ),
                        legend=alt.Legend(
                            orient='right',
                            titleFontSize=12,
                            labelFontSize=10,
                            gradientLength=200
                        )),
        opacity=alt.condition(
            (year_selection & session_selection & click_selection),
            alt.value(1.0),
            alt.value(0.05)
        )
    )
    .add_params(year_selection, session_selection, search_param, click_selection, zoom_transform)
    .transform_filter(alt.expr.test(alt.expr.regexp(search_param, 'i'), alt.datum.abstracttitle))
    .properties(width=1000, height=800)
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
    :root { --primary-color: #2c3e50; --accent-color: #3498db; --bg-color: #f0f2f5; --card-bg: #ffffff; --text-muted: #94a3b8; }
    body { background-color: var(--bg-color); font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
    .header-bar { width: 100%; background-color: var(--primary-color); color: white; padding: 1.5rem 0; text-align: center; }
    .tab-container { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; margin-top: 20px; }
    .tab-button { padding: 10px 25px; border: none; background: #e2e8f0; border-radius: 25px; cursor: pointer; font-weight: 600; font-family: inherit; transition: all 0.3s ease; color: var(--primary-color); }
    .tab-button.active { background: var(--accent-color); color: white; box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3); }
    .app-wrapper { padding: 10px 30px 30px 30px; display: flex; flex-direction: column; align-items: center; }
    #vis-container { position: relative; background: var(--card-bg); padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); width: fit-content; max-width: 95vw; }
    #instruction-container { display: none; background: var(--card-bg); padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); max-width: 800px; }
    .zoom-controls { position: absolute; bottom: 80px; right: 80px; display: flex; flex-direction: column; gap: 8px; z-index: 1000; }
    .zoom-btn { width: 44px; height: 44px; border-radius: 50%; border: 1px solid #e2e8f0; background: white; color: var(--primary-color); font-size: 20px; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
    .zoom-btn:hover { background: var(--accent-color); color: white; border-color: var(--accent-color); transform: scale(1.05); }
    #sidebar { width: 450px; height: 100vh; background: white; position: fixed; right: -500px; top: 0; box-shadow: -10px 0 30px rgba(0,0,0,0.1); transition: right 0.4s ease; padding: 40px 30px 100px 30px; overflow-y: auto; z-index: 2000; box-sizing: border-box; }
    #sidebar.open { right: 0; }
    .close-btn { position: absolute; top: 20px; right: 25px; font-size: 24px; cursor: pointer; color: #bdc3c7; }
    .field-label { font-weight: 800; color: var(--text-muted); text-transform: uppercase; font-size: 0.65rem; display: block; margin-bottom: 4px; letter-spacing: 1px; }
    .field-value { color: var(--primary-color); font-size: 0.95rem; line-height: 1.4; font-weight: 500; margin-bottom: 25px; }
    .abstract-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #edf2f7; font-size: 0.92rem; line-height: 1.6; white-space: pre-wrap; color: #4a5568; }
    .info-row { display: flex; gap: 30px; margin-bottom: 20px; }
    .info-block { flex: 1; }
    .instruction-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 20px; }
    .instruction-item h3 { color: var(--accent-color); margin-bottom: 10px; font-size: 1.1rem; }
    .instruction-item p { color: #4a5568; line-height: 1.6; font-size: 0.95rem; }
    .vega-bindings { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #f1f5f9; display: flex; flex-wrap: wrap; gap: 20px; }
    .vega-bind { display: inline-flex !important; align-items: center !important; font-size: 11px !important; margin: 0 !important; }
    .vega-bind-name { margin-right: 10px !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; font-size: 10px !important; color: var(--text-muted) !important; font-weight: 800 !important; }
    .vega-bind select, .vega-bind input { appearance: none !important; background-color: #fff !important; border: 1px solid #e2e8f0 !important; border-radius: 20px !important; padding: 6px 16px !important; outline: none !important; color: var(--primary-color) !important; font-family: inherit !important; font-size: 12px !important; }
  </style>
</head>
<body>
  <div class="header-bar"><h1 style="margin:0; font-weight: 700; font-size: 1.8rem;">t-SNE Exploration Map</h1></div>
  <div class="app-wrapper">
    <div class="tab-container">
        <button class="tab-button active" onclick="showTab('map')">Interactive Map</button>
        <button class="tab-button" onclick="showTab('help')">How to Use</button>
    </div>
    <div id="vis-container">
      <div id="vis"></div>
      <div class="zoom-controls">
        <button class="zoom-btn" onclick="zoom(1.4)">+</button>
        <button class="zoom-btn" onclick="zoom(0.7)">−</button>
        <button class="zoom-btn" onclick="resetZoom()" style="font-size: 9px; font-weight: 800;">RESET</button>
      </div>
    </div>
    <div id="instruction-container">
      <h2 style="color: var(--primary-color); margin-top: 0;">Map Navigation Guide</h2>
      <p style="color: #64748b;">Follow these steps to explore the data clusters and research trends.</p>
      <div class="instruction-grid">
        <div class="instruction-item">
            <h3>Filtering & Search</h3>
            <p>Use the dropdown menus at the bottom of figure to filter abstracts by year or session type. Keyword search bar returns abstracts whose titles contain the given search query.</p>
        </div>
        <div class="instruction-item">
            <h3>Clicking for Details</h3>
            <p>Click any datapoint (abstract) in the map to open a detailed sidebar. The sidebar provides the abstract title in bold, as well as the abstract's cluster, the absolute growth of that specific cluster from 2019 to 2023, the year the abstract was published, the session title, session type, and the content of the abstract itself.</p>
        </div>
        <div class="instruction-item">
            <h3>Growth Scale</h3>
            <p>The color scale indicates the absolute growth of clusters (in number of abstracts) over the 2019 - 2023 period. A red cluster indicates a growth in the number of abstracts, while a blue cluster indicates a decline in the number of abstracts.</p>
        </div>
        <div class="instruction-item">
            <h3>Zoom & Pan</h3>
            <p>Use the "+", "-", and "Reset" buttons to zoom in on specific areas of the map. Click and drag the map to pan.</p>
        </div>
      </div>
      <button class="tab-button active" style="margin-top: 30px;" onclick="showTab('map')">Back to Map</button>
    </div>
  </div>
  <div id="sidebar">
    <span class="close-btn" onclick="document.getElementById('sidebar').classList.remove('open')">✕</span>
    <div id="sidebar-data"></div>
  </div>
  <script>
    let view;
    function showTab(tabName) {
        const mapTab = document.getElementById('vis-container');
        const helpTab = document.getElementById('instruction-container');
        const buttons = document.querySelectorAll('.tab-button');
        if (tabName === 'map') {
            mapTab.style.display = 'block';
            helpTab.style.display = 'none';
            buttons[0].classList.add('active');
            buttons[1].classList.remove('active');
        } else {
            mapTab.style.display = 'none';
            helpTab.style.display = 'block';
            buttons[1].classList.add('active');
            buttons[0].classList.remove('active');
        }
    }

    function zoom(factor) {
      if (!view) return;
      const xDomain = view.signal('zoom_transform_tsne_x') || view.scale('x').domain();
      const yDomain = view.signal('zoom_transform_tsne_y') || view.scale('y').domain();
      
      const xMid = (xDomain[0] + xDomain[1]) / 2;
      const yMid = (yDomain[0] + yDomain[1]) / 2;
      const xRange = (xDomain[1] - xDomain[0]) / factor;
      const yRange = (yDomain[1] - yDomain[0]) / factor;

      view.signal('zoom_transform_tsne_x', [xMid - xRange/2, xMid + xRange/2]);
      view.signal('zoom_transform_tsne_y', [yMid - yRange/2, yMid + yRange/2]);
      view.runAsync();
    }

    function resetZoom() {
      if (!view) return;
      view.signal('zoom_transform_tsne_x', null);
      view.signal('zoom_transform_tsne_y', null);
      view.runAsync();
    }

    const spec = REPLACE_ME_WITH_JSON;
    vegaEmbed('#vis', spec, { "actions": false }).then(result => {
        view = result.view;
        view.addEventListener('click', (event, item) => {
            if (item && item.mark && item.mark.marktype === 'symbol' && item.datum) {
                const d = item.datum;
                const arrow = d.growth_val >= 0 ? '↑' : '↓';
                document.getElementById('sidebar-data').innerHTML = `
                    <h2 style="color:var(--primary-color); font-size: 1.35rem; margin-top: 0; margin-bottom: 25px; line-height: 1.4; font-weight: 800;">` + (d.abstracttitle || "Untitled") + `</h2>
                    <div class="info-row">
                      <div class="info-block"><span class="field-label">Cluster Name</span><div class="field-value" style="margin-bottom:0;">${d.cluster_label}</div></div>
                    </div>
                    <div class="info-row">
                        <div class="info-block"><span class="field-label">Cluster Growth</span><div class="field-value" style="margin-bottom:0;">${arrow} ${Math.abs(d.growth_val)} abstracts</div></div>
                        <div class="info-block"><span class="field-label">Year</span><div class="field-value" style="margin-bottom:0;">${d.year}</div></div>
                    </div>
                    <div class="info-row">
                        <div class="info-block"><span class="field-label">Session Title</span><div class="field-value" style="margin-bottom:0;">${d.sessiontitle || "N/A"}</div></div>
                        <div class="info-block"><span class="field-label">Session Type</span><div class="field-value" style="margin-bottom:0;">${d.sessiontype || "N/A"}</div></div>
                    </div>
                    <div style="margin-top: 30px;">
                      <span class="field-label">Abstract Content</span><div class="abstract-box">${d.abstract}</div>
                    </div>
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
    '''

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
click_selection = alt.selection_point(fields=['abstracttitle'], on='click', toggle=True, empty='all')

zoom_transform = alt.selection_interval(bind='scales', name="zoom_transform")

essential_cols = ['tsne_x', 'tsne_y', 'cluster_label', 'abstracttitle', 'sessiontitle', 'sessiontype', 'growth_val', 'year', 'abstract']
df_final = df_final[essential_cols].copy()
df_final['tsne_x'] = df_final['tsne_x'].round(4)
df_final['tsne_y'] = df_final['tsne_y'].round(4)
df_final['abstract'] = df_final['abstract'].str.replace(r'\s+', ' ', regex=True).str.strip()

final_chart = (
    alt.Chart(df_final)
    .mark_circle(size=25)
    .encode(
        x=alt.X('tsne_x:Q', title='TSNE-x'),
        y=alt.Y('tsne_y:Q', title='TSNE-y'),
        color=alt.Color('growth_val:Q', 
                        title='Growth Scale',
                        scale=alt.Scale(
                            scheme='redblue', 
                            reverse=True, 
                            domain=[-25, 25],
                            clamp=True
                        ),
                        legend=alt.Legend(
                            orient='right',
                            titleFontSize=12,
                            labelFontSize=10,
                            gradientLength=200
                        )),
        opacity=alt.condition(
            (year_selection & session_selection & click_selection),
            alt.value(1.0),
            alt.value(0.05)
        )
    )
    .add_params(year_selection, session_selection, search_param, click_selection, zoom_transform)
    .transform_filter(alt.expr.test(alt.expr.regexp(search_param, 'i'), alt.datum.abstracttitle))
    .properties(width=1000, height=800)
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
    :root { --primary-color: #2c3e50; --accent-color: #3498db; --bg-color: #f0f2f5; --card-bg: #ffffff; --text-muted: #94a3b8; }
    body { background-color: var(--bg-color); font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
    .header-bar { width: 100%; background-color: var(--primary-color); color: white; padding: 1.5rem 0; text-align: center; }
    .tab-container { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; margin-top: 20px; }
    .tab-button { padding: 10px 25px; border: none; background: #e2e8f0; border-radius: 25px; cursor: pointer; font-weight: 600; font-family: inherit; transition: all 0.3s ease; color: var(--primary-color); }
    .tab-button.active { background: var(--accent-color); color: white; box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3); }
    .app-wrapper { padding: 10px 30px 30px 30px; display: flex; flex-direction: column; align-items: center; }
    #vis-container { position: relative; background: var(--card-bg); padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); width: fit-content; max-width: 95vw; }
    #instruction-container { display: none; background: var(--card-bg); padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); max-width: 800px; }
    .zoom-controls { position: absolute; bottom: 80px; right: 80px; display: flex; flex-direction: column; gap: 8px; z-index: 1000; }
    .zoom-btn { width: 44px; height: 44px; border-radius: 50%; border: 1px solid #e2e8f0; background: white; color: var(--primary-color); font-size: 20px; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
    .zoom-btn:hover { background: var(--accent-color); color: white; border-color: var(--accent-color); transform: scale(1.05); }
    #sidebar { width: 450px; height: 100vh; background: white; position: fixed; right: -500px; top: 0; box-shadow: -10px 0 30px rgba(0,0,0,0.1); transition: right 0.4s ease; padding: 40px 30px 100px 30px; overflow-y: auto; z-index: 2000; box-sizing: border-box; }
    #sidebar.open { right: 0; }
    .close-btn { position: absolute; top: 20px; right: 25px; font-size: 24px; cursor: pointer; color: #bdc3c7; }
    .field-label { font-weight: 800; color: var(--text-muted); text-transform: uppercase; font-size: 0.65rem; display: block; margin-bottom: 4px; letter-spacing: 1px; }
    .field-value { color: var(--primary-color); font-size: 0.95rem; line-height: 1.4; font-weight: 500; margin-bottom: 25px; }
    .abstract-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #edf2f7; font-size: 0.92rem; line-height: 1.6; white-space: pre-wrap; color: #4a5568; }
    .info-row { display: flex; gap: 30px; margin-bottom: 20px; }
    .info-block { flex: 1; }
    .instruction-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 20px; }
    .instruction-item h3 { color: var(--accent-color); margin-bottom: 10px; font-size: 1.1rem; }
    .instruction-item p { color: #4a5568; line-height: 1.6; font-size: 0.95rem; }
    .vega-bindings { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #f1f5f9; display: flex; flex-wrap: wrap; gap: 20px; }
    .vega-bind { display: inline-flex !important; align-items: center !important; font-size: 11px !important; margin: 0 !important; }
    .vega-bind-name { margin-right: 10px !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; font-size: 10px !important; color: var(--text-muted) !important; font-weight: 800 !important; }
    .vega-bind select, .vega-bind input { appearance: none !important; background-color: #fff !important; border: 1px solid #e2e8f0 !important; border-radius: 20px !important; padding: 6px 16px !important; outline: none !important; color: var(--primary-color) !important; font-family: inherit !important; font-size: 12px !important; }
  </style>
</head>
<body>
  <div class="header-bar"><h1 style="margin:0; font-weight: 700; font-size: 1.8rem;">t-SNE Visualization</h1></div>
  <div class="app-wrapper">
    <div class="tab-container">
        <button class="tab-button active" onclick="showTab('map')">Interactive Map</button>
        <button class="tab-button" onclick="showTab('help')">How to Use</button>
    </div>
    <div id="vis-container">
      <div id="vis"></div>
      <div class="zoom-controls">
        <button class="zoom-btn" onclick="animateZoom(1.5)">+</button>
        <button class="zoom-btn" onclick="animateZoom(0.66)">−</button>
        <button class="zoom-btn" onclick="resetZoom()" style="font-size: 9px; font-weight: 800;">RESET</button>
      </div>
    </div>
    <div id="instruction-container">
      <h2 style="color: var(--primary-color); margin-top: 0;">Navigation Guide</h2>
      <div class="instruction-grid">
        <div class="instruction-item">
            <h3>Filtering & Search</h3>
            <p>Use the dropdown menus at the bottom of figure to filter abstracts by year or session type. Keyword search bar returns abstracts whose titles contain the given search query.</p>
        </div>
        <div class="instruction-item">
            <h3>Clicking for Details</h3>
            <p>Click any datapoint (abstract) in the map to open a detailed sidebar. The sidebar provides the abstract title in bold, as well as the abstract's cluster, the absolute growth of that specific cluster from 2019 to 2023, the year the abstract was published, the session title, session type, and the content of the abstract itself.</p>
        </div>
        <div class="instruction-item">
            <h3>Growth Scale</h3>
            <p>The color scale indicates the absolute growth of clusters (in number of abstracts) over the 2019 - 2023 period. A red cluster indicates a growth in the number of abstracts, while a blue cluster indicates a decline in the number of abstracts.</p>
        </div>
        <div class="instruction-item">
            <h3>Zoom & Pan</h3>
            <p>Use the "+", "-", and "Reset" buttons to zoom in on specific areas of the map. Click and drag the map to pan.</p>
        </div>
      </div>
      <button class="tab-button active" style="margin-top: 30px;" onclick="showTab('map')">Back to Map</button>
    </div>
  </div>
  <div id="sidebar">
    <span class="close-btn" onclick="document.getElementById('sidebar').classList.remove('open')">✕</span>
    <div id="sidebar-data"></div>
  </div>
  <script>
    let view;
    function showTab(tabName) {
        const mapTab = document.getElementById('vis-container');
        const helpTab = document.getElementById('instruction-container');
        const buttons = document.querySelectorAll('.tab-button');
        if (tabName === 'map') {
            mapTab.style.display = 'block';
            helpTab.style.display = 'none';
            buttons[0].classList.add('active');
            buttons[1].classList.remove('active');
        } else {
            mapTab.style.display = 'none';
            helpTab.style.display = 'block';
            buttons[1].classList.add('active');
            buttons[0].classList.remove('active');
        }
    }

    function animateZoom(factor) {
      if (!view) return;
      
      const xStart = view.signal('zoom_transform_tsne_x') || view.scale('x').domain();
      const yStart = view.signal('zoom_transform_tsne_y') || view.scale('y').domain();
      
      const xMid = (xStart[0] + xStart[1]) / 2;
      const yMid = (yStart[0] + yStart[1]) / 2;
      const xRangeTarget = (xStart[1] - xStart[0]) / factor;
      const yRangeTarget = (yStart[1] - yStart[0]) / factor;
      
      const xTarget = [xMid - xRangeTarget/2, xMid + xRangeTarget/2];
      const yTarget = [yMid - yRangeTarget/2, yMid + yRangeTarget/2];
      
      const duration = 250;
      const start = performance.now();
      
      function step(now) {
        const progress = Math.min(1, (now - start) / duration);
        const ease = progress * (2 - progress); // Simple ease-out
        
        const curX = [
          xStart[0] + (xTarget[0] - xStart[0]) * ease,
          xStart[1] + (xTarget[1] - xStart[1]) * ease
        ];
        const curY = [
          yStart[0] + (yTarget[0] - yStart[0]) * ease,
          yStart[1] + (yTarget[1] - yStart[1]) * ease
        ];
        
        view.signal('zoom_transform_tsne_x', curX);
        view.signal('zoom_transform_tsne_y', curY);
        view.runAsync();
        
        if (progress < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    function resetZoom() {
      if (!view) return;
      view.signal('zoom_transform_tsne_x', null);
      view.signal('zoom_transform_tsne_y', null);
      view.runAsync();
    }

    const spec = REPLACE_ME_WITH_JSON;
    vegaEmbed('#vis', spec, { "actions": false }).then(result => {
        view = result.view;
        view.addEventListener('click', (event, item) => {
            if (item && item.mark && item.mark.marktype === 'symbol' && item.datum) {
                const d = item.datum;
                const arrow = d.growth_val >= 0 ? '↑' : '↓';
                document.getElementById('sidebar-data').innerHTML = `
                    <h2 style="color:var(--primary-color); font-size: 1.35rem; margin-top: 0; margin-bottom: 25px; line-height: 1.4; font-weight: 800;">` + (d.abstracttitle || "Untitled") + `</h2>
                    <div class="info-row">
                      <div class="info-block"><span class="field-label">Cluster Name</span><div class="field-value" style="margin-bottom:0;">${d.cluster_label}</div></div>
                    </div>
                    <div class="info-row">
                        <div class="info-block"><span class="field-label">Cluster Growth</span><div class="field-value" style="margin-bottom:0;">${arrow} ${Math.abs(d.growth_val)} abstracts</div></div>
                        <div class="info-block"><span class="field-label">Year</span><div class="field-value" style="margin-bottom:0;">${d.year}</div></div>
                    </div>
                    <div class="info-row">
                        <div class="info-block"><span class="field-label">Session Title</span><div class="field-value" style="margin-bottom:0;">${d.sessiontitle || "N/A"}</div></div>
                        <div class="info-block"><span class="field-label">Session Type</span><div class="field-value" style="margin-bottom:0;">${d.sessiontype || "N/A"}</div></div>
                    </div>
                    <div style="margin-top: 30px;">
                      <span class="field-label">Abstract Content</span><div class="abstract-box">${d.abstract}</div>
                    </div>
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

