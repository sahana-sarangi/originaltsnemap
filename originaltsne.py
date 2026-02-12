import pandas as pd
import numpy as np
import altair as alt
import os

# No more Drive mounting!
# This assumes your CSVs are in the same folder as your script
DATA_PATH = '.' 
alt.data_transformers.disable_max_rows()

def add_leading_zeroes(x):
    if pd.isna(x): x = 0
    return "{:02d}".format(int(x))

# Load data locally from the repo folder
data = pd.read_csv("updated_astro_dataset60.csv", index_col=0, low_memory=False)
# ... rest of your processing code ...

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
    df['Index'] = np.arange(1, df.shape[0] + 1)
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

selection = alt.selection_point(fields=['Cluster_Num'], bind='legend')

final_chart = (
    alt.Chart(df)
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
            alt.Tooltip('session_title:N', title='SessionTitle'),
            alt.Tooltip('TopicName:N', title='Cluster Name'),
            alt.Tooltip('session_type:N', title='SessionType'),
            alt.Tooltip('AbstractTitle:N', title='AbstractTitle'),
            alt.Tooltip('Abstract:N', title='Abstract')
        ]
    )
    .properties(
        title='TSNE Plot: Oncology Abstracts',
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

final_chart