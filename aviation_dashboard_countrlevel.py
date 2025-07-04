# ─────── Kepler map (replace your old map section with this) ───────

from keplergl import KeplerGl

# 1) build country centroids
coords = (
    df[['Origin Country Name','Origin Lat','Origin Lon']]
      .rename(columns={
          'Origin Country Name':'Country',
          'Origin Lat':'Lat','Origin Lon':'Lon'
      })
    .append(
      df[['Destination Country Name','Dest Lat','Dest Lon']]
        .rename(columns={
          'Destination Country Name':'Country',
          'Dest Lat':'Lat','Dest Lon':'Lon'
        })
    )
)
centroids = (
    coords
    .dropna(subset=['Lat','Lon'])
    .groupby('Country', as_index=False)[['Lat','Lon']]
    .mean()
)

# 2) aggregate inbound+outbound per unordered pair
pair_agg = (
    df.groupby(
        [df['Origin Country Name'].where(
            df['Origin Country Name'] < df['Destination Country Name'],
            df['Destination Country Name']
        ).rename('A'),
        df['Destination Country Name'].where(
            df['Origin Country Name'] < df['Destination Country Name'],
            df['Origin Country Name']
        ).rename('B')],
        as_index=False
    )
    .agg({
        'Passengers':'sum',
        'Passengers after policy':'sum'
    })
)
pair_agg['Traffic Δ (%)'] = (
    pair_agg['Passengers after policy'] / pair_agg['Passengers'] - 1
) * 100

# 3) merge centroids onto A & B
pair_agg = (
    pair_agg
    .merge(centroids.rename(columns={'Country':'A','Lat':'A Lat','Lon':'A Lon'}),
           on='A', how='left')
    .merge(centroids.rename(columns={'Country':'B','Lat':'B Lat','Lon':'B Lon'}),
           on='B', how='left')
)

# 4) build Kepler config with an arc layer
kepler_config = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [{
        "id": "arc_layer",
        "type": "arc",
        "config": {
          "dataId": "pairs",
          "label": "Traffic Δ (%)",
          "color": [255, 153, 31],
          "columns": {
            "lat0": "A Lat",
            "lng0": "A Lon",
            "lat1": "B Lat",
            "lng1": "B Lon"
          },
          "isVisible": True,
          "visConfig": {
            "thickness": 3,
            "opacity": 0.8,
            "colorRange": {
              "name": "Global Warming",
              "type": "sequential",
              "category": "Uber",
              "colors": ["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]
            },
            "sizeField": "Traffic Δ (%)",
            "sizeScale": 10
          }
        }
      }],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "pairs": ["A","B","Traffic Δ (%)"]
          },
          "enabled": True
        }
      }
    },
    "mapState": {
      "latitude": centroids['Lat'].mean(),
      "longitude": centroids['Lon'].mean(),
      "zoom": 2.2,
      "pitch": 30
    },
    "mapStyle": {}
  }
}

# 5) render
kepler_map = KeplerGl(
    height=800,
    data={"pairs": pair_agg},
    config=kepler_config
)
# embed in Streamlit
st.components.v1.html(kepler_map._repr_html_(), height=820)
