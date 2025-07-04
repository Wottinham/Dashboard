# ----------------------
# Kepler map: country-level arcs with color ramp
# ----------------------
# compute centroids as before…
origin_centroids = (
    df.groupby("Origin Country Name")[["Origin Lat", "Origin Lon"]]
      .mean()
      .reset_index()
)
dest_centroids = (
    df.groupby("Destination Country Name")[["Dest Lat", "Dest Lon"]]
      .mean()
      .reset_index()
)

country_agg = (
    df.groupby(["Origin Country Name", "Destination Country Name"], as_index=False)
      .agg({"Passengers": "sum", "Passengers after policy": "sum"})
)
country_agg["traffic_change"] = (
    country_agg["Passengers after policy"] / country_agg["Passengers"] - 1
) * 100

country_agg = (
    country_agg
    .merge(origin_centroids, on="Origin Country Name", how="left")
    .merge(dest_centroids,   on="Destination Country Name", how="left")
)

kepler_df = country_agg.dropna(subset=["Origin Lat", "Origin Lon", "Dest Lat", "Dest Lon"])
if not kepler_df.empty:
    st.subheader("🌍 Air Traffic Change Map (Country-Level)")

    kepler_df = kepler_df.rename(columns={
        "Origin Lat": "origin_lat",
        "Origin Lon": "origin_lng",
        "Dest Lat":   "dest_lat",
        "Dest Lon":   "dest_lng",
        "Origin Country Name":      "origin_country",
        "Destination Country Name": "dest_country",
    })

    kepler_config = {
      "version": "v1",
      "config": {
        "visState": {
          "layers": [{
            "id": "country-traffic-arcs",
            "type": "arc",
            "config": {
              "dataId": "Country Traffic Change",
              "label": "Traffic Δ Arcs",
              "columns": {
                "lat0": "origin_lat",
                "lng0": "origin_lng",
                "lat1": "dest_lat",
                "lng1": "dest_lng"
              },
              "isVisible": True,
              "visConfig": {
                "opacity": 0.9,
                "thickness": 4,
                "colorField": {"name": "traffic_change", "type": "real"},
                "colorScale": "quantile",
                "colorRange": {
                  "name": "Traffic Change",
                  "type": "sequential",
                  "category": "Uber",
                  "colors": ["#2ca25f", "#fee08b", "#f03b20"]
                }
              }
            }
          }],
          "interactionConfig": {
            "tooltip": {
              "fieldsToShow": {
                "Country Traffic Change": [
                  {"name": "origin_country"},
                  {"name": "dest_country"},
                  {"name": "traffic_change"}
                ]
              }
            }
          },
          "layerBlending": "normal"
        }
      }
    }

    m = KeplerGl(config=kepler_config, height=800)  # ↑ bigger map height
    m.add_data(data=kepler_df, name="Country Traffic Change")
    keplergl_static(m)
