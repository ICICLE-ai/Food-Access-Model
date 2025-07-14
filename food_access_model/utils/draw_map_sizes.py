import folium
from shapely import wkt
from pyproj import Transformer

def generate_map_html(households, stores):
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    fmap = folium.Map(location=[39.9388, -82.9724], zoom_start=13)

    for household in households[:100000]:
        try:
            geom = wkt.loads(household["Geometry"])
            centroid = geom.centroid
            lat, lon = transformer.transform(centroid.x, centroid.y)[::-1]

            popup_html = "<b>Household Info</b><br/>" + "<br/>".join(
                f"{k}: {household.get(k, 'N/A')}" for k in [
                    "Income", "Household Size", "Vehicles", "Number of Workers",
                    "Stores within 1 Mile", "Closest Store (Miles)", "Transit time", "Food Access Score"
                ]
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(fmap)
        except Exception:
            continue

    for store in stores:
        try:
            geom = wkt.loads(store[1])
            centroid = geom.centroid
            lat, lon = transformer.transform(centroid.x, centroid.y)[::-1]

            popup_html = f"Name: {store[2]}<br/>Type: {store[0]}"

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="blue",
                fill=True,
                fill_opacity=0.5,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(fmap)
        except Exception:
            continue

    html = fmap._repr_html_()

    # JavaScript to attach moveend listener only once
    script = """
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        // Wait to ensure the map is fully initialized
        setTimeout(function () {
          const map = Object.values(window).find(obj =>
            obj && typeof obj.getBounds === 'function' &&
            typeof obj.on === 'function' &&
            typeof obj.fire === 'function'
          );
          if (!map) {
            console.warn("Leaflet map instance not found in window.");
            return;
          }
          window._leaflet_map = map;

          // Add the moveend listener
          map.on("moveend", () => {
            const bounds = map.getBounds();
            const params = new URLSearchParams({
              north: bounds.getNorth(),
              south: bounds.getSouth(),
              east: bounds.getEast(),
              west: bounds.getWest()
            });

            fetch(`/api/agents-view?` + params.toString())
              .then(resp => resp.text())
              .then(html => {
                const wrapper = document.querySelector('#leaflet-map-container');
                if (wrapper) {
                  wrapper.innerHTML = html;
                }
              });
          });

          // 🔥 Trigger the fetch once on first load
          map.fire("moveend");

        }, 1000); // wait 1 second after DOM is ready
      });
    </script>
    """


    return html + script
