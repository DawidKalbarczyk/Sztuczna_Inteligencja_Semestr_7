# !pip install -q "osmnx==2.1.1" mapclassify
# !pip install geopandas
# !pip install shapely
# !pip install networkx
# !pip install requests
# !pip install IPython

import os, re, json, time, zipfile, warnings
import datetime as DT
from pathlib import Path
import numpy as NP
import pandas as PD
import geopandas as GPD
import shapely
import networkx
import osmnx
import requests
import matplotlib.pyplot as plt_plot
import matplotlib.colors as plt_colors
import IPython.display as display



#CONFIGURATION##########################
PD.set_option("display.max_columns", 20)
PD.set_option("display.max_rows", 20)
PD.set_option("display.width", 200)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#c9cdd2"
BLUE = plt_colors.LinearSegmentedColormap.from_list(
    "niebieski", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

#"This is a dictionary-like variable that stores the current 
# configuration settings. Many of the values control styling, 
# but others control various aspects of Matplotlib's behavior." ~ Docs
plt_plot.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.axisbelow": True,
})
#########################################

# Rozkłady jazdy GTF
GTFS = {
    "Autobus": "https://gtfs.ztp.krakow.pl/GTFS_KRK_A.zip",
    "Tram": "https://gtfs.ztp.krakow.pl/GTFS_KRK_T.zip",
    "Mobilis": "https://gtfs.ztp.krakow.pl/GTFS_KRK_M.zip"
}

# Generalny Pomiar Ruchu 2025 (GDDKiA): 
# odcinki pomiarowe dróg krajowych z wynikami, plik SHP w ZIP
# Wzięte z: https://www.gov.pl/web/gddkia-rzeszow/kto-gdzie-i-ile-jezdzil-gpr-juz-to-policzyl
GPR_SHP_URL = "https://www.gov.pl/attachment/7f8ca805-e384-4802-bf10-d36538e6ef84"

# Zapis danych do folderu (NIE COLAB)
FILE = Path(os.environ.get("L1_DATA", "L1_Dane_Transportowe"))
FILE.mkdir(parents=True, exist_ok=True)
DOWNLOADED = FILE / "pobrane"
DOWNLOADED.mkdir(exist_ok=True)
GRAPHS = FILE / "graphs"
GRAPHS.mkdir(exist_ok=True)

import gpt_functions as GPT

print('Folder z danymi: ', os.path.abspath(FILE))
print('\nFolder z pobranymi danymi: ', os.path.abspath(DOWNLOADED), '\n')

miejsca = PD.DataFrame({
    "nazwa":  ["Rynek Główny", "Kraków Główny (dworzec)", "Wawel", "AGH (budynek A-0)"],
    "lat":    [50.0617, 50.0677, 50.0540, 50.0647],     # szerokość geograficzna (stopnie)
    "lon":    [19.9372, 19.9474, 19.9354, 19.9234],     # długość geograficzna (stopnie)
    "rodzaj": ["plac", "dworzec", "zabytek", "uczelnia"],
})

display.display(miejsca)


#GeoPandas GeoDataFrame
punkty = GPD.GeoDataFrame(
    miejsca,
    geometry=GPD.points_from_xy(miejsca["lon"], miejsca["lat"]),  # x = lon, y = lat
    crs="EPSG:4326",
)

print("\nUkład:", punkty.crs)
display.display(punkty[["nazwa", "geometry"]])

punkty_m = punkty.to_crs("EPSG:2180")      # te same punkty w metrach
print("\nUkład po przeliczeniu:", punkty_m.crs)
display.display(punkty_m[["nazwa", "geometry"]])

ax = punkty_m.plot(color=COLORS[0], markersize=40, figsize=(6, 4))
for x, y, n in zip(punkty_m.geometry.x, punkty_m.geometry.y, punkty_m["nazwa"]):
    ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.ticklabel_format(useOffset=False, style="plain")
ax.xaxis.set_major_locator(plt_plot.MaxNLocator(4))
ax.set_xlabel("x [m] (EPSG:2180)")
ax.set_ylabel("y [m]")
ax.set_title("Cztery miejsca w Krakowie (EPSG:2180, metry)", loc="left");
ax.figure.tight_layout()
graph_file = GRAPHS / "punkty_m.png"
ax.figure.savefig(graph_file, dpi=300, bbox_inches="tight")


# Zadanie 1.1.
# To 0 zwraca po prostu wartości z wiersza dla geometry
rynek = punkty[punkty["nazwa"] == "Rynek Główny"].geometry.iloc[0]
dworzec = punkty[punkty["nazwa"] == "Kraków Główny (dworzec)"].geometry.iloc[0]
odleglosc_stopnie = rynek.distance(dworzec)
print(f"Odległość między Rynkiem Głównym i dworcem w stopniach: {odleglosc_stopnie:.2f}°")

rynek_metry = punkty_m[punkty_m["nazwa"] == "Rynek Główny"].geometry.iloc[0]
dworzec_metry = punkty_m[punkty_m["nazwa"] == "Kraków Główny (dworzec)"].geometry.iloc[0]
odleglosc_metry = rynek_metry.distance(dworzec_metry)
print(f"Odległość między Rynkiem Głównym i dworcem w metrach: {odleglosc_metry:.2f} m")


# 2 - Sieć drogowa z OSM
# tagi wzięte z https://taginfo.openstreetmap.org/tags
#Granica miasta Kraków

#CONFIG dla OSMnx
OVERPASS_SERVERS = ["https://overpass-api.de/api", "https://overpass.private.coffee/api",
                    "https://maps.mail.ru/osm/tools/overpass/api"]
osmnx.settings.use_cache = True
osmnx.settings.cache_folder = str(FILE / "cache_osm")
osmnx.settings.requests_timeout = 300
osmnx.settings.log_console = False

def fromOSM(function, *args, **kwargs):
    """Wywołuje funkcję OSMnx; gdy serwer nie odpowiada, próbuje kolejnego."""
    last = None
    for i, url in enumerate(OVERPASS_SERVERS):
        osmnx.settings.overpass_url = url
        osmnx.settings.overpass_rate_limit = (i == 0)   # tylko serwer główny ma limit zapytań
        try:
            return function(*args, **kwargs)
        except Exception as e:
            last = e
            print(f"Serwer {url} nie odpowiedział ({type(e).__name__}: {str(e)[:120]}). Próbuję następnego…")
            time.sleep(3)
    raise RuntimeError("Serwery OSM nie odpowiadają. Odczekaj kilka minut i uruchom komórkę ponownie.") from last
###################

city = "Kraków, Polska"
voivodeship = "województwo małopolskie, Polska"
shortcut = "krakow"
# Celowe nadpisanie, aby mieć pod ręką kod
punkty = GPD.GeoDataFrame(
    miejsca,
    geometry=GPD.points_from_xy(miejsca["lon"], miejsca["lat"]),  # x = lon, y = lat
    crs="EPSG:4326",
)
punkty_m = punkty.to_crs("EPSG:2180")

boundary = fromOSM(osmnx.geocode_to_gdf, city)
boundary_m = boundary.to_crs("EPSG:2180")
print("\n\nGranica miasta Kraków (EPSG:4326):")
display.display(boundary)
print(f"""\nGranica {boundary[['display_name']].display_name.iloc[0]} z bbox 
      [{boundary[['bbox_west']].bbox_west.iloc[0]}, 
      {boundary[['bbox_east']].bbox_east.iloc[0]}, 
      {boundary[['bbox_south']].bbox_south.iloc[0]}, 
      {boundary[['bbox_north']].bbox_north.iloc[0]}] 
      (EPSG:4326)""")

#Subplots tworzy pusty wykres gdzie potem można dodać elementy
#Plot tworzy od razu wykres z danymi
fig, ax = plt_plot.subplots(figsize=(5, 5))
boundary_m.plot(ax=ax, color=COLORS[1], edgecolor=COLORS[0])
ax.set_title('Granica miasta Kraków (EPSG:2180)', loc="left", fontsize=12)
ax.set_axis_off()
ax.text(0.99, 0.01, 'Dane: © autorzy OpenStreetMap', transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#52514e")
ax.set_xlabel("x [m] (EPSG:2180)")
ax.set_ylabel("y [m]")
graph_file = GRAPHS / "granica_krakow.png"
ax.figure.savefig(graph_file, dpi=300, bbox_inches="tight")

#Obliczenie powierzchni miasta w km²
pole_km2 = boundary_m.geometry.iloc[0].area / 1e6

#Pobranie danych z OSM
cache_osm = FILE / "cache_osm"
cache_osm_graph = cache_osm / "krakow_graph.graphml"

dataOSM = None
if not cache_osm_graph.is_file():
    print("\nTworzę graf OSM…")
    t0 = time.time()
    dataOSM = fromOSM(
        osmnx.graph_from_place, city, network_type="drive"
    )
    osmnx.save_graphml(dataOSM, filepath=cache_osm_graph)

    print(f"\nUtworzenie i zapis grafu zajęły {time.time() - t0:.1f} s")
else:
    print("\nWczytuję zapisany graf OSM…")
    dataOSM = osmnx.load_graphml(filepath=cache_osm_graph)

print(f"Liczba węzłów: {dataOSM.number_of_nodes()}")
print(f"Liczba krawędzi: {dataOSM.number_of_edges()}")
print(f"Układ współrzędnych: {dataOSM.graph['crs']}")
print(f"Data OSM", dataOSM.graph)

dataOSM_m = osmnx.project_graph(dataOSM, to_crs="EPSG:2180")
dataOSM_GPDS = osmnx.graph_to_gdfs(dataOSM_m)
dataOSM_GPDS_nodes = dataOSM_GPDS[0]
dataOSM_GPDS_edges = dataOSM_GPDS[1]
print(f"\nData OSM EPSG:2180 (metry).")
display.display(dataOSM_GPDS_nodes.keys())
display.display(dataOSM_GPDS_edges.keys())