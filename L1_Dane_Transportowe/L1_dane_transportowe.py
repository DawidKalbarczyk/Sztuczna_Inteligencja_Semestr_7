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

# 2.2
dataOSM_m = osmnx.project_graph(dataOSM, to_crs="EPSG:2180")
dataOSM_GPDS = osmnx.graph_to_gdfs(dataOSM_m)
dataOSM_GPDS_nodes = dataOSM_GPDS[0]
dataOSM_GPDS_edges = dataOSM_GPDS[1]
print(f"\nData OSM EPSG:2180 (metry).")
display.display(dataOSM_GPDS_nodes.keys())
display.display(dataOSM_GPDS_edges.keys())

dataOSM_GPDS_edges["class_made"] = \
    dataOSM_GPDS_edges["highway"]. \
        apply(lambda x: (x[0] if len(x) else None) if isinstance(x, (list, tuple)) else x)

display.display(dataOSM_GPDS_edges
                [["name", "highway", "class_made", "maxspeed", "lanes", "oneway", "length"]].head(10))

print(f"\nLiczba krawędzi w klasach: {dataOSM_GPDS_edges['class_made'].value_counts().head(10)}")

len_directed = dataOSM_GPDS_edges['length'].sum() / 1_000

dataOSM_directed = dataOSM_m
dataOSM_undirected = osmnx.convert.to_undirected(dataOSM_directed)
len_undirected = (osmnx.graph_to_gdfs(dataOSM_undirected, nodes=False))['length'].sum() / 1_000

print(f"\nSuma długości krawędzi grafu skierowanego [km]: {dataOSM_GPDS_edges['length'].sum() / 1000:.2f}")
print(f"Suma długości ulic (graf nieskierowany) [km]: {len_undirected:.2f}")
print(f"Stosunek: {len_directed / len_undirected:.3f}")

#Sprawdzenie gdzie nie ma wartości w kolumnie name i length
hasValue = dataOSM_GPDS_edges["name"].notna()
percent = dataOSM_GPDS_edges.loc[hasValue, "length"].sum() / dataOSM_GPDS_edges["length"].sum() * 100
print(f"\nNazwę ma {percent:.2f}% długości krawędzi")

example = dataOSM_GPDS_edges.assign(len_with_name=dataOSM_GPDS_edges["length"].where(hasValue, 0))
example_grouped = example.groupby("class_made")[["length", "len_with_name"]].sum()
example_grouped["percent_with_name"] = 100 * example_grouped["len_with_name"] / example_grouped["length"] 
display.display(example_grouped.sort_values("length", ascending=False).head(8).round(1))
#############################

#Stworzenie tabeli kompletności dla ograniczeń prędkości, długości i liczby pasów q
temp_data = dataOSM_GPDS_edges.copy()
temp_w_maxspeed_percent = temp_data.assign(dl_maxspeed=temp_data["length"].where(temp_data["maxspeed"].notna(), 0))
temp_w_lanes_percent = temp_w_maxspeed_percent.assign(dl_lanes=temp_data["length"].where(temp_data["lanes"].notna(), 0))

table = temp_w_lanes_percent.groupby("class_made")[["length", "dl_maxspeed", "dl_lanes"]].sum()
table["len_km"] = table["length"] / 1_000
table["maxspeed_percent"] = 100 * table["dl_maxspeed"] / table["length"]
table["lanes_percent"] = 100 * table["dl_lanes"] / table["length"]

completed = (table[["len_km", "maxspeed_percent", "lanes_percent"]].sort_values("len_km", ascending=False).round(1))

print(f"\n\nTabela kompletności atrybutów w klasach dróg (długość w km, procenty w %):")
display.display(completed.head(12))

## Wykres do powyższej tabeli kompletności
reversed_table = completed.head(8).iloc[::-1]
fig, ax = plt_plot.subplots(figsize=(10, 5))
y = NP.arange(len(reversed_table))
ax.barh(y + 0.2, reversed_table["maxspeed_percent"], height=0.4, color=COLORS[0], label="maxspeed")
ax.barh(y - 0.2, reversed_table["lanes_percent"], height=0.4, color=COLORS[1], label="lanes")
ax.set_yticks(y, [f"{k} ({d:,.0f} km)" for k, d in zip(reversed_table.index, reversed_table["len_km"])])
ax.set_xlim(0, 100)
ax.set_xlabel("% długości krawędzi z wypełnionym tagiem")
ax.set_title("Kompletność tagów OSM w klasach dróg", loc="left", fontsize=12)
ax.legend(loc="lower right", frameon=False, fontsize=10)
plt_plot.tight_layout()
ax.figure.savefig(GRAPHS / "kompletnosc_atrybutow.png", dpi=300, bbox_inches="tight")


#Mapa sieci według hierarchii klas dróg - Główne - Zbiorcze

GROUP = {
    "motorway": "Główne", "motorway_link": "Główne", "trunk": "Główne", "trunk_link": "Główne", "primary": "Główne", "primary_link": "Główne",
    "secondary": "Zbiorcze", "secondary_link": "Zbiorcze", "teritary": "Zbiorcze", "territary_link": "Zbiorcze"
}

dataOSM_GPDS_edges["group"] = dataOSM_GPDS_edges["class_made"].map(GROUP).fillna("Inne - Lokalne")

fig, ax = plt_plot.subplots(figsize=(10,10))

boundary = fromOSM(osmnx.geocode_to_gdf, city)
boundary_m = boundary.to_crs("EPSG:2180")

boundary_m.boundary.plot(ax=ax, color="#353431", linewidth=0.8)
style = {
    "Inne - Lokalne": (GRAY, 0.5),
    "Zbiorcze": (COLORS[1], 1),
    "Główne": (COLORS[0], 1.5)
}

for grupa, (color, width) in style.items():
    dataOSM_GPDS_edges[dataOSM_GPDS_edges["group"] == grupa].plot(
        ax=ax, color=color, linewidth=width, label=grupa
    )

ax.legend(title="Roads", loc="upper center", bbox_to_anchor=(1, 1), frameon=False)
ax.set_title("Mapa sieci według hierarchii klas dróg", loc="left", fontsize=12)
ax.set_axis_off()
ax.text(0.99, 0.01, "Źródło: OSM", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#52514e")
ax.figure.tight_layout()
graph_file = GRAPHS / "mapa_sieci_hierarchia_klasy.png"
ax.figure.savefig(graph_file, dpi=300, bbox_inches="tight")


####GPT FUNCTIONS
def pobierz(url, plik, min_bajtow=1000):
    """Pobiera plik z internetu tylko raz; przy kolejnym uruchomieniu używa kopii z dysku."""
    plik = Path(plik)
    if plik.exists() and plik.stat().st_size >= min_bajtow:
        print(f"Używam pobranego wcześniej pliku {plik.name} ({plik.stat().st_size / 1e6:.1f} MB)")
        return plik
    print(f"Pobieram {url}")
    ostatni = None
    for proba in range(3):
        try:
            with requests.get(url, stream=True, timeout=120,
                              headers={"User-Agent": "GeoAI-transport-lab (zajecia dydaktyczne)"}) as r:
                r.raise_for_status()
                czesc = plik.with_name(plik.name + ".czesc")
                with open(czesc, "wb") as f:
                    for kawalek in r.iter_content(chunk_size=1 << 20):
                        f.write(kawalek)
            czesc.replace(plik)
            print(f"Gotowe: {plik.name} ({plik.stat().st_size / 1e6:.1f} MB)")
            return plik
        except Exception as e:
            ostatni = e
            print(f"Próba {proba + 1} nieudana: {e}")
            time.sleep(5 * (proba + 1))
    raise RuntimeError(f"Nie udało się pobrać {url}. Sprawdź połączenie i spróbuj ponownie za kilka minut.") from ostatni

def wczytaj_gtfs(plik):
    """Wczytuje potrzebne pliki GTFS z archiwum ZIP do słownika tabel. Wszystkie kolumny jako tekst."""
    tabele = {}
    with zipfile.ZipFile(plik) as z:
        pliki = {Path(n).name: n for n in z.namelist()}
        for nazwa in ["agency", "feed_info", "stops", "routes", "trips", "calendar", "calendar_dates", "stop_times"]:
            if f"{nazwa}.txt" not in pliki:
                continue
            with z.open(pliki[f"{nazwa}.txt"]) as f:
                if nazwa == "stop_times":   # największy plik: wczytujemy tylko potrzebne kolumny
                    potrzebne = {"trip_id", "stop_id", "departure_time", "stop_sequence"}
                    df = PD.read_csv(f, dtype=str, encoding="utf-8-sig", usecols=lambda k: k.strip() in potrzebne)
                else:
                    df = PD.read_csv(f, dtype=str, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            if nazwa != "stop_times":
                df = df.apply(lambda s: s.str.strip())
            tabele[nazwa] = df
    return tabele
###########################################


pliki_gtfs = {kod: pobierz(url, DOWNLOADED / f"gtfs_{kod}.zip") for kod, url in GTFS.items()}
gtfs = {kod: wczytaj_gtfs(plik) for kod, plik in pliki_gtfs.items()}
print("\n\nWczytane rozkłady jazdy GTF (GTFS):")
for kod, t in gtfs.items():
    print(f"Rozkład {kod}:", ", ".join(f"{n} {len(df):,}" for n, df in t.items()))

trams = gtfs["Tram"]
print(f"Przystaniki: {t["stops"].head(5)}, Kursy: {t["trips"].head(5)}, Rozkłady: {t["stop_times"].head(5)}")
if ("feed_info" in t): 
    print(f"\nDodatkowa informacja o rozkładzie:")
    print(t["feed_info"])

lists = []
for code, tram in gtfs.items():
    stops = tram["stops"].copy()
    stops["source"] = code
    lists.append(stops)

stops = PD.concat(lists, ignore_index=True)

#Zamiana string na float
stops["stop_lat"] = stops["stop_lat"].astype(float)
stops["stop_lon"] = stops["stop_lon"].astype(float)

bus_stops = GPD.GeoDataFrame(stops, geometry = GPD.points_from_xy(
    stops["stop_lon"], stops["stop_lat"]), crs="EPSG:4326").to_crs("EPSG:2180")

print(f"\n{len(bus_stops)} przystanków tramwajowych; UKŁAD: {bus_stops.crs}")



DNI = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]          # nazwy kolumn w calendar.txt
DNI_PL = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
#### GPT FUNCTIONS
def uslugi_w_dniu(t, dzien):
    """Zbiór service_id, które kursują w danym dniu (dzien: datetime.date)."""
    data_txt = dzien.strftime("%Y%m%d")
    aktywne = set()
    if "calendar" in t:
        c = t["calendar"]
        warunek = (c["start_date"] <= data_txt) & (c["end_date"] >= data_txt) & (c[DNI[dzien.weekday()]] == "1")
        aktywne |= set(c.loc[warunek, "service_id"])
    if "calendar_dates" in t:
        cd = t["calendar_dates"]
        tego_dnia = cd[cd["date"] == data_txt]
        aktywne |= set(tego_dnia.loc[tego_dnia["exception_type"] == "1", "service_id"])
        aktywne -= set(tego_dnia.loc[tego_dnia["exception_type"] == "2", "service_id"])
    return aktywne


def swieta_pl(rok):
    """Dni ustawowo wolne od pracy w Polsce: stałe i zależne od Wielkanocy."""
    a, b, c = rok % 19, rok // 100, rok % 100            # algorytm Meeusa/Jonesa/Butchera (Wielkanoc)
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    miesiac, dzien = divmod(h + l - 7 * m + 114, 31)
    wielkanoc = DT.date(rok, miesiac, dzien + 1)
    stale = [(1, 1), (1, 6), (5, 1), (5, 3), (8, 15), (11, 1), (11, 11), (12, 24), (12, 25), (12, 26)]
    dni = {DT.date(rok, mm, dd) for mm, dd in stale}
    return dni | {wielkanoc + DT.timedelta(days=n) for n in (0, 1, 49, 60)}   # Wielkanoc, poniedziałek, Zielone Świątki, Boże Ciało


def wybierz_dzien(rozklady, dzien_tygodnia=2, od=None):
    """Najbliższy dzień tygodnia (2 = środa), który nie jest świętem i ma kursy we wszystkich rozkładach."""
    dzien = od or DT.date.today()
    for _ in range(400):
        if (dzien.weekday() == dzien_tygodnia and dzien not in swieta_pl(dzien.year)
                and all(uslugi_w_dniu(t, dzien) for t in rozklady.values())):
            return dzien
        dzien += DT.timedelta(days=1)
    raise ValueError("Nie znaleziono dnia z kursami – sprawdź daty w calendar.txt (czy rozkład nie wygasł?).")


DATA_ANALIZY = wybierz_dzien(gtfs)
print(f"\nAnalizujemy dzień: {DATA_ANALIZY.isoformat()} ({DNI_PL[DATA_ANALIZY.weekday()]}) – wszystkie rozkłady mają kursy tego dnia.")
for code, tram in gtfs.items():
    print(f"Rozkład {code}: {len(uslugi_w_dniu(tram, DATA_ANALIZY))} aktywnych kursów tego dnia")

    