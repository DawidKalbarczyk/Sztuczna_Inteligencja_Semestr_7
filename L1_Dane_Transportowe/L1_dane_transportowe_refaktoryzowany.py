import datetime as dt
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import osmnx as ox
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display


# =============================================================================
# USTAWIENIA
# =============================================================================

# Obszar analizy i źródła danych.
MIASTO = "Kraków, Polska"
WOJEWODZTWO = "województwo małopolskie, Polska"
SKROT = "krakow"
GTFS = {
    "A": "https://gtfs.ztp.krakow.pl/GTFS_KRK_A.zip",
    "T": "https://gtfs.ztp.krakow.pl/GTFS_KRK_T.zip",
    "M": "https://gtfs.ztp.krakow.pl/GTFS_KRK_M.zip",
}
GPR_SHP_URL = "https://www.gov.pl/attachment/7f8ca805-e384-4802-bf10-d36538e6ef84"

# Wszystkie wyniki trafiają do folderu skryptu, chyba że ustawisz L1_DATA.
FOLDER = Path(os.environ.get("L1_DATA", Path(__file__).resolve().parent))
POBRANE = FOLDER / "pobrane"
WYKRESY = FOLDER / "graphs"

# Zadanie dodatkowe: ustaw True po pobraniu paczki BDOT10k.
WYKONAJ_BDOT = False
PLIK_BDOT = FOLDER / "1261_GPKG.zip"

POBRANE.mkdir(parents=True, exist_ok=True)
WYKRESY.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.cache_folder = str(FOLDER / "cache_osm")
ox.settings.requests_timeout = 300
ox.settings.log_console = False

# Wygląd tabel, map i wykresów.
KOLORY = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SZARY = "#c9cdd2"
NIEBIESKI = LinearSegmentedColormap.from_list(
    "niebieski", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

DNI_TYGODNIA = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]


# =============================================================================
# FUNKCJE UŻYWANE W KILKU MIEJSCACH
# =============================================================================


def pobierz(url, plik):
    """Pobierz plik, jeśli nie ma go jeszcze na dysku."""
    if not plik.exists():
        with requests.get(url, stream=True, timeout=120) as odpowiedz:
            odpowiedz.raise_for_status()
            czesc = plik.with_suffix(".czesc")
            with czesc.open("wb") as zapis:
                for dane in odpowiedz.iter_content(chunk_size=1 << 20):
                    zapis.write(dane)
            czesc.replace(plik)
    return plik


def uslugi_w_dniu(tabele, dzien):
    """Zwróć service_id kursujące danego dnia, z uwzględnieniem wyjątków."""
    data = dzien.strftime("%Y%m%d")
    uslugi = set()

    # Zwykły kalendarz: okres ważności i dzień tygodnia.
    if "calendar" in tabele:
        kalendarz = tabele["calendar"]
        dzien_tygodnia = DNI_TYGODNIA[dzien.weekday()]
        pasuje = (
            (kalendarz["start_date"] <= data)
            & (kalendarz["end_date"] >= data)
            & (kalendarz[dzien_tygodnia] == "1")
        )
        uslugi = set(kalendarz.loc[pasuje, "service_id"])

    # Wyjątki: 1 dodaje usługę, 2 ją odwołuje.
    if "calendar_dates" in tabele:
        wyjatki = tabele["calendar_dates"]
        wyjatki = wyjatki[wyjatki["date"] == data]
        dodane = wyjatki.loc[wyjatki["exception_type"] == "1", "service_id"]
        odwolane = wyjatki.loc[wyjatki["exception_type"] == "2", "service_id"]
        uslugi.update(dodane)
        uslugi.difference_update(odwolane)

    return uslugi


def swieta_pl(rok):
    """Zwróć daty świąt stałych i ruchomych, jak w notebooku."""
    # Algorytm daty Wielkanocy. Litery oznaczają jego kolejne kroki.
    a, b, c = rok % 19, rok // 100, rok % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    miesiac, dzien = divmod(h + l - 7 * m + 114, 31)
    wielkanoc = dt.date(rok, miesiac, dzien + 1)

    # Pary (miesiąc, dzień) dla świąt o stałej dacie.
    daty_stalych_swiat = [
        (1, 1), (1, 6), (5, 1), (5, 3), (8, 15),
        (11, 1), (11, 11), (12, 24), (12, 25), (12, 26),
    ]
    swieta = {dt.date(rok, miesiac, dzien) for miesiac, dzien in daty_stalych_swiat}

    # Wielkanoc, poniedziałek wielkanocny, Zielone Świątki, Boże Ciało.
    swieta_ruchome = {
        wielkanoc + dt.timedelta(days=przesuniecie)
        for przesuniecie in (0, 1, 49, 60)
    }
    return swieta | swieta_ruchome


# =============================================================================
# 1. PUNKTY I UKŁADY WSPÓŁRZĘDNYCH
# =============================================================================

# W EPSG:4326 współrzędne są w stopniach, w EPSG:2180 są w metrach.
miejsca = pd.DataFrame({
    "nazwa": ["Rynek Główny", "Kraków Główny (dworzec)", "Wawel", "AGH (budynek A-0)"],
    "lat": [50.0617, 50.0677, 50.0540, 50.0647],
    "lon": [19.9372, 19.9474, 19.9354, 19.9234],
    "rodzaj": ["plac", "dworzec", "zabytek", "uczelnia"],
})
punkty = gpd.GeoDataFrame(
    miejsca,
    geometry=gpd.points_from_xy(miejsca["lon"], miejsca["lat"]),
    crs="EPSG:4326",
)
punkty_m = punkty.to_crs("EPSG:2180")

# Pierwsze dwa wiersze to Rynek i dworzec.
rynek, dworzec = punkty.geometry.iloc[:2]
rynek_m, dworzec_m = punkty_m.geometry.iloc[:2]
odleglosc_stopnie = rynek.distance(dworzec)
odleglosc_m = rynek_m.distance(dworzec_m)
print(f"Rynek – dworzec: {odleglosc_stopnie:.5f} stopnia, {odleglosc_m:.0f} m")

# Mapa czterech miejsc.
ax = punkty_m.plot(color=KOLORY[0], markersize=40, figsize=(6, 4))
for x, y, nazwa in zip(punkty_m.geometry.x, punkty_m.geometry.y, punkty_m["nazwa"]):
    ax.annotate(nazwa, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.ticklabel_format(useOffset=False, style="plain")
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xlabel("x [m] (EPSG:2180)")
ax.set_ylabel("y [m]")
ax.set_title("Cztery miejsca w Krakowie", loc="left")
ax.figure.tight_layout()
ax.figure.savefig(WYKRESY / "punkty_m.png", dpi=300, bbox_inches="tight")
plt.close(ax.figure)


# =============================================================================
# 2. GRANICA MIASTA I SIEĆ DROGOWA OSM
# =============================================================================

# 2.1. Granica miasta i powierzchnia.
granica = ox.geocode_to_gdf(MIASTO)
granica_m = granica.to_crs("EPSG:2180")
pole_km2 = granica_m.geometry.iloc[0].area / 1e6
print(f"Powierzchnia miasta: {pole_km2:.1f} km²")

fig, ax = plt.subplots(figsize=(5, 5))
granica_m.plot(ax=ax, color="#cde2fb", edgecolor="#256abf")
ax.set_title(f"Granica: {MIASTO}", loc="left")
ax.set_axis_off()
ax.text(
    0.99,
    0.01,
    "Dane: © autorzy OpenStreetMap",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
)
fig.savefig(WYKRESY / "granica_krakow.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2.2. Wczytanie grafu z dysku lub pobranie go z OSM.
plik_grafu_cache = FOLDER / "cache_osm" / "krakow_graph.graphml"
if plik_grafu_cache.exists():
    graf = ox.load_graphml(plik_grafu_cache)
else:
    graf = ox.graph_from_place(MIASTO, network_type="drive")
    ox.save_graphml(graf, plik_grafu_cache)

graf_metry = ox.project_graph(graf, to_crs="EPSG:2180")
wezly, krawedzie = ox.graph_to_gdfs(graf_metry)

# Brakujący tag oznacza brak danych. OSM może też zapisać klasę jako listę.
for kolumna in ["name", "maxspeed", "lanes", "oneway"]:
    if kolumna not in krawedzie:
        krawedzie[kolumna] = np.nan
krawedzie["klasa"] = krawedzie["highway"].apply(
    lambda x: (x[0] if x else None) if isinstance(x, (list, tuple)) else x
)

# 2.3. Długość sieci. Graf nieskierowany scala przeciwne kierunki ulicy.
dlugosc_krawedzi_km = krawedzie["length"].sum() / 1000
graf_nieskierowany = ox.convert.to_undirected(graf_metry)
dlugosc_ulic_km = ox.graph_to_gdfs(graf_nieskierowany, nodes=False)["length"].sum() / 1000
print(f"Krawędzie skierowane: {dlugosc_krawedzi_km:.0f} km; ulice: {dlugosc_ulic_km:.0f} km")

# 2.4. Kompletność nazw i tagów, ważona długością dróg.
ma_nazwe_ulicy = krawedzie["name"].notna()
dlugosc_drog_z_nazwa = krawedzie.loc[ma_nazwe_ulicy, "length"].sum()
udzial_drog_z_nazwa_proc = 100 * dlugosc_drog_z_nazwa / krawedzie["length"].sum()
tabela = krawedzie.assign(dl_z_nazwa=krawedzie["length"].where(ma_nazwe_ulicy, 0))
tabela = tabela.groupby("klasa")[["length", "dl_z_nazwa"]].sum()
tabela["nazwa_proc"] = 100 * tabela["dl_z_nazwa"] / tabela["length"]
tabela = tabela.sort_values("length", ascending=False).round(1)

tabela_tagow = krawedzie.assign(
    dl_maxspeed=krawedzie["length"].where(krawedzie["maxspeed"].notna(), 0),
    dl_lanes=krawedzie["length"].where(krawedzie["lanes"].notna(), 0),
)
tabela_tagow = tabela_tagow.groupby("klasa")[["length", "dl_maxspeed", "dl_lanes"]].sum()
tabela_tagow["dl_km"] = tabela_tagow["length"] / 1000
tabela_tagow["maxspeed_proc"] = 100 * tabela_tagow["dl_maxspeed"] / tabela_tagow["length"]
tabela_tagow["lanes_proc"] = 100 * tabela_tagow["dl_lanes"] / tabela_tagow["length"]
kompletnosc = tabela_tagow[["dl_km", "maxspeed_proc", "lanes_proc"]]
kompletnosc = kompletnosc.sort_values("dl_km", ascending=False).round(1)
display(kompletnosc.head(12))

najdluzsze_klasy = kompletnosc.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.5))
y = np.arange(len(najdluzsze_klasy))
ax.barh(y + 0.2, najdluzsze_klasy["maxspeed_proc"], height=0.38, color=KOLORY[0], label="maxspeed")
ax.barh(y - 0.2, najdluzsze_klasy["lanes_proc"], height=0.38, color=KOLORY[1], label="lanes")
etykiety_klas = [
    f"{klasa} ({dlugosc:,.0f} km)"
    for klasa, dlugosc in zip(najdluzsze_klasy.index, najdluzsze_klasy["dl_km"])
]
ax.set_yticks(y, etykiety_klas)
ax.set_xlim(0, 100)
ax.set_xlabel("% długości krawędzi z wypełnionym tagiem")
ax.set_title("Kompletność tagów OSM w klasach dróg", loc="left")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(WYKRESY / "kompletnosc_atrybutow.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2.5. Mapa sieci z podziałem na grupy dróg.
GRUPY_DROG = {
    "motorway": "główne",
    "motorway_link": "główne",
    "trunk": "główne",
    "trunk_link": "główne",
    "primary": "główne",
    "primary_link": "główne",
    "secondary": "zbiorcze",
    "secondary_link": "zbiorcze",
    "tertiary": "zbiorcze",
    "tertiary_link": "zbiorcze",
}
krawedzie["grupa"] = krawedzie["klasa"].map(GRUPY_DROG).fillna("lokalne")
fig, ax = plt.subplots(figsize=(9, 9))
granica_m.boundary.plot(ax=ax, color="#52514e", linewidth=0.8)
styl = {
    "lokalne": (SZARY, 0.3),
    "zbiorcze": ("#5598e7", 0.9),
    "główne": ("#0d366b", 1.6),
}
for grupa, (kolor, grubosc) in styl.items():
    krawedzie[krawedzie["grupa"] == grupa].plot(ax=ax, color=kolor, linewidth=grubosc, label=grupa)
ax.legend(title="Drogi", loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
ax.set_title("Sieć drogowa OSM", loc="left")
ax.set_axis_off()
ax.text(
    0.99,
    0.01,
    "Dane: © autorzy OpenStreetMap",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
)
fig.savefig(WYKRESY / "mapa_sieci_hierarchia_klasy.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 3. ROZKŁADY GTFS, PRZYSTANKI I ODJAZDY
# =============================================================================

# 3.1. Wczytanie tabel z trzech archiwów GTFS.
nazwy_tabel_gtfs = [
    "agency", "feed_info", "stops", "routes",
    "trips", "calendar", "calendar_dates", "stop_times",
]
kolumny_zatrzyman = {"trip_id", "stop_id", "departure_time", "stop_sequence"}

rozklady_gtfs = {}
for kod, url in GTFS.items():
    plik = pobierz(url, POBRANE / f"gtfs_{kod}.zip")
    tabele = {}
    with zipfile.ZipFile(plik) as archiwum:
        pliki = {Path(sciezka).name: sciezka for sciezka in archiwum.namelist()}
        for nazwa in nazwy_tabel_gtfs:
            if f"{nazwa}.txt" not in pliki:
                continue
            with archiwum.open(pliki[f"{nazwa}.txt"]) as dane:
                if nazwa == "stop_times":
                    # Największa tabela: wczytujemy tylko potrzebne kolumny.
                    tabela = pd.read_csv(
                        dane,
                        dtype=str,
                        encoding="utf-8-sig",
                        usecols=lambda kolumna: kolumna.strip() in kolumny_zatrzyman,
                    )
                else:
                    tabela = pd.read_csv(dane, dtype=str, encoding="utf-8-sig")
            tabela.columns = tabela.columns.str.strip()
            if nazwa != "stop_times":
                tabela = tabela.apply(lambda kolumna: kolumna.str.strip())
            tabele[nazwa] = tabela
    rozklady_gtfs[kod] = tabele

# 3.2. Połączenie przystanków. Źródło rozróżnia powtarzające się stop_id.
tabele_przystankow = []
for kod, tabele in rozklady_gtfs.items():
    tabela_przystankow = tabele["stops"].copy()
    tabela_przystankow["zrodlo"] = kod
    tabele_przystankow.append(tabela_przystankow)

tabela_przystankow = pd.concat(tabele_przystankow, ignore_index=True)
tabela_przystankow["stop_lat"] = tabela_przystankow["stop_lat"].astype(float)
tabela_przystankow["stop_lon"] = tabela_przystankow["stop_lon"].astype(float)
przystanki = gpd.GeoDataFrame(
    tabela_przystankow,
    geometry=gpd.points_from_xy(
        tabela_przystankow["stop_lon"],
        tabela_przystankow["stop_lat"],
    ),
    crs="EPSG:4326",
)
przystanki = przystanki.to_crs("EPSG:2180")


# 3.3. Najbliższa nieświąteczna środa z kursami we wszystkich rozkładach.
DATA_ANALIZY = dt.date.today()
for _ in range(400):
    if (
        DATA_ANALIZY.weekday() == 2
        and DATA_ANALIZY not in swieta_pl(DATA_ANALIZY.year)
        and all(uslugi_w_dniu(tabele, DATA_ANALIZY) for tabele in rozklady_gtfs.values())
    ):
        break
    DATA_ANALIZY += dt.timedelta(days=1)
else:
    raise ValueError("Nie znaleziono dnia z kursami. Pobierz aktualne rozkłady GTFS.")

# 3.4. Liczba odjazdów: wybór aktywnych kursów i zliczenie zatrzymań.
odjazdy_z_rozkladow = []
for kod, tabele in rozklady_gtfs.items():
    uslugi = uslugi_w_dniu(tabele, DATA_ANALIZY)
    wszystkie_kursy = tabele["trips"]
    wszystkie_zatrzymania = tabele["stop_times"]

    kursy = wszystkie_kursy[wszystkie_kursy["service_id"].isin(uslugi)]
    zatrzymania = wszystkie_zatrzymania[
        wszystkie_zatrzymania["trip_id"].isin(kursy["trip_id"])
    ]
    liczba_odjazdow = zatrzymania["stop_id"].value_counts()
    odjazdy_z_rozkladow.append(pd.DataFrame({
        "zrodlo": kod,
        "stop_id": liczba_odjazdow.index,
        "odjazdy": liczba_odjazdow.values,
    }))

odjazdy = pd.concat(odjazdy_z_rozkladow, ignore_index=True)
przystanki = przystanki.merge(odjazdy, on=["zrodlo", "stop_id"], how="left")
przystanki["odjazdy"] = przystanki["odjazdy"].fillna(0).astype(int)
print(f"Odjazdy w dniu {DATA_ANALIZY}: {przystanki['odjazdy'].sum():,}")

# 3.5. Zespoły przystankowe i mapa liczby odjazdów.
zespoly_przystankowe = przystanki.groupby("stop_name", as_index=False).agg(
    odjazdy=("odjazdy", "sum"), slupki=("stop_id", "count")
)
zespoly_przystankowe = zespoly_przystankowe.sort_values("odjazdy", ascending=False)
display(zespoly_przystankowe.head(10))

fig, ax = plt.subplots(figsize=(9, 9))
granica_m.boundary.plot(ax=ax, color="#52514e", linewidth=0.8)
krawedzie.plot(ax=ax, color=SZARY, linewidth=0.2)
czynne_przystanki = przystanki[przystanki["odjazdy"] > 0]
wielkosc_punktow = czynne_przystanki["odjazdy"] / czynne_przystanki["odjazdy"].max() * 120
czynne_przystanki.plot(ax=ax, color=KOLORY[0], alpha=0.5, markersize=wielkosc_punktow)
ax.set_title(f"Odjazdy ze słupków przystankowych, {DATA_ANALIZY}", loc="left")
ax.set_axis_off()
ax.text(
    0.99,
    0.01,
    "Dane: ZTP Kraków (GTFS), © autorzy OpenStreetMap",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
)
fig.savefig(WYKRESY / "odjazdy_przystanki.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 4. GENERALNY POMIAR RUCHU (GPR)
# =============================================================================

# 4.1. Pobranie, rozpakowanie i ujednolicenie nazw kolumn.
plik_gpr = pobierz(GPR_SHP_URL, POBRANE / "gpr2025_drogi_krajowe_shp.zip")
folder_gpr = plik_gpr.with_suffix("")
with zipfile.ZipFile(plik_gpr) as archiwum:
    archiwum.extractall(folder_gpr)
plik_shp = sorted(folder_gpr.rglob("*.shp"))[0]
gpr = gpd.read_file(plik_shp)
gpr = gpr.rename(columns={
    "NR2025": "nr_odcinka",
    "NRDROGI": "nr_drogi",
    "SDRR2025": "sdrr",
})
gpr = gpr[["nr_odcinka", "nr_drogi", "sdrr", "geometry"]].copy()
# SDRR: średnia liczba pojazdów na dobę. Usuwamy spacje i zamieniamy przecinki.
ruch_jako_tekst = gpr["sdrr"].astype(str)
ruch_jako_tekst = ruch_jako_tekst.str.replace(r"\s", "", regex=True)
ruch_jako_tekst = ruch_jako_tekst.str.replace(",", ".")
gpr["sdrr"] = pd.to_numeric(ruch_jako_tekst, errors="coerce")
gpr["nr_odcinka"] = gpr["nr_odcinka"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
gpr["nr_drogi"] = gpr["nr_drogi"].astype(str).str.strip()

# 4.2. Przeliczenie współrzędnych i wybór odcinków w obszarze analizy.
if gpr.crs is None:
    gpr = gpr.set_crs("EPSG:2180")
else:
    gpr = gpr.to_crs("EPSG:2180")

wojewodztwo_metry = ox.geocode_to_gdf(WOJEWODZTWO).to_crs("EPSG:2180")
gpr_wojewodztwo = gpr[gpr.intersects(wojewodztwo_metry.geometry.iloc[0])].copy()
gpr_miasto = gpr[gpr.intersects(granica_m.geometry.iloc[0])].copy()
display(gpr_wojewodztwo["sdrr"].describe().round(0).to_frame().T)
najbardziej_ruchliwe_odcinki = gpr_wojewodztwo.sort_values("sdrr", ascending=False)
display(najbardziej_ruchliwe_odcinki[["nr_odcinka", "nr_drogi", "sdrr"]].head(5))

# 4.3. Mapa natężenia ruchu w województwie.
fig, ax = plt.subplots(figsize=(10, 8))
wojewodztwo_metry.boundary.plot(ax=ax, color="#52514e", linewidth=0.8)
granica_m.plot(ax=ax, color="#e6e6e3")
grubosc_odcinkow = 0.8 + 3 * gpr_wojewodztwo["sdrr"] / gpr_wojewodztwo["sdrr"].max()
gpr_wojewodztwo.plot(
    ax=ax,
    column="sdrr",
    cmap=NIEBIESKI,
    linewidth=grubosc_odcinkow,
    legend=True,
    legend_kwds={"label": "SDRR [pojazdy/dobę]", "shrink": 0.6},
)
ax.set_title("Generalny Pomiar Ruchu 2025 – drogi krajowe w województwie", loc="left")
ax.set_axis_off()
ax.text(
    0.99,
    0.01,
    "Dane: GDDKiA (GPR 2025), © autorzy OpenStreetMap",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
)
fig.savefig(WYKRESY / "gpr_wojewodztwo.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 5. ŁĄCZENIE WARSTW: PRZYSTANKI I ULICE
# =============================================================================

# Każdemu przystankowi przypisujemy najbliższą ulicę w promieniu 150 m.
ulice = krawedzie.reset_index()[["klasa", "name", "geometry"]]
przystanki_z_ulica = gpd.sjoin_nearest(
    przystanki,
    ulice,
    how="left",
    max_distance=150,
    distance_col="odl_ulica_m",
)
# Przy remisie odległości zostawiamy jeden wynik dla danego przystanku.
przystanki_z_ulica = przystanki_z_ulica[~przystanki_z_ulica.index.duplicated()]
odleglosc_od_ulicy = przystanki_z_ulica["odl_ulica_m"]
daleko_od_ulicy = odleglosc_od_ulicy.isna() | (odleglosc_od_ulicy > 30)
print(
    f"Przystanki dalej niż 30 m od jezdni: {daleko_od_ulicy.sum()} "
    f"({100 * daleko_od_ulicy.mean():.1f}%)"
)
odlegle_przystanki = przystanki_z_ulica.loc[
    daleko_od_ulicy, ["stop_name", "zrodlo", "odl_ulica_m"]
]
odlegle_przystanki = odlegle_przystanki.sort_values("odl_ulica_m", ascending=False)
display(odlegle_przystanki.head(10))

# Histogram odległości od ulicy.
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(
    przystanki_z_ulica["odl_ulica_m"].dropna(),
    bins=np.arange(0, 155, 5),
    color=KOLORY[0],
    edgecolor="white",
)
ax.set_xlabel("odległość słupka od najbliższej krawędzi sieci drogowej [m]")
ax.set_ylabel("liczba słupków")
ax.set_title("Jak daleko od jezdni są słupki przystankowe?", loc="left")
fig.tight_layout()
fig.savefig(WYKRESY / "odleglosc_przystankow.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Wspólna mapa sieci OSM, przystanków i odcinków GPR.
fig, ax = plt.subplots(figsize=(10, 10))
granica_m.boundary.plot(ax=ax, color="#52514e", linewidth=0.8)
krawedzie.plot(ax=ax, color=SZARY, linewidth=0.25)
przystanki.plot(ax=ax, color=KOLORY[0], markersize=3, label="słupki przystankowe")
gpr_miasto.plot(ax=ax, column="sdrr", cmap=NIEBIESKI, linewidth=3, legend=True,
                legend_kwds={"label": "SDRR [pojazdy/dobę]", "shrink": 0.5})
xmin, ymin, xmax, ymax = granica_m.total_bounds
ax.set_xlim(xmin - 1000, xmax + 1000)
ax.set_ylim(ymin - 1000, ymax + 1000)
ax.set_title("Sieć OSM, przystanki GTFS i odcinki GPR 2025", loc="left")
ax.set_axis_off()
ax.text(
    0.99,
    0.01,
    "Dane: © autorzy OpenStreetMap, ZTP Kraków, GDDKiA",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
)
fig.savefig(WYKRESY / "wszystkie_warstwy.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 6. ZAPIS WYNIKÓW DO GEOPACKAGE I GRAPHML
# =============================================================================
PLIK_GPKG = FOLDER / f"L1_{SKROT}.gpkg"
warstwy = {
    "granica_miasta": granica_m,
    "siec_krawedzie": krawedzie.reset_index(),
    "przystanki": przystanki,
    "gpr_wojewodztwo": gpr_wojewodztwo,
    "granica_wojewodztwa": wojewodztwo_metry,
}
for nazwa, warstwa in warstwy.items():
    warstwa = warstwa.copy()
    # GeoPackage wymaga zamiany list atrybutów OSM na zwykły tekst.
    for kolumna in warstwa.select_dtypes(include="object"):
        warstwa[kolumna] = warstwa[kolumna].apply(
            lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple, set)) else x
        )
    warstwa.to_file(PLIK_GPKG, layer=nazwa, driver="GPKG")

PLIK_GRAF = FOLDER / f"L1_{SKROT}_graf.graphml"
ox.save_graphml(graf_metry, PLIK_GRAF)
print(f"Zapisano: {PLIK_GPKG}, {PLIK_GRAF}; wykresy: {WYKRESY}")


# =============================================================================
# 7. OPCJONALNE PORÓWNANIE Z BDOT10k
# =============================================================================

if WYKONAJ_BDOT:
    import pyogrio

    # 7.1. Rozpakowanie danych i znalezienie warstwy jezdni OT_SKJZ_L.
    if PLIK_BDOT.suffix.lower() == ".zip":
        folder_bdot = PLIK_BDOT.with_suffix("")
        with zipfile.ZipFile(PLIK_BDOT) as archiwum:
            archiwum.extractall(folder_bdot)
        pliki_bdot = sorted(folder_bdot.rglob("*"))
    else:
        pliki_bdot = [PLIK_BDOT]

    jezdnie = None
    for plik in pliki_bdot:
        if plik.suffix.lower() == ".gpkg":
            for nazwa in pyogrio.list_layers(plik)[:, 0]:
                if "skjz" in nazwa.lower():
                    jezdnie = gpd.read_file(plik, layer=nazwa)
                    break
        elif plik.suffix.lower() in (".shp", ".gml", ".xml") and "skjz" in plik.name.lower():
            jezdnie = gpd.read_file(plik)
        if jezdnie is not None:
            break
    if jezdnie is None:
        raise FileNotFoundError("Nie znaleziono warstwy OT_SKJZ_L w pliku BDOT10k.")

    jezdnie = jezdnie.to_crs("EPSG:2180")
    jezdnie = jezdnie[jezdnie.intersects(granica_m.geometry.iloc[0])].copy()
    dlugosc_jezdni_bdot_km = jezdnie.length.sum() / 1000

    # 7.2. Punkty co 25 m wzdłuż ulic OSM. Krótka ulica dostaje jeden punkt.
    ulice_nieskierowane = ox.graph_to_gdfs(graf_nieskierowany, nodes=False)
    fragmenty_punktow = []
    for indeks, linia in ulice_nieskierowane.geometry.items():
        if linia is None or linia.is_empty:
            continue
        if linia.length > 25:
            odleglosci = np.arange(12.5, linia.length, 25)
        else:
            odleglosci = np.array([linia.length / 2])

        punkty_linii = shapely.line_interpolate_point(linia, odleglosci)
        fragmenty_punktow.append(gpd.GeoDataFrame(
            {"id_linii": [indeks] * len(punkty_linii)},
            geometry=punkty_linii,
            crs=ulice_nieskierowane.crs,
        ))

    punkty_osm = pd.concat(fragmenty_punktow, ignore_index=True)

    # 7.3. Udział punktów leżących do 10 m od jezdni BDOT10k.
    punkty_z_jezdnia = gpd.sjoin_nearest(
        punkty_osm,
        jezdnie,
        how="left",
        max_distance=10,
        distance_col="odl",
    )
    punkty_z_jezdnia = punkty_z_jezdnia[~punkty_z_jezdnia.index.duplicated()]
    pokrycie_bdot_proc = punkty_z_jezdnia["odl"].notna().mean() * 100
    print(f"Jezdnie BDOT10k: {dlugosc_jezdni_bdot_km:.0f} km; ulice OSM: {dlugosc_ulic_km:.0f} km")
    print(f"{pokrycie_bdot_proc:.1f}% ulic OSM leży do 10 m od jezdni BDOT10k")

    fig, ax = plt.subplots(figsize=(9, 9))
    jezdnie.plot(ax=ax, color=KOLORY[1], linewidth=0.6, label="BDOT10k (OT_SKJZ_L)")
    ulice_nieskierowane.plot(ax=ax, color=KOLORY[0], linewidth=0.3, label="OSM (drive)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Jezdnie BDOT10k i ulice OSM", loc="left")
    ax.set_axis_off()
    ax.text(
        0.99,
        0.01,
        "Dane: GUGiK (BDOT10k), © autorzy OpenStreetMap",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
    )
    fig.savefig(WYKRESY / "bdot_osm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
