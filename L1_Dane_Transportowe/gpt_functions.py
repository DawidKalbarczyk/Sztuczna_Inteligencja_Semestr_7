import time, re
from pathlib import Path

import requests
import osmnx
import numpy as NP
import pandas as PD
from L1_dane_transportowe import FILE as FOLDER
import IPython.display
WYNIKI_SPRAWDZEN = {}

def sprawdz(zadanie, opis, warunek, podpowiedz=""):
    """Sprawdza warunek i wypisuje ✅ albo ❌ z podpowiedzią. `warunek` to zwykle funkcja lambda."""
    try:
        ok = bool(warunek() if callable(warunek) else warunek)
        blad = ""
    except Exception as e:
        ok, blad = False, f"{type(e).__name__}: {e}"
    print(f"{'✅' if ok else '❌'} [{zadanie}] {opis}")
    if not ok:
        if blad:
            print(f"   ⚠️ Błąd podczas sprawdzania: {blad}")
        if podpowiedz:
            print(f"   👉 {podpowiedz}")
    WYNIKI_SPRAWDZEN[(zadanie, opis)] = ok


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


# Serwery Overpass API, z których OSMnx pobiera dane OSM (główny i dwa zapasowe)
SERWERY_OVERPASS = ["https://overpass-api.de/api", "https://overpass.private.coffee/api",
                    "https://maps.mail.ru/osm/tools/overpass/api"]
osmnx.settings.use_cache = True
osmnx.settings.cache_folder = str(FOLDER / "cache_osm")
osmnx.settings.requests_timeout = 300
osmnx.settings.log_console = False

def z_osm(funkcja, *args, **kwargs):
    """Wywołuje funkcję OSMnx; gdy serwer nie odpowiada, próbuje kolejnego."""
    ostatni = None
    for i, url in enumerate(SERWERY_OVERPASS):
        osmnx.settings.overpass_url = url
        osmnx.settings.overpass_rate_limit = (i == 0)   # tylko serwer główny ma limit zapytań
        try:
            return funkcja(*args, **kwargs)
        except Exception as e:
            ostatni = e
            print(f"Serwer {url} nie odpowiedział ({type(e).__name__}: {str(e)[:120]}). Próbuję następnego…")
            time.sleep(3)
    raise RuntimeError("Serwery OSM nie odpowiadają. Odczekaj kilka minut i uruchom komórkę ponownie.") from ostatni


def pierwszy(x):
    """OSMnx przy upraszczaniu grafu czasem zapisuje atrybut jako listę, np. ['primary', 'secondary'].
    Funkcja zwraca pierwszy element listy albo samą wartość."""
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else None
    return x


def liczba(x):
    """Zamienia napis typu '50', '2;3' albo '50 mph' na pierwszą występującą liczbę; inaczej NaN."""
    x = pierwszy(x)
    if x is None or (isinstance(x, float) and NP.isnan(x)):
        return NP.nan
    m = re.search(r"\d+(?:[.,]\d+)?", str(x))
    return float(m.group().replace(",", ".")) if m else NP.nan


def do_zapisu(gdf):
    """GeoPackage nie przyjmuje list w kolumnach – zamieniamy je na napisy."""
    gdf = gdf.copy()
    for k in gdf.columns:
        if k != gdf.geometry.name and gdf[k].dtype == object:
            gdf[k] = gdf[k].apply(lambda v: ";".join(map(str, v)) if isinstance(v, (list, tuple, set)) else v)
    return gdf


def opisz_mape(ax, tytul, zrodlo="Dane: © autorzy OpenStreetMap"):
    """Wspólny wygląd map: tytuł, bez osi, źródło danych w rogu."""
    ax.set_title(tytul, loc="left", fontsize=12)
    ax.set_axis_off()
    ax.text(0.99, 0.01, zrodlo, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#52514e")


def podsumowanie():
    """Zestawienie wszystkich sprawdzeń z tego notatnika."""
    if not WYNIKI_SPRAWDZEN:
        print("Nie uruchomiono jeszcze żadnej komórki sprawdzającej.")
        return
    tab = PD.DataFrame([(z, o, "✅" if ok else "❌") for (z, o), ok in WYNIKI_SPRAWDZEN.items()],
                       columns=["zadanie", "sprawdzenie", "wynik"])
    IPython.display.display(tab)
    print(f"Zaliczone sprawdzenia: {(tab['wynik'] == '✅').sum()} z {len(tab)}")