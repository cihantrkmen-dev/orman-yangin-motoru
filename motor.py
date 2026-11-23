import re
import math
import heapq
import json
import xml.etree.ElementTree as ET
import requests
import numpy as np


def _doymus_buhar_basinci_kpa(sicaklik_c: float) -> float:
    """Magnus-Tetens formülü ile doymuş su buharı basıncı (kPa)."""
    return 0.61078 * math.exp((17.27 * sicaklik_c) / (sicaklik_c + 237.3))


def _buhar_basinci_acigi_vpd(sicaklik_c: float, bagil_nem: float) -> float:
    """Buhar Basıncı Açığı (VPD - kPa): Gece/gündüz yanabilirlik bariyeri."""
    es = _doymus_buhar_basinci_kpa(sicaklik_c)
    rh = max(1.0, min(100.0, bagil_nem))
    return max(0.0, es * (1.0 - rh / 100.0))


def _denge_nemi_emc(yakit_sicaklik_c: float, yakit_bagil_nem: float) -> float:
    """Simard (1968) denklemi ile ölü yanıcı madde denge nemi (EMC %)."""
    rh = max(1.0, min(100.0, yakit_bagil_nem))
    tf = yakit_sicaklik_c
    if rh < 10.0:
        emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * tf
    elif rh < 50.0:
        emc = 2.22749 + 0.160107 * rh - 0.01478 * tf
    else:
        emc = 21.0606 + 0.005565 * (rh ** 2) - 0.00035 * rh * tf - 0.483199 * rh
    return max(2.0, min(35.0, emc))


# 1. SON 10 GÜNLÜK YAĞIŞ/GÜNEŞLENME + 12 SAATLİK SAATLİK METEOROLOJİ TAHMİNİ
def hava_durumu_getir(enlem: float, boylam: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": enlem,
        "longitude": boylam,
        "past_days": 10,
        "forecast_days": 2,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "shortwave_radiation,sunshine_duration,et0_fao_evapotranspiration,"
            "wind_speed_10m,wind_direction_10m,is_day"
        ),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,is_day",
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        veri = r.json()

        saatlik = veri["hourly"]
        anlik = veri["current"]

        gecmis_saat = min(240, len(saatlik["temperature_2m"]))
        if gecmis_saat < 24:
            raise ValueError("Yetersiz geçmiş meteoroloji verisi")

        t_arr = saatlik["temperature_2m"][:gecmis_saat]
        rh_arr = saatlik["relative_humidity_2m"][:gecmis_saat]
        p_arr = saatlik["precipitation"][:gecmis_saat]
        rad_arr = saatlik.get("shortwave_radiation", [300.0] * gecmis_saat)[:gecmis_saat]
        sun_arr = saatlik.get("sunshine_duration", [1800.0] * gecmis_saat)[:gecmis_saat]
        et0_arr = saatlik.get("et0_fao_evapotranspiration", [0.15] * gecmis_saat)[:gecmis_saat]
        w_arr = saatlik["wind_speed_10m"][:gecmis_saat]

        # Zaman gecikmeli ölü yakıt nemi katmanları (1s, 10s, 100s, 240s)
        fm1, fm10, fm100, fm240 = 16.0, 17.0, 18.0, 19.0

        for i in range(gecmis_saat):
            t_air = float(t_arr[i] or 15.0)
            rh_air = float(rh_arr[i] or 55.0)
            yagis = float(p_arr[i] or 0.0)
            radyasyon = float(rad_arr[i] or 0.0)
            et0 = float(et0_arr[i] or 0.0)
            ruzgar = float(w_arr[i] or 10.0)

            if radyasyon > 40.0:
                t_fuel = t_air + (0.018 * radyasyon) / (1.0 + 0.14 * ruzgar)
            else:
                t_fuel = t_air - 0.8

            es_air = _doymus_buhar_basinci_kpa(t_air)
            es_fuel = _doymus_buhar_basinci_kpa(t_fuel)
            rh_fuel = min(98.0, rh_air * (es_air / max(0.05, es_fuel)))
            emc = _denge_nemi_emc(t_fuel, rh_fuel)

            if rh_air >= 72.0 and radyasyon < 80.0:
                emc = min(32.0, emc + (rh_air - 70.0) * 0.16)

            if yagis > 0.1:
                fm1 = min(45.0, fm1 + min(18.0, 5.0 + yagis * 3.5))
                fm10 = min(40.0, fm10 + min(10.0, 2.2 + yagis * 1.7))
                fm100 = min(36.0, fm100 + min(5.0, 0.9 + yagis * 0.7))
                fm240 = min(45.0, fm240 + min(3.5, 0.6 + yagis * 0.5))
            elif emc > fm1:
                fm1 = emc + (fm1 - emc) * math.exp(-0.65)
                fm10 = emc + (fm10 - emc) * math.exp(-0.08)
                fm100 = emc + (fm100 - emc) * math.exp(-0.01)
                fm240 = emc + (fm240 - emc) * math.exp(-0.004)
            else:
                radyasyon_carpan = 0.85 + (radyasyon / 1000.0) * 0.45 + (et0 * 0.35)
                fm1 = emc + (fm1 - emc) * math.exp(-min(0.85, radyasyon_carpan))
                fm10 = emc + (fm10 - emc) * math.exp(-radyasyon_carpan / 10.0)
                fm100 = emc + (fm100 - emc) * math.exp(-radyasyon_carpan / 100.0)
                fm240 = emc + (fm240 - emc) * math.exp(-radyasyon_carpan / 240.0)

        bilesik_yakit_nemi = round(0.45 * fm1 + 0.30 * fm10 + 0.15 * fm100 + 0.10 * fm240, 1)

        ort_sicaklik_10g = round(float(np.mean([x for x in t_arr if x is not None])), 1)
        ort_nem_10g = round(float(np.mean([x for x in rh_arr if x is not None])), 1)
        toplam_yagis_10g = round(float(np.sum([x for x in p_arr if x is not None])), 1)
        toplam_gunes_saat = round(float(np.sum([x for x in sun_arr if x is not None])) / 3600.0, 1)
        ort_radyasyon = round(float(np.mean([x for x in rad_arr if x is not None])), 0)

        # Canlı Yaprak Nemi (LFMC %) tahmini (Akdeniz Kızılçam/Maki: %75 yaz sonu - %125 kış)
        lfmc = round(float(np.clip(95.0 + (toplam_yagis_10g * 0.65) - max(0.0, ort_sicaklik_10g - 18.0) * 1.4, 72.0, 128.0)), 1)

        anlik_t = round(float(anlik["temperature_2m"]), 1)
        anlik_rh = round(float(anlik["relative_humidity_2m"]), 1)
        anlik_vpd = round(_buhar_basinci_acigi_vpd(anlik_t, anlik_rh), 2)

        # Önümüzdeki 12 saatin tam saatlik meteoroloji dizisi (Gece/Gündüz döngüsü için)
        gelecek_12saat = []
        is_day_arr = saatlik.get("is_day", [1] * (gecmis_saat + 12))
        for h_idx in range(gecmis_saat, gecmis_saat + 12):
            if h_idx < len(saatlik["temperature_2m"]):
                ht = float(saatlik["temperature_2m"][h_idx] or anlik_t)
                hrh = float(saatlik["relative_humidity_2m"][h_idx] or anlik_rh)
                hw = float(saatlik["wind_speed_10m"][h_idx] or anlik["wind_speed_10m"])
                hwd = int(saatlik["wind_direction_10m"][h_idx] or anlik["wind_direction_10m"])
                hrad = float(rad_arr[-1] if h_idx >= len(saatlik.get("shortwave_radiation", [])) else (saatlik["shortwave_radiation"][h_idx] or 0.0))
                hday = int(is_day_arr[h_idx]) if h_idx < len(is_day_arr) else (1 if hrad > 20 else 0)
            else:
                ht, hrh, hw, hwd, hrad, hday = anlik_t, anlik_rh, float(anlik["wind_speed_10m"]), int(anlik["wind_direction_10m"]), 0.0, int(anlik.get("is_day", 1))

            gelecek_12saat.append({
                "sicaklik": ht,
                "nem": hrh,
                "ruzgar_hiz": hw,
                "ruzgar_yon": hwd,
                "radyasyon": hrad,
                "is_day": hday,
                "vpd": _buhar_basinci_acigi_vpd(ht, hrh)
            })

        return {
            "anlik_sicaklik": anlik_t,
            "anlik_nem": anlik_rh,
            "anlik_vpd_kpa": anlik_vpd,
            "anlik_ruzgar_hizi": round(float(anlik["wind_speed_10m"]), 1),
            "anlik_ruzgar_yonu": int(anlik["wind_direction_10m"]),
            "son_5_gun_ort_sicaklik": ort_sicaklik_10g,
            "son_5_gun_ort_nem": ort_nem_10g,
            "son_5_gun_yagis_mm": toplam_yagis_10g,
            "son_10_gun_guneslenme_saat": toplam_gunes_saat,
            "son_10_gun_ort_radyasyon_wm2": ort_radyasyon,
            "olu_yakit_nemi_100saat": bilesik_yakit_nemi,
            "canli_yaprak_nemi_lfmc": lfmc,
            "fm1_baslangic": round(fm1, 1),
            "gelecek_12saat": gelecek_12saat,
            "kaynak": "Open-Meteo (10 Günlük Yağış/Güneşlenme + 12 Saatlik VPD Döngüsü)"
        }
    except Exception:
        yedek_12 = [{
            "sicaklik": 24.0, "nem": 42.0, "ruzgar_hiz": 16.0, "ruzgar_yon": 45,
            "radyasyon": 400.0, "is_day": 1, "vpd": 1.73
        } for _ in range(12)]
        return {
            "anlik_sicaklik": 24.0,
            "anlik_nem": 42.0,
            "anlik_vpd_kpa": 1.73,
            "anlik_ruzgar_hizi": 16.0,
            "anlik_ruzgar_yonu": 45,
            "son_5_gun_ort_sicaklik": 22.0,
            "son_5_gun_ort_nem": 46.0,
            "son_5_gun_yagis_mm": 2.0,
            "son_10_gun_guneslenme_saat": 68.0,
            "son_10_gun_ort_radyasyon_wm2": 420.0,
            "olu_yakit_nemi_100saat": 14.5,
            "canli_yaprak_nemi_lfmc": 92.0,
            "fm1_baslangic": 13.5,
            "gelecek_12saat": yedek_12,
            "kaynak": "Varsayılan Akdeniz-Ege Profili (Yedek)"
        }


# 2. ARAZİ YAPISI - COPERNICUS DEM + DERE / SIRT (TPI) VE BAKI ANALİZİ
def arazi_dem_getir(enlem: float, boylam: float, boyut: int = 60) -> dict:
    yari_cap = 0.055
    ornek_n = 10
    lats = np.linspace(enlem - yari_cap, enlem + yari_cap, ornek_n)
    lons = np.linspace(boylam - yari_cap, boylam + yari_cap, ornek_n)

    lat_list, lon_list = [], []
    for la in lats:
        for lo in lons:
            lat_list.append(f"{la:.5f}")
            lon_list.append(f"{lo:.5f}")

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": ",".join(lat_list), "longitude": ",".join(lon_list)},
            timeout=10
        )
        r.raise_for_status()
        elev_10 = np.array(r.json()["elevation"], dtype=float).reshape((ornek_n, ornek_n))
    except Exception:
        elev_10 = np.full((ornek_n, ornek_n), 450.0)

    y_old = np.linspace(0, 1, ornek_n)
    x_old = np.linspace(0, 1, ornek_n)
    y_new = np.linspace(0, 1, boyut)
    x_new = np.linspace(0, 1, boyut)

    temp = np.array([np.interp(x_new, x_old, row) for row in elev_10])
    elev_grid = np.array([np.interp(y_new, y_old, temp[:, c]) for c in range(boyut)]).T

    hucre_m = 200.0
    dy, dx = np.gradient(elev_grid, hucre_m)
    egim_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    egim_derece = np.degrees(egim_rad)
    baki_derece = (np.degrees(np.arctan2(-dx, dy)) + 360.0) % 360.0

    pad_elev = np.pad(elev_grid, 3, mode="edge")
    komsu_ort = np.zeros_like(elev_grid)
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            komsu_ort += pad_elev[3 + dr:3 + dr + boyut, 3 + dc:3 + dc + boyut]
    komsu_ort /= 49.0
    tpi_grid = elev_grid - komsu_ort

    merkez = boyut // 2
    merkez_baki = float(baki_derece[merkez, merkez])
    yonler = ["Kuzey", "Kuzeydoğu", "Doğu", "Güneydoğu", "Güney", "Güneybatı", "Batı", "Kuzeybatı"]
    baki_adi = yonler[int((merkez_baki + 22.5) // 45) % 8]

    return {
        "yukselti_grid": elev_grid,
        "egim_grid": egim_derece,
        "baki_grid": baki_derece,
        "tpi_grid": tpi_grid,
        "yari_cap": yari_cap,
        "hucre_m": hucre_m,
        "merkez_rakim_m": int(round(float(elev_grid[merkez, merkez]))),
        "ortalama_egim_derece": round(float(np.mean(egim_derece)), 1),
        "merkez_baki": baki_adi,
        "kaynak": "Copernicus DEM (TPI Dere/Sırt & Bakı)"
    }


def _cizgi_izgaraya_isle(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, deger, carpan_grid: np.ndarray = None, carpan: float = 1.0, cbh_grid: np.ndarray = None, cbh: float = 3.5):
    adim = max(abs(r1 - r0), abs(c1 - c0), 1)
    for k in range(adim + 1):
        rr = int(round(r0 + (r1 - r0) * (k / adim)))
        cc = int(round(c0 + (c1 - c0) * (k / adim)))
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = deger
            if carpan_grid is not None:
                carpan_grid[rr, cc] = carpan
            if cbh_grid is not None:
                cbh_grid[rr, cc] = cbh


def _poligon_icini_doldur(grid: np.ndarray, halka: list[tuple[int, int]], deger: int, carpan_grid: np.ndarray, carpan: float, cbh_grid: np.ndarray, cbh: float):
    if len(halka) < 3:
        return
    for idx in range(len(halka) - 1):
        _cizgi_izgaraya_isle(grid, halka[idx][0], halka[idx][1], halka[idx + 1][0], halka[idx + 1][1], deger, carpan_grid, carpan, cbh_grid, cbh)

    rs = [p[0] for p in halka]
    cs = [p[1] for p in halka]
    r_min, r_max = max(0, min(rs)), min(grid.shape[0] - 1, max(rs))
    c_min, c_max = max(0, min(cs)), min(grid.shape[1] - 1, max(cs))

    n = len(halka)
    for rr in range(r_min, r_max + 1):
        for cc in range(c_min, c_max + 1):
            icerde = False
            p1r, p1c = halka[0]
            for i in range(1, n + 1):
                p2r, p2c = halka[i % n]
                if cc > min(p1c, p2c) and cc <= max(p1c, p2c) and rr <= max(p1r, p2r):
                    if p1c != p2c:
                        xinters = (cc - p1c) * (p2r - p1r) / (p2c - p1c) + p1r
                    if p1r == p2r or rr <= xinters:
                        icerde = not icerde
                p1r, p1c = p2r, p2c
            if icerde:
                grid[rr, cc] = deger
                carpan_grid[rr, cc] = carpan
                cbh_grid[rr, cc] = cbh


# 3. AKDENİZ VE EGE ORTALAMASI: KIZILÇAM + MAKİ TABAN MODELİ & ENGELLER
def bitki_ve_engeller_getir(enlem: float, boylam: float, dem: dict) -> dict:
    elev_grid = dem["yukselti_grid"]
    baki_grid = dem["baki_grid"]
    boyut = elev_grid.shape[0]
    yari_cap = dem["yari_cap"]
    min_lat, max_lat = enlem - yari_cap, enlem + yari_cap
    min_lon, max_lon = boylam - yari_cap, boylam + yari_cap

    guney_yamac = (baki_grid >= 110.0) & (baki_grid <= 250.0)
    yakit_grid = np.where(
        elev_grid <= 1.0, 0,
        np.where(
            elev_grid < 220.0, 2,
            np.where(
                elev_grid < 950.0,
                np.where(guney_yamac & (elev_grid < 550.0), 2, 1),
                3
            )
        )
    )
    yakit_carpan_grid = np.ones((boyut, boyut), dtype=float)
    # Tepe Taban Yüksekliği (CBH - metre): Kızılçam+Maki=2.2m, Saf Maki=0.8m, Karaçam=5.5m
    cbh_grid = np.where(yakit_grid == 1, 2.2, np.where(yakit_grid == 2, 0.8, np.where(yakit_grid == 3, 5.5, 8.0)))

    engel_sayisi = 0
    osm_basarili = False

    query = f"""
    [out:json][timeout:12];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|track"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["waterway"~"river|stream|canal"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["natural"~"water|wood|scrub|bare_rock|coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["landuse"~"forest|residential|farmland|orchard"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """

    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=12)
        if r.status_code == 200:
            data = r.json()
            nodes = {el["id"]: (el["lat"], el["lon"]) for el in data.get("elements", []) if el["type"] == "node"}
            for el in data.get("elements", []):
                if el["type"] != "way":
                    continue
                tags = el.get("tags", {})
                sinif = None
                cbh_val = 2.5
                if (
                    "highway" in tags
                    or "waterway" in tags
                    or tags.get("natural") in ("water", "bare_rock", "coastline")
                    or tags.get("landuse") == "residential"
                ):
                    sinif = 0
                    engel_sayisi += 1
                elif tags.get("natural") == "scrub":
                    sinif = 2
                    cbh_val = 0.8
                elif tags.get("landuse") in ("farmland", "orchard"):
                    sinif = 4
                    cbh_val = 9.0
                elif tags.get("natural") == "wood" or tags.get("landuse") == "forest":
                    sinif = 3 if tags.get("leaf_type") == "broadleaved" else 1
                    cbh_val = 5.0 if sinif == 3 else 2.2

                if sinif is not None:
                    yol_noktalari = []
                    for nid in el.get("nodes", []):
                        if nid in nodes:
                            la, lo = nodes[nid]
                            ri = int((la - min_lat) / (max_lat - min_lat) * (boyut - 1))
                            ci = int((lo - min_lon) / (max_lon - min_lon) * (boyut - 1))
                            yol_noktalari.append((ri, ci))
                    for idx in range(len(yol_noktalari) - 1):
                        r0, c0 = yol_noktalari[idx]
                        r1, c1 = yol_noktalari[idx + 1]
                        _cizgi_izgaraya_isle(yakit_grid, r0, c0, r1, c1, sinif, cbh_grid=cbh_grid, cbh=cbh_val)
            osm_basarili = True
    except Exception:
        pass

    yakit_grid[elev_grid <= 1.0] = 0

    kz_oran = int(round(np.mean(yakit_grid == 1) * 100))
    maki_oran = int(round(np.mean(yakit_grid == 2) * 100))
    if kz_oran + maki_oran >= 45:
        baskin_ad = f"Kızılçam (%{kz_oran}) + Maki (%{maki_oran})"
    else:
        sinif_adlari = {
            1: "Kızılçam + Maki Alt Örtülü Orman",
            2: "Akdeniz–Ege Maki Formasyonu",
            3: "Karaçam / Yüksek Rakım Ormanı",
            4: "Zeytinlik / Tarım ve Açıklık",
            0: "Su / Kıyı / Yanmaz Engel"
        }
        degerler, sayilar = np.unique(yakit_grid[yakit_grid > 0], return_counts=True)
        baskin_kod = int(degerler[np.argmax(sayilar)]) if len(degerler) > 0 else 1
        baskin_ad = sinif_adlari[baskin_kod]

    return {
        "yakit_grid": yakit_grid,
        "yakit_carpan_grid": yakit_carpan_grid,
        "cbh_grid": cbh_grid,
        "baskin_bitki_ortusu": baskin_ad,
        "tespit_edilen_engel_hatti": engel_sayisi,
        "kaynak": "Akdeniz-Ege Kızılçam/Maki Modeli + OSM" if osm_basarili else "Akdeniz-Ege Kızılçam/Maki Ekolojik Modeli"
    }


def _mescere_rumuzu_coz(etiket_metni: str, varsayilan_poligon: bool = True) -> tuple[int, float, float]:
    """Meşcere tipi, yaş sınıfı (a,b,c,d) ve kapalılık (1,2,3) -> (yakıt_sınıfı, hız_çarpanı, CBH_m)."""
    temiz = re.sub(r"<[^>]+>", " ", etiket_metni).lower()

    if any(k in temiz for k in ["yey", "emniyet", "orman yolu", "yol", "dere", "akarsu", "gölet", "baraj", "su", "kaya", "iskân", "iskan"]):
        return 0, 0.0, 99.0

    eslesme = re.search(r"\b(çz|cz|çk|ck|mb|md|m|bçz|bçk|kz|kn|me|z|ot|a|t)([abcd]{1,2})?([0123])?\b", temiz)
    if eslesme:
        tur = eslesme.group(1)
        cag = eslesme.group(2) or ""
        kapalilik = eslesme.group(3) or ""

        if tur in ("çz", "cz", "bçz", "kz"):
            sinif = 1
        elif tur in ("m", "mb", "md"):
            sinif = 2
        elif tur in ("çk", "ck", "bçk", "kn", "me"):
            sinif = 3
        else:
            sinif = 4

        if "a" in cag and "b" not in cag:
            cag_carpan, cbh = 1.30, 1.0  # Gençlik / Sıklık (dallar yere yakın)
        elif "a" in cag and "b" in cag:
            cag_carpan, cbh = 1.24, 1.5
        elif "b" in cag and "c" not in cag:
            cag_carpan, cbh = 1.15, 2.2  # Sırıklık çağı
        elif "b" in cag and "c" in cag:
            cag_carpan, cbh = 1.08, 3.2
        elif "c" in cag and "d" not in cag:
            cag_carpan, cbh = 1.00, 4.8  # Direklik çağı
        elif "d" in cag:
            cag_carpan, cbh = 0.85, 7.5  # Ağaçlık / Yaşlı meşcere
        else:
            cag_carpan = 0.85 if tur == "mb" else 1.00
            cbh = 0.8 if sinif == 2 else (5.5 if sinif == 3 else 2.5)

        if kapalilik == "3":
            kap_carpan = 1.12
        elif kapalilik == "1":
            kap_carpan = 0.88
            cbh += 1.2
        elif kapalilik == "0":
            kap_carpan = 0.65
            cbh += 3.0
        else:
            kap_carpan = 1.00

        return sinif, round(cag_carpan * kap_carpan, 2), cbh

    if "maki" in temiz or "çalı" in temiz or "cali" in temiz:
        return 2, 1.0, 0.8
    if "gençlik" in temiz or "sıklık" in temiz:
        return 1, 1.30, 1.0
    if "sırıklık" in temiz:
        return 1, 1.15, 2.2
    if "ağaçlık" in temiz or "yaşlı" in temiz:
        return 1, 0.85, 7.5
    if "zeytin" in temiz or "tarım" in temiz or "açıklık" in temiz:
        return 4, 0.80, 9.0

    return (1, 1.0, 2.5) if varsayilan_poligon else (0, 0.0, 99.0)


# 4. İSTEĞE BAĞLI KML / GEOJSON MEŞCERE VE YOL KATMAN ENTEGRASYONU
def ozel_katman_uygula(yakit_grid: np.ndarray, enlem: float, boylam: float, yari_cap: float, dosya_icerik: str, carpan_grid: np.ndarray = None, cbh_grid: np.ndarray = None) -> tuple[np.ndarray, int]:
    boyut = yakit_grid.shape[0]
    if carpan_grid is None:
        carpan_grid = np.ones((boyut, boyut), dtype=float)
    if cbh_grid is None:
        cbh_grid = np.full((boyut, boyut), 2.5, dtype=float)

    min_lat, max_lat = enlem - yari_cap, enlem + yari_cap
    min_lon, max_lon = boylam - yari_cap, boylam + yari_cap
    islenen_nesne = 0

    def koordinat_noktasi(lon_val: float, lat_val: float) -> tuple[int, int]:
        ri = int((lat_val - min_lat) / (max_lat - min_lat) * (boyut - 1))
        ci = int((lon_val - min_lon) / (max_lon - min_lon) * (boyut - 1))
        return ri, ci

    metin = dosya_icerik.strip()
    if metin.startswith("{"):
        geo = json.loads(metin)
        for feat in geo.get("features", []):
            props_str = str(feat.get("properties", {}))
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])

            if gtype == "LineString" and len(coords) >= 2:
                sinif, carpan, cbh = _mescere_rumuzu_coz(props_str, varsayilan_poligon=False)
                for idx in range(len(coords) - 1):
                    r0, c0 = koordinat_noktasi(coords[idx][0], coords[idx][1])
                    r1, c1 = koordinat_noktasi(coords[idx + 1][0], coords[idx + 1][1])
                    _cizgi_izgaraya_isle(yakit_grid, r0, c0, r1, c1, sinif, carpan_grid, carpan, cbh_grid, cbh)
                islenen_nesne += 1
            elif gtype == "Polygon" and coords:
                sinif, carpan, cbh = _mescere_rumuzu_coz(props_str, varsayilan_poligon=True)
                halka = [koordinat_noktasi(c[0], c[1]) for c in coords[0]]
                _poligon_icini_doldur(yakit_grid, halka, sinif, carpan_grid, carpan, cbh_grid, cbh)
                islenen_nesne += 1

    elif "<kml" in metin.lower():
        root = ET.fromstring(metin)
        for pm in root.iter():
            if not pm.tag.endswith("Placemark"):
                continue

            etiket_parcalari = []
            for alt in pm.iter():
                if alt.tag.endswith(("name", "description", "value", "SimpleData")) and alt.text:
                    etiket_parcalari.append(alt.text.strip())
            tum_etiket = " ".join(etiket_parcalari)

            poligon_var = any(alt.tag.endswith("Polygon") for alt in pm.iter())
            sinif, carpan, cbh = _mescere_rumuzu_coz(tum_etiket, varsayilan_poligon=poligon_var)

            for alt in pm.iter():
                if alt.tag.endswith("coordinates") and alt.text:
                    parcalar = alt.text.strip().split()
                    noktalar = []
                    for p in parcalar:
                        xyz = p.split(",")
                        if len(xyz) >= 2:
                            noktalar.append(koordinat_noktasi(float(xyz[0]), float(xyz[1])))
                    if len(noktalar) >= 3 and poligon_var:
                        _poligon_icini_doldur(yakit_grid, noktalar, sinif, carpan_grid, carpan, cbh_grid, cbh)
                        islenen_nesne += 1
                    elif len(noktalar) >= 2:
                        for idx in range(len(noktalar) - 1):
                            _cizgi_izgaraya_isle(yakit_grid, noktalar[idx][0], noktalar[idx][1], noktalar[idx + 1][0], noktalar[idx + 1][1], sinif, carpan_grid, carpan, cbh_grid, cbh)
                        islenen_nesne += 1

    return yakit_grid, islenen_nesne


# 5. ROTHERMEL (1972) + BYRAM (1959) + VAN WAGNER (1977) + BALCH VPD (2022) SİMÜLASYON MOTORU
def _rothermel_hesapla(yakit_sinif: int, mf_yuzde: float, ruzgar_10m_kmh: float, cos_ruzgar_aci: float, egim_tan: float) -> tuple[float, float, float]:
    """
    Rothermel (1972) yarı-deneysel yüzey yangını denklemi ve Byram (1959) yangın şiddeti.
    Dönüş: (yuzey_yayilim_hizi_m_dk, byram_siddeti_kw_m, alev_boyu_m)
    """
    # Yakıt yatağı fiziksel parametreleri (NFFL & Akdeniz-Ege kalibrasyonu)
    # (w0 kg/m2, sigma m^-1, rho_b kg/m3, Mx_yuzde, waf_ruzgar_azaltma)
    parametreler = {
        1: (1.85, 5700.0, 16.0, 24.0, 0.42),  # Kızılçam + Maki alt örtüsü
        2: (1.60, 6200.0, 18.0, 23.0, 0.55),  # Saf Maki ve Sık Çalılık
        3: (1.35, 5100.0, 19.0, 25.0, 0.35),  # Karaçam / Yüksek Rakım Ormanı
        4: (0.65, 7800.0, 8.5, 20.0, 0.65),   # Zeytinlik / Ot ve Tarım
    }
    w0, sigma, rho_b, mx, waf = parametreler.get(yakit_sinif, (1.5, 5600.0, 16.0, 24.0, 0.45))

    if mf_yuzde >= mx:
        return 0.0, 0.0, 0.0

    # Rothermel nem sönümleme katsayısı (eta_M)
    r_m = mf_yuzde / mx
    eta_m = max(0.0, 1.0 - 2.59 * r_m + 5.11 * (r_m ** 2) - 3.52 * (r_m ** 3))
    eta_s = 0.417  # Mineral sönümleme katsayısı
    h_yanma = 18700.0  # Düşük yanma ısısı (kJ/kg)
    w_n = w0 * 0.9445

    sigma_imp = sigma / 3.28084
    rho_p = 513.0
    beta = rho_b / rho_p
    beta_op = 3.348 * (sigma_imp ** -0.8189)
    beta_oran = beta / beta_op

    gamma_max = (sigma_imp ** 1.5) / (495.0 + 0.0594 * (sigma_imp ** 1.5))
    a_exp = 133.0 * (sigma_imp ** -0.7913)
    gamma_prime = gamma_max * (beta_oran ** a_exp) * math.exp(a_exp * (1.0 - beta_oran))

    # Reaksiyon şiddeti I_R (kJ / m^2 * dk)
    i_r = gamma_prime * w_n * h_yanma * eta_m * eta_s

    # Yayılan ısı akısı oranı (xi) ve tutuşma ısısı (Q_ig)
    xi = math.exp((0.792 + 0.681 * math.sqrt(sigma_imp)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma_imp)
    epsilon = math.exp(-138.0 / sigma_imp)
    q_ig = 581.0 + 2594.0 * (mf_yuzde / 100.0)

    r0 = (i_r * xi) / max(1.0, rho_b * epsilon * q_ig)

    # Orta alev rüzgar hızı (ft/min) ve Rothermel phi_W faktörü
    u_mid_ft_min = (ruzgar_10m_kmh * waf * 1000.0 / 60.0) * 3.28084
    c_wind = 7.47 * math.exp(-0.133 * (sigma_imp ** 0.55))
    b_wind = 0.02526 * (sigma_imp ** 0.54)
    e_wind = 0.715 * math.exp(-3.59e-4 * sigma_imp)

    if cos_ruzgar_aci >= 0.0:
        phi_w = c_wind * ((u_mid_ft_min * cos_ruzgar_aci) ** b_wind) * (beta_oran ** -e_wind)
    else:
        phi_w = -0.35 * abs(cos_ruzgar_aci)

    # Rothermel eğim faktörü (phi_S)
    if egim_tan >= 0.0:
        phi_s = 5.275 * (beta ** -0.3) * (egim_tan ** 2.0)
    else:
        phi_s = max(-0.65, -1.8 * abs(egim_tan))

    ros = max(0.05, r0 * (1.0 + phi_w + phi_s))

    # Byram (1959) Yangın Hattı Şiddeti (kW/m) ve Alev Boyu (m)
    i_r_kw_m2 = i_r / 60.0
    kalma_suresi_dk = 384.0 / sigma_imp
    alev_derinligi_m = ros * kalma_suresi_dk
    byram_ib = i_r_kw_m2 * alev_derinligi_m
    alev_boyu = 0.0775 * (byram_ib ** 0.46) if byram_ib > 0 else 0.0

    return ros, byram_ib, alev_boyu


def yangin_simule_et(enlem: float, boylam: float, hava: dict, dem: dict, bitki: dict) -> dict:
    yakit = bitki["yakit_grid"]
    carpan_grid = bitki.get("yakit_carpan_grid", np.ones_like(yakit, dtype=float))
    cbh_grid = bitki.get("cbh_grid", np.full_like(yakit, 2.5, dtype=float))
    elev = dem["yukselti_grid"]
    baki = dem["baki_grid"]
    tpi = dem["tpi_grid"]
    boyut = yakit.shape[0]
    hucre_m = dem["hucre_m"]
    yari_cap = dem["yari_cap"]
    merkez = boyut // 2

    fm_bilesik = hava["olu_yakit_nemi_100saat"]
    lfmc = hava.get("canli_yaprak_nemi_lfmc", 92.0)
    gelecek_12 = hava["gelecek_12saat"]
    radyasyon_ort = hava.get("son_10_gun_ort_radyasyon_wm2", 400.0)

    yanma_saati = np.full((boyut, boyut), 999.0)
    yanma_saati[merkez, merkez] = 0.0

    komsular = [
        (1, 0, hucre_m, 0.0),
        (1, 1, hucre_m * 1.414, 45.0),
        (0, 1, hucre_m, 90.0),
        (-1, 1, hucre_m * 1.414, 135.0),
        (-1, 0, hucre_m, 180.0),
        (-1, -1, hucre_m * 1.414, 225.0),
        (0, -1, hucre_m, 270.0),
        (1, -1, hucre_m * 1.414, 315.0),
    ]

    saatlik_max_byram = [0.0] * 12
    saatlik_tepe_aktif = [False] * 12
    saatlik_gece_freni = [False] * 12
    saatlik_vpd = [round(h["vpd"], 2) for h in gelecek_12]

    kuyruk = [(0.0, merkez, merkez)]
    while kuyruk:
        mevcut_t, r, c = heapq.heappop(kuyruk)
        if mevcut_t > yanma_saati[r, c] or mevcut_t >= 12.0:
            continue

        saat_indeks = min(11, int(mevcut_t))
        saat_hava = gelecek_12[saat_indeks]
        ruzgar_hiz = saat_hava["ruzgar_hiz"]
        ruzgar_gidis_yon = (saat_hava["ruzgar_yon"] + 180.0) % 360.0
        saat_vpd = saat_hava["vpd"]
        is_day = saat_hava["is_day"]

        # BALCH (2022) GECE / GÜNDÜZ VPD BARİYERİ VE GECE NEM GERİ KAZANIMI
        gece_nem_eklemesi = 0.0
        vpd_freni = 1.0
        if is_day == 0:
            # Gece radyatif soğuma ile ince yakıt nemi artar
            gece_nem_eklemesi = max(0.0, (saat_hava["nem"] - 55.0) * 0.14)
            # Sırt hatları (TPI > 15m) gece inversiyonunun üzerinde kaldığı için rüzgarla daha kuru kalır
            if tpi[r, c] > 15.0 and ruzgar_hiz > 18.0:
                gece_nem_eklemesi *= 0.45
            if saat_vpd < 0.45:
                vpd_freni = 0.12  # Gece VPD eşiği altında yangın sönümlenir
                saatlik_gece_freni[saat_indeks] = True
            elif saat_vpd < 0.75:
                vpd_freni = 0.55

        # McAlpine (1991) Nokta Kaynak Hızlanma Fazı (ilk 35 dakikada kademeli ivmelenme)
        baslangic_ivme = 1.0 - 0.55 * math.exp(-mevcut_t / 0.35) if mevcut_t < 1.5 else 1.0

        for dr, dc, mesafe, yon_azimut in komsular:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < boyut and 0 <= nc < boyut):
                continue

            hedef_yakit = int(yakit[nr, nc])
            if hedef_yakit == 0:
                if elev[nr, nc] <= 1.0:
                    continue
                # Albini (1979) Kozalak Sıçraması (Spotting): yalnızca yüksek Byram şiddeti ve rüzgarda yol atlar
                if saatlik_max_byram[saat_indeks] > 1800.0 and ruzgar_hiz >= 28.0 and int(yakit[r, c]) in (1, 2):
                    yayilim_hizi_m_dk = 2.0
                    gecis_saati = (mesafe / yayilim_hizi_m_dk) / 60.0
                    yeni_t = mevcut_t + gecis_saati
                    if yeni_t < yanma_saati[nr, nc] and yeni_t <= 12.0:
                        yanma_saati[nr, nc] = yeni_t
                        heapq.heappush(kuyruk, (yeni_t, nr, nc))
                continue

            # Bakı + Güneş Radyasyonu lokal yakıt nemi düzeltmesi
            baki_rad = math.radians(float(baki[nr, nc]))
            gunes_faktoru = min(1.0, saat_hava["radyasyon"] / 650.0) if is_day == 1 else 0.0
            baki_nem_farki = 1.6 * math.cos(baki_rad) * gunes_faktoru  # Kuzey(+), Güney(-)
            yerel_mf = max(3.5, fm_bilesik + gece_nem_eklemesi + baki_nem_farki)

            cos_aci = math.cos(math.radians(yon_azimut - ruzgar_gidis_yon))
            egim_tan = (elev[nr, nc] - elev[r, c]) / mesafe

            ros_yuzey, byram_ib, _ = _rothermel_hesapla(hedef_yakit, yerel_mf, ruzgar_hiz, cos_aci, egim_tan)
            if ros_yuzey <= 0.0:
                continue

            # Viegas (2005) Dar Dere / Vadi Baca Etkisi ve Sırt Freni
            hedef_tpi = float(tpi[nr, nc])
            if hedef_tpi < -10.0 and egim_tan > 0.06:
                topo_carpan = min(1.90, 1.0 + abs(hedef_tpi) / 32.0)
            elif hedef_tpi > 16.0 and egim_tan < 0.0:
                topo_carpan = 0.65
            else:
                topo_carpan = 1.0

            ros_etkin = ros_yuzey * float(carpan_grid[nr, nc]) * topo_carpan * vpd_freni * baslangic_ivme
            byram_etkin = byram_ib * float(carpan_grid[nr, nc]) * topo_carpan

            # VAN WAGNER (1977) TEPE YANGINI GEÇİŞ EŞİĞİ (I_0)
            cbh_m = float(cbh_grid[nr, nc])
            i0_kritik = (0.010 * cbh_m * (460.0 + 25.9 * lfmc)) ** 1.5
            if byram_etkin >= i0_kritik and hedef_yakit in (1, 2, 3):
                # Cruz vd. (2005) Aktif Tepe Yangını Hızlanması
                ros_etkin *= 1.75
                saatlik_tepe_aktif[saat_indeks] = True

            if byram_etkin > saatlik_max_byram[saat_indeks]:
                saatlik_max_byram[saat_indeks] = byram_etkin

            yayilim_hizi_m_dk = max(0.08, ros_etkin)
            gecis_saati = (mesafe / yayilim_hizi_m_dk) / 60.0
            yeni_t = mevcut_t + gecis_saati

            if yeni_t < yanma_saati[nr, nc] and yeni_t <= 12.0:
                yanma_saati[nr, nc] = yeni_t
                heapq.heappush(kuyruk, (yeni_t, nr, nc))

    min_lat, max_lat = enlem - yari_cap, enlem + yari_cap
    min_lon, max_lon = boylam - yari_cap, boylam + yari_cap
    hucre_hektar = (hucre_m * hucre_m) / 10000.0

    def saat_poligonu_uret(hedef_saat: float) -> list:
        yanan_r, yanan_c = np.where(yanma_saati <= hedef_saat)
        if len(yanan_r) <= 1:
            d = 0.0008
            return [[
                [round(boylam + d * math.cos(math.radians(a)), 5), round(enlem + d * math.sin(math.radians(a)), 5)]
                for a in range(0, 361, 20)
            ]]

        sektor_sayisi = 36
        max_uzaklik = np.full(sektor_sayisi, 0.0009)
        for rr, cc in zip(yanan_r, yanan_c):
            la = min_lat + (rr / (boyut - 1)) * (max_lat - min_lat)
            lo = min_lon + (cc / (boyut - 1)) * (max_lon - min_lon)
            dlat = la - enlem
            dlon = lo - boylam
            mes = math.hypot(dlat, dlon) + 0.0009
            aci = (math.degrees(math.atan2(dlat, dlon)) + 360.0) % 360.0
            sek = int(aci // 10) % sektor_sayisi
            if mes > max_uzaklik[sek]:
                max_uzaklik[sek] = mes

        yumusak = np.copy(max_uzaklik)
        for i in range(sektor_sayisi):
            yumusak[i] = 0.25 * max_uzaklik[(i - 1) % sektor_sayisi] + 0.5 * max_uzaklik[i] + 0.25 * max_uzaklik[(i + 1) % sektor_sayisi]

        halka = []
        for i in range(sektor_sayisi):
            rad = math.radians(i * 10 + 5)
            lo = round(boylam + yumusak[i] * math.cos(rad), 5)
            la = round(enlem + yumusak[i] * math.sin(rad), 5)
            halka.append([lo, la])
        halka.append(halka[0])
        return [halka]

    izokronlar = []
    renkler = {1: "#B71C1C", 3: "#E65100", 6: "#F57C00", 12: "#FBC02D"}
    for s in [1, 3, 6, 12]:
        yanan_hucre = int(np.sum(yanma_saati <= s))
        alan_hektar = round(yanan_hucre * hucre_hektar, 1)
        izokronlar.append({
            "saat": s,
            "renk": renkler[s],
            "etkilenen_alan_hektar": alan_hektar,
            "geometri": {
                "type": "Polygon",
                "coordinates": saat_poligonu_uret(float(s))
            }
        })

    saatlik_ozet = []
    for s in range(1, 13):
        idx = s - 1
        yanan_hucre = int(np.sum(yanma_saati <= s))
        alan_ha = round(yanan_hucre * hucre_hektar, 1)
        byram_val = round(saatlik_max_byram[idx], 0)

        if fm_bilesik >= 24.0 or yanan_hucre <= 1:
            ort_hiz = 0.0
            risk = "Yok (Sönme Eşiği)"
        else:
            yaricap_m = math.sqrt((alan_ha * 10000.0) / math.pi)
            ort_hiz = round(yaricap_m / (s * 60.0), 1)
            if saatlik_tepe_aktif[idx]:
                risk = f"Kritik Tepe Yangını ({int(byram_val)} kW/m)"
            elif saatlik_gece_freni[idx]:
                risk = f"Gece VPD Freni ({saatlik_vpd[idx]} kPa)"
            elif byram_val >= 500.0:
                risk = f"Orta-Şiddetli Örtü ({int(byram_val)} kW/m)"
            else:
                risk = f"Düşük Örtü Yangını ({int(byram_val)} kW/m)"

        saatlik_ozet.append({
            "saat": s,
            "alan_hektar": alan_ha,
            "yayilim_hizi_m_dk": ort_hiz,
            "tepe_yangini_riski": risk,
            "byram_siddeti_kw_m": byram_val,
            "vpd_kpa": saatlik_vpd[idx]
        })

    return {
        "izokronlar": izokronlar,
        "saatlik_ozet": saatlik_ozet,
        "kuruluk_katsayisi": round(hava["anlik_vpd_kpa"], 2)
    }
