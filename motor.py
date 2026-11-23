import re
import math
import heapq
import json
import xml.etree.ElementTree as ET
import requests
import numpy as np


def _doymus_buhar_basinci(sicaklik_c: float) -> float:
    return 6.1078 * math.exp((17.27 * sicaklik_c) / (sicaklik_c + 237.3))


def _denge_nemi_emc(yakit_sicaklik_c: float, yakit_bagil_nem: float) -> float:
    rh = max(1.0, min(100.0, yakit_bagil_nem))
    tf = yakit_sicaklik_c
    if rh < 10.0:
        emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * tf
    elif rh < 50.0:
        emc = 2.22749 + 0.160107 * rh - 0.01478 * tf
    else:
        emc = 21.0606 + 0.005565 * (rh ** 2) - 0.00035 * rh * tf - 0.483199 * rh
    return max(2.0, min(35.0, emc))


# 1. SON 10 GÜNLÜK YAĞIŞ + GÜNEŞLENME + ANLIK VE 12 SAATLİK HAVA DURUMU
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
            "wind_speed_10m,wind_direction_10m"
        ),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
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

        fm1, fm10, fm100, fm240 = 14.0, 15.0, 16.0, 18.0

        for i in range(gecmis_saat):
            t_air = float(t_arr[i] or 18.0)
            rh_air = float(rh_arr[i] or 50.0)
            yagis = float(p_arr[i] or 0.0)
            radyasyon = float(rad_arr[i] or 0.0)
            et0 = float(et0_arr[i] or 0.0)
            ruzgar = float(w_arr[i] or 10.0)

            t_fuel = t_air + (0.024 * radyasyon) / (1.0 + 0.12 * ruzgar)
            es_air = _doymus_buhar_basinci(t_air)
            es_fuel = _doymus_buhar_basinci(t_fuel)
            rh_fuel = rh_air * (es_air / max(0.1, es_fuel))
            emc = _denge_nemi_emc(t_fuel, rh_fuel)

            if yagis > 0.1:
                fm1 = min(45.0, fm1 + min(18.0, 4.5 + yagis * 3.2))
                fm10 = min(40.0, fm10 + min(10.0, 2.0 + yagis * 1.6))
                fm100 = min(36.0, fm100 + min(5.0, 0.8 + yagis * 0.65))
                fm240 = min(45.0, fm240 + min(3.5, 0.5 + yagis * 0.45))
            else:
                radyasyon_carpan = 1.0 + (radyasyon / 900.0) * 0.45 + (et0 * 0.4)
                fm1 = emc + (fm1 - emc) * math.exp(-min(0.95, radyasyon_carpan))
                fm10 = emc + (fm10 - emc) * math.exp(-radyasyon_carpan / 10.0)
                fm100 = emc + (fm100 - emc) * math.exp(-radyasyon_carpan / 100.0)
                fm240 = emc + (fm240 - emc) * math.exp(-radyasyon_carpan / 240.0)

        bilesik_yakit_nemi = round(0.40 * fm1 + 0.30 * fm10 + 0.20 * fm100 + 0.10 * fm240, 1)

        ort_sicaklik_10g = round(float(np.mean([x for x in t_arr if x is not None])), 1)
        ort_nem_10g = round(float(np.mean([x for x in rh_arr if x is not None])), 1)
        toplam_yagis_10g = round(float(np.sum([x for x in p_arr if x is not None])), 1)
        toplam_gunes_saat = round(float(np.sum([x for x in sun_arr if x is not None])) / 3600.0, 1)
        ort_radyasyon = round(float(np.mean([x for x in rad_arr if x is not None])), 0)

        gelecek_ruzgar_hiz = [round(float(x or 10.0), 1) for x in saatlik["wind_speed_10m"][gecmis_saat:gecmis_saat + 12]]
        gelecek_ruzgar_yon = [int(x or 0) for x in saatlik["wind_direction_10m"][gecmis_saat:gecmis_saat + 12]]

        while len(gelecek_ruzgar_hiz) < 12:
            gelecek_ruzgar_hiz.append(round(float(anlik["wind_speed_10m"]), 1))
            gelecek_ruzgar_yon.append(int(anlik["wind_direction_10m"]))

        return {
            "anlik_sicaklik": round(float(anlik["temperature_2m"]), 1),
            "anlik_nem": round(float(anlik["relative_humidity_2m"]), 1),
            "anlik_ruzgar_hizi": round(float(anlik["wind_speed_10m"]), 1),
            "anlik_ruzgar_yonu": int(anlik["wind_direction_10m"]),
            "son_5_gun_ort_sicaklik": ort_sicaklik_10g,
            "son_5_gun_ort_nem": ort_nem_10g,
            "son_5_gun_yagis_mm": toplam_yagis_10g,
            "son_10_gun_guneslenme_saat": toplam_gunes_saat,
            "son_10_gun_ort_radyasyon_wm2": ort_radyasyon,
            "olu_yakit_nemi_100saat": bilesik_yakit_nemi,
            "gelecek_12saat_ruzgar_hiz": gelecek_ruzgar_hiz,
            "gelecek_12saat_ruzgar_yon": gelecek_ruzgar_yon,
            "kaynak": "Open-Meteo (10 Günlük Yağış & Güneş Radyasyonu + Canlı)"
        }
    except Exception:
        return {
            "anlik_sicaklik": 24.0,
            "anlik_nem": 42.0,
            "anlik_ruzgar_hizi": 16.0,
            "anlik_ruzgar_yonu": 45,
            "son_5_gun_ort_sicaklik": 22.0,
            "son_5_gun_ort_nem": 46.0,
            "son_5_gun_yagis_mm": 2.0,
            "son_10_gun_guneslenme_saat": 68.0,
            "son_10_gun_ort_radyasyon_wm2": 420.0,
            "olu_yakit_nemi_100saat": 14.5,
            "gelecek_12saat_ruzgar_hiz": [16.0] * 12,
            "gelecek_12saat_ruzgar_yon": [45] * 12,
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


def _cizgi_izgaraya_isle(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, deger, carpan_grid: np.ndarray = None, carpan: float = 1.0):
    adim = max(abs(r1 - r0), abs(c1 - c0), 1)
    for k in range(adim + 1):
        rr = int(round(r0 + (r1 - r0) * (k / adim)))
        cc = int(round(c0 + (c1 - c0) * (k / adim)))
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = deger
            if carpan_grid is not None:
                carpan_grid[rr, cc] = carpan


def _poligon_icini_doldur(grid: np.ndarray, halka: list[tuple[int, int]], deger: int, carpan_grid: np.ndarray, carpan: float):
    """Kapalı meşcere poligonunun hem sınırlarını hem de tüm iç hücrelerini doldurur."""
    if len(halka) < 3:
        return
    for idx in range(len(halka) - 1):
        _cizgi_izgaraya_isle(grid, halka[idx][0], halka[idx][1], halka[idx + 1][0], halka[idx + 1][1], deger, carpan_grid, carpan)

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


# 3. AKDENİZ VE EGE ORTALAMASI: KIZILÇAM + MAKİ TABAN MODELİ & ENGELLER
def bitki_ve_engeller_getir(enlem: float, boylam: float, dem: dict) -> dict:
    elev_grid = dem["yukselti_grid"]
    baki_grid = dem["baki_grid"]
    boyut = elev_grid.shape[0]
    yari_cap = dem["yari_cap"]
    min_lat, max_lat = enlem - yari_cap, enlem + yari_cap
    min_lon, max_lon = boylam - yari_cap, boylam + yari_cap

    # Yakıt sınıfları:
    # 1 = Kızılçam + Maki Alt Örtülü Orman (Akdeniz-Ege 0-950m baskın)
    # 2 = Saf Maki ve Sık Çalılık (Kıyı ve güney bakılı kurak yamaçlar)
    # 3 = Karaçam / Karışık Orman (950m üzeri yüksek kesimler)
    # 4 = Zeytinlik / Tarım ve Açıklık
    # 0 = Su / Kıyı / Yol / Yanmaz Engel
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
                elif tags.get("landuse") in ("farmland", "orchard"):
                    sinif = 4
                elif tags.get("natural") == "wood" or tags.get("landuse") == "forest":
                    sinif = 3 if tags.get("leaf_type") == "broadleaved" else 1

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
                        _cizgi_izgaraya_isle(yakit_grid, r0, c0, r1, c1, sinif)
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
        "baskin_bitki_ortusu": baskin_ad,
        "tespit_edilen_engel_hatti": engel_sayisi,
        "kaynak": "Akdeniz-Ege Kızılçam/Maki Modeli + OSM" if osm_basarili else "Akdeniz-Ege Kızılçam/Maki Ekolojik Modeli"
    }


def _mescere_rumuzu_coz(etiket_metni: str, varsayilan_poligon: bool = True) -> tuple[int, float, str]:
    """
    KML/GeoJSON içindeki meşcere tipi, gelişme çağı (yaş sınıfı: a, b, c, d) ve kapalılık (1, 2, 3)
    bilgilerini okuyarak yakıt sınıfını ve hız çarpanını döndürür.
    """
    temiz = re.sub(r"<[^>]+>", " ", etiket_metni).lower()

    # 1. Yol / Yangın Emniyet Yolu / Dere / Su / Kayalık
    if any(k in temiz for k in ["yey", "emniyet", "orman yolu", "yol", "dere", "akarsu", "gölet", "baraj", "su", "kaya", "iskân", "iskan"]):
        return 0, 0.0, "Engel/Yol"

    # 2. Meşcere Rumuzu Eşleştirme (Örn: Çza3, Çzbc2, Çzd3, M, Mb, Çkb2, OT, Z)
    eslesme = re.search(r"\b(çz|cz|çk|ck|mb|md|m|bçz|bçk|kz|kn|me|z|ot|a|t)([abcd]{1,2})?([0123])?\b", temiz)
    if eslesme:
        tur = eslesme.group(1)
        cag = eslesme.group(2) or ""
        kapalilik = eslesme.group(3) or ""

        if tur in ("çz", "cz", "bçz", "kz"):
            sinif = 1  # Kızılçam
        elif tur in ("m", "mb", "md"):
            sinif = 2  # Maki
        elif tur in ("çk", "ck", "bçk", "kn", "me"):
            sinif = 3  # Karaçam / Geniş Yapraklı
        else:
            sinif = 4  # Zeytinlik / Açıklık / Ziraat

        # Yaş sınıfı / Gelişme çağı katsayısı
        if "a" in cag and "b" not in cag:
            cag_carpan = 1.35  # Gençlik / Sıklık çağı (en hızlı ve tepeye açık)
        elif "a" in cag and "b" in cag:
            cag_carpan = 1.28
        elif "b" in cag and "c" not in cag:
            cag_carpan = 1.20  # Sırıklık çağı
        elif "b" in cag and "c" in cag:
            cag_carpan = 1.10
        elif "c" in cag and "d" not in cag:
            cag_carpan = 1.00  # Direklik çağı
        elif "d" in cag:
            cag_carpan = 0.85  # Ağaçlık / Yaşlı orman
        else:
            cag_carpan = 0.85 if tur == "mb" else 1.00

        # Kapalılık katsayısı (1, 2, 3)
        if kapalilik == "3":
            kap_carpan = 1.15
        elif kapalilik == "1":
            kap_carpan = 0.85
        elif kapalilik == "0":
            kap_carpan = 0.65
        else:
            kap_carpan = 1.00

        kod_ozet = f"{tur.upper()}{cag}{kapalilik}"
        return sinif, round(cag_carpan * kap_carpan, 2), kod_ozet

    # 3. Açık metin kelimeleri
    if "maki" in temiz or "çalı" in temiz or "cali" in temiz:
        return 2, 1.0, "Maki"
    if "gençlik" in temiz or "sıklık" in temiz:
        return 1, 1.35, "Gençlik(a)"
    if "sırıklık" in temiz:
        return 1, 1.20, "Sırıklık(b)"
    if "ağaçlık" in temiz or "yaşlı" in temiz:
        return 1, 0.85, "Ağaçlık(d)"
    if "zeytin" in temiz or "tarım" in temiz or "açıklık" in temiz:
        return 4, 0.80, "Zeytin/Açıklık"

    return (1, 1.0, "Kızılçam") if varsayilan_poligon else (0, 0.0, "Hat")


# 4. İSTEĞE BAĞLI KML / GEOJSON MEŞCERE VE YOL KATMAN ENTEGRASYONU
def ozel_katman_uygula(yakit_grid: np.ndarray, enlem: float, boylam: float, yari_cap: float, dosya_icerik: str, carpan_grid: np.ndarray = None) -> tuple[np.ndarray, int]:
    boyut = yakit_grid.shape[0]
    if carpan_grid is None:
        carpan_grid = np.ones((boyut, boyut), dtype=float)

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
                sinif, carpan, _ = _mescere_rumuzu_coz(props_str, varsayilan_poligon=False)
                for idx in range(len(coords) - 1):
                    r0, c0 = koordinat_noktasi(coords[idx][0], coords[idx][1])
                    r1, c1 = koordinat_noktasi(coords[idx + 1][0], coords[idx + 1][1])
                    _cizgi_izgaraya_isle(yakit_grid, r0, c0, r1, c1, sinif, carpan_grid, carpan)
                islenen_nesne += 1
            elif gtype == "Polygon" and coords:
                sinif, carpan, _ = _mescere_rumuzu_coz(props_str, varsayilan_poligon=True)
                halka = [koordinat_noktasi(c[0], c[1]) for c in coords[0]]
                _poligon_icini_doldur(yakit_grid, halka, sinif, carpan_grid, carpan)
                islenen_nesne += 1

    elif "<kml" in metin.lower():
        root = ET.fromstring(metin)
        for pm in root.iter():
            if not pm.tag.endswith("Placemark"):
                continue

            # Placemark içindeki name, description ve ExtendedData metinlerini birleştir
            etiket_parcalari = []
            for alt in pm.iter():
                if alt.tag.endswith(("name", "description", "value", "SimpleData")) and alt.text:
                    etiket_parcalari.append(alt.text.strip())
            tum_etiket = " ".join(etiket_parcalari)

            poligon_var = any(alt.tag.endswith("Polygon") for alt in pm.iter())
            sinif, carpan, _ = _mescere_rumuzu_coz(tum_etiket, varsayilan_poligon=poligon_var)

            for alt in pm.iter():
                if alt.tag.endswith("coordinates") and alt.text:
                    parcalar = alt.text.strip().split()
                    noktalar = []
                    for p in parcalar:
                        xyz = p.split(",")
                        if len(xyz) >= 2:
                            noktalar.append(koordinat_noktasi(float(xyz[0]), float(xyz[1])))
                    if len(noktalar) >= 3 and poligon_var:
                        _poligon_icini_doldur(yakit_grid, noktalar, sinif, carpan_grid, carpan)
                        islenen_nesne += 1
                    elif len(noktalar) >= 2:
                        for idx in range(len(noktalar) - 1):
                            _cizgi_izgaraya_isle(yakit_grid, noktalar[idx][0], noktalar[idx][1], noktalar[idx + 1][0], noktalar[idx + 1][1], sinif, carpan_grid, carpan)
                        islenen_nesne += 1

    return yakit_grid, islenen_nesne


# 5. FİZİK TABANLI YANGIN YAYILIM SİMÜLASYONU
def yangin_simule_et(enlem: float, boylam: float, hava: dict, dem: dict, bitki: dict) -> dict:
    yakit = bitki["yakit_grid"]
    carpan_grid = bitki.get("yakit_carpan_grid", np.ones_like(yakit, dtype=float))
    elev = dem["yukselti_grid"]
    baki = dem["baki_grid"]
    tpi = dem["tpi_grid"]
    boyut = yakit.shape[0]
    hucre_m = dem["hucre_m"]
    yari_cap = dem["yari_cap"]
    merkez = boyut // 2

    fm_bilesik = hava["olu_yakit_nemi_100saat"]
    anlik_sicaklik = hava["anlik_sicaklik"]
    radyasyon_ort = hava.get("son_10_gun_ort_radyasyon_wm2", 400.0)

    sonme_nemi = 24.0
    if fm_bilesik >= sonme_nemi:
        nem_katsayisi = 0.0
    else:
        nem_orani = fm_bilesik / sonme_nemi
        nem_katsayisi = float(np.clip(1.65 * ((1.0 - nem_orani) ** 1.35), 0.04, 1.65))

    if anlik_sicaklik < 14.0:
        sicaklik_freni = float(np.clip(0.35 + (anlik_sicaklik / 22.0), 0.25, 1.0))
        nem_katsayisi *= sicaklik_freni

    # Akdeniz-Ege kalibrasyonlu temel yayılım hızları (m/dk):
    # 1: Kızılçam + Maki (16.0), 2: Saf Maki (13.5), 3: Karaçam/Karışık (9.0), 4: Zeytin/Tarım (5.5)
    temel_hiz = {0: 0.0, 1: 16.0, 2: 13.5, 3: 9.0, 4: 5.5}

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

    if nem_katsayisi > 0.01:
        kuyruk = [(0.0, merkez, merkez)]
        while kuyruk:
            mevcut_t, r, c = heapq.heappop(kuyruk)
            if mevcut_t > yanma_saati[r, c] or mevcut_t >= 12.0:
                continue

            saat_indeks = min(11, int(mevcut_t))
            ruzgar_hiz = hava["gelecek_12saat_ruzgar_hiz"][saat_indeks]
            ruzgar_gidis_yon = (hava["gelecek_12saat_ruzgar_yon"][saat_indeks] + 180.0) % 360.0

            for dr, dc, mesafe, yon_azimut in komsular:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < boyut and 0 <= nc < boyut):
                    continue

                hedef_yakit = int(yakit[nr, nc])
                r0 = temel_hiz.get(hedef_yakit, 10.0) * float(carpan_grid[nr, nc])

                if hedef_yakit == 0:
                    if elev[nr, nc] <= 1.0:
                        continue
                    # Kızılçam kozalak atlama (spotting): şiddetli rüzgar ve kuru yakıtta yol/dere atlama
                    if ruzgar_hiz >= 32.0 and fm_bilesik < 11.5 and int(yakit[r, c]) in (1, 2):
                        r0 = 2.1
                    else:
                        continue

                aci_fark = math.radians(yon_azimut - ruzgar_gidis_yon)
                ruzgar_etkisi = 1.0 + (ruzgar_hiz / 17.5) * max(-0.45, math.cos(aci_fark))

                kot_farki = elev[nr, nc] - elev[r, c]
                egim_tan = kot_farki / mesafe
                if egim_tan >= 0.0:
                    egim_etkisi = 1.0 + 5.2 * (egim_tan ** 1.35)
                else:
                    egim_etkisi = max(0.35, 1.0 + 1.8 * egim_tan)

                hedef_tpi = float(tpi[nr, nc])
                if hedef_tpi < -12.0 and egim_tan > 0.05:
                    topo_etkisi = min(1.85, 1.0 + abs(hedef_tpi) / 35.0)
                elif hedef_tpi > 18.0 and egim_tan < 0.0:
                    topo_etkisi = 0.65
                else:
                    topo_etkisi = 1.0

                baki_rad = math.radians(float(baki[nr, nc]))
                gunes_faktoru = min(1.0, radyasyon_ort / 600.0)
                baki_etkisi = 1.0 - 0.22 * math.cos(baki_rad) * gunes_faktoru

                yayilim_hizi_m_dk = max(0.2, r0 * nem_katsayisi * ruzgar_etkisi * egim_etkisi * topo_etkisi * baki_etkisi)
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
        yanan_hucre = int(np.sum(yanma_saati <= s))
        alan_ha = round(yanan_hucre * hucre_hektar, 1)
        if fm_bilesik >= sonme_nemi or yanan_hucre <= 1:
            ort_hiz = 0.0
            risk = "Yok (Yakıt Islak / Sönme Eşiği)"
        else:
            yaricap_m = math.sqrt((alan_ha * 10000.0) / math.pi)
            ort_hiz = round(yaricap_m / (s * 60.0), 1)
            if fm_bilesik < 11.5 and hava["anlik_ruzgar_hizi"] > 20.0 and anlik_sicaklik >= 22.0:
                risk = "Yüksek (Tepe Yangını Riski)"
            elif fm_bilesik < 17.5:
                risk = "Orta (Kızılçam/Maki Örtü Yangını)"
            else:
                risk = "Düşük (Yavaş Örtü Yangını)"

        saatlik_ozet.append({
            "saat": s,
            "alan_hektar": alan_ha,
            "yayilim_hizi_m_dk": ort_hiz,
            "tepe_yangini_riski": risk
        })

    return {
        "izokronlar": izokronlar,
        "saatlik_ozet": saatlik_ozet,
        "kuruluk_katsayisi": round(nem_katsayisi, 2)
    }
