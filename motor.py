import math
import json
import xml.etree.ElementTree as ET
import requests
import numpy as np


# 1. SON 5 GÜNLÜK + ANLIK + 12 SAATLİK HAVA DURUMU (OPEN-METEO)
def hava_durumu_getir(enlem: float, boylam: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": enlem,
        "longitude": boylam,
        "past_days": 5,
        "forecast_days": 2,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        veri = r.json()

        saatlik = veri["hourly"]
        anlik = veri["current"]

        # İlk 120 saat = yangın öncesi son 5 gün
        gecmis_sicaklik = saatlik["temperature_2m"][:120]
        gecmis_nem = saatlik["relative_humidity_2m"][:120]
        gecmis_yagis = saatlik["precipitation"][:120]

        if not gecmis_nem:
            raise ValueError("Geçmiş hava verisi boş döndü")

        ort_sicaklik_5g = round(float(np.mean(gecmis_sicaklik)), 1)
        ort_nem_5g = round(float(np.mean(gecmis_nem)), 1)
        toplam_yagis_5g = round(float(np.sum(gecmis_yagis)), 1)

        # 100 Saatlik Ölü Yakıt Nemi (FM100) hesabı (son 5 güne ağırlıklı)
        agirliklar = np.linspace(0.5, 1.5, len(gecmis_nem))
        agirlikli_nem = float(np.average(gecmis_nem, weights=agirliklar))
        agirlikli_sicaklik = float(np.average(gecmis_sicaklik, weights=agirliklar))

        emc = 0.03229 + 0.281073 * agirlikli_nem - 0.000578 * agirlikli_nem * agirlikli_sicaklik
        yagis_etkisi = min(14.0, toplam_yagis_5g * 0.9)
        fm100 = round(float(np.clip(emc + yagis_etkisi, 4.0, 35.0)), 1)

        # Önümüzdeki 12 saatlik rüzgar tahmini
        gelecek_ruzgar_hiz = saatlik["wind_speed_10m"][120:132]
        gelecek_ruzgar_yon = saatlik["wind_direction_10m"][120:132]

        return {
            "anlik_sicaklik": round(float(anlik["temperature_2m"]), 1),
            "anlik_nem": round(float(anlik["relative_humidity_2m"]), 1),
            "anlik_ruzgar_hizi": round(float(anlik["wind_speed_10m"]), 1),
            "anlik_ruzgar_yonu": int(anlik["wind_direction_10m"]),
            "son_5_gun_ort_sicaklik": ort_sicaklik_5g,
            "son_5_gun_ort_nem": ort_nem_5g,
            "son_5_gun_yagis_mm": toplam_yagis_5g,
            "olu_yakit_nemi_100saat": fm100,
            "gelecek_12saat_ruzgar_hiz": [round(float(x), 1) for x in gelecek_ruzgar_hiz],
            "gelecek_12saat_ruzgar_yon": [int(x) for x in gelecek_ruzgar_yon],
            "kaynak": "Open-Meteo (Canlı + 5 Günlük Arşiv)"
        }
    except Exception:
        return {
            "anlik_sicaklik": 31.0,
            "anlik_nem": 28.0,
            "anlik_ruzgar_hizi": 22.0,
            "anlik_ruzgar_yonu": 45,
            "son_5_gun_ort_sicaklik": 29.5,
            "son_5_gun_ort_nem": 32.0,
            "son_5_gun_yagis_mm": 0.0,
            "olu_yakit_nemi_100saat": 9.8,
            "gelecek_12saat_ruzgar_hiz": [22.0] * 12,
            "gelecek_12saat_ruzgar_yon": [45] * 12,
            "kaynak": "Varsayılan Akdeniz Yaz Profili (Yedek)"
        }


# 2. ARAZİ YAPISI - COPERNICUS DEM (10x10 KM IZGARA)
def arazi_dem_getir(enlem: float, boylam: float, boyut: int = 25) -> dict:
    yari_cap = 0.045
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

    hucre_m = 400.0
    dy, dx = np.gradient(elev_grid, hucre_m)
    egim_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    egim_derece = np.degrees(egim_rad)
    baki_derece = (np.degrees(np.arctan2(-dx, dy)) + 360.0) % 360.0

    merkez = boyut // 2
    merkez_baki = float(baki_derece[merkez, merkez])
    yonler = ["Kuzey", "Kuzeydoğu", "Doğu", "Güneydoğu", "Güney", "Güneybatı", "Batı", "Kuzeybatı"]
    baki_adi = yonler[int((merkez_baki + 22.5) // 45) % 8]

    return {
        "yukselti_grid": elev_grid,
        "egim_grid": egim_derece,
        "baki_grid": baki_derece,
        "merkez_rakim_m": int(round(float(elev_grid[merkez, merkez]))),
        "ortalama_egim_derece": round(float(np.mean(egim_derece)), 1),
        "merkez_baki": baki_adi,
        "kaynak": "Copernicus DEM (Open-Meteo Elevation)"
    }


# 3. BİTKİ ÖRTÜSÜ VE DOĞAL/YAPAY ENGELLER (OPENSTREETMAP + EKOLOJİK MODEL)
def bitki_ve_engeller_getir(enlem: float, boylam: float, elev_grid: np.ndarray) -> dict:
    boyut = elev_grid.shape[0]
    yari_cap = 0.045
    min_lat, max_lat = enlem - yari_cap, enlem + yari_cap
    min_lon, max_lon = boylam - yari_cap, boylam + yari_cap

    # Yakıt sınıfları: 1=Kızılçam/İğne, 2=Maki/Çalı, 3=Karışık/Geniş, 4=Tarım/Otluk, 0=Yanmaz Engel
    yakit_grid = np.where(elev_grid < 850, 1, np.where(elev_grid < 1550, 3, 2))
    engel_sayisi = 0
    osm_basarili = False

    query = f"""
    [out:json][timeout:12];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|track"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["waterway"~"river|stream|canal"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["natural"~"water|wood|scrub|bare_rock"]({min_lat},{min_lon},{max_lat},{max_lon});
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
                if "highway" in tags or "waterway" in tags or tags.get("natural") in ("water", "bare_rock") or tags.get("landuse") == "residential":
                    sinif = 0
                    engel_sayisi += 1
                elif tags.get("natural") == "scrub":
                    sinif = 2
                elif tags.get("landuse") in ("farmland", "orchard"):
                    sinif = 4
                elif tags.get("natural") == "wood" or tags.get("landuse") == "forest":
                    sinif = 3 if tags.get("leaf_type") == "broadleaved" else 1

                if sinif is not None:
                    for nid in el.get("nodes", []):
                        if nid in nodes:
                            la, lo = nodes[nid]
                            ri = int((la - min_lat) / (max_lat - min_lat) * (boyut - 1))
                            ci = int((lo - min_lon) / (max_lon - min_lon) * (boyut - 1))
                            if 0 <= ri < boyut and 0 <= ci < boyut:
                                yakit_grid[ri, ci] = sinif
            osm_basarili = True
    except Exception:
        pass

    sinif_adlari = {
        1: "Kızılçam / İğne Yapraklı Orman",
        2: "Maki ve Sık Çalılık",
        3: "Karışık / Geniş Yapraklı Orman",
        4: "Zeytinlik / Tarım ve Otluk",
        0: "Yol / Su / Yanmaz Engel"
    }
    degerler, sayilar = np.unique(yakit_grid[yakit_grid > 0], return_counts=True)
    baskin_kod = int(degerler[np.argmax(sayilar)]) if len(degerler) > 0 else 1

    return {
        "yakit_grid": yakit_grid,
        "baskin_bitki_ortusu": sinif_adlari[baskin_kod],
        "tespit_edilen_engel_hatti": engel_sayisi,
        "kaynak": "OpenStreetMap + Akdeniz Ekolojik Katmanı" if osm_basarili else "Akdeniz Ekolojik Yükselti Modeli"
    }


# 4. İSTEĞE BAĞLI KML / GEOJSON KATMAN ENTEGRASYONU
def ozel_katman_uygula(yakit_grid: np.ndarray, enlem: float, boylam: float, dosya_icerik: str) -> tuple[np.ndarray, int]:
    boyut = yakit_grid.shape[0]
    yari_cap = 0.045
    min_lat, max_lat = enlem - yari_cap, enlem + yari_cap
    min_lon, max_lon = boylam - yari_cap, boylam + yari_cap
    islenen_nesne = 0

    def koordinat_isle(lon_val: float, lat_val: float, sinif_kodu: int):
        ri = int((lat_val - min_lat) / (max_lat - min_lat) * (boyut - 1))
        ci = int((lon_val - min_lon) / (max_lon - min_lon) * (boyut - 1))
        if 0 <= ri < boyut and 0 <= ci < boyut:
            yakit_grid[ri, ci] = sinif_kodu

    metin = dosya_icerik.strip()
    if metin.startswith("{"):
        geo = json.loads(metin)
        for feat in geo.get("features", []):
            props = str(feat.get("properties", {})).lower()
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            sinif = 0 if ("yol" in props or "emniyet" in props or "dere" in props or gtype == "LineString") else (2 if "maki" in props else 1)
            coords = geom.get("coordinates", [])
            if gtype == "LineString":
                for c in coords:
                    koordinat_isle(c[0], c[1], sinif)
                islenen_nesne += 1
            elif gtype == "Polygon" and coords:
                for c in coords[0]:
                    koordinat_isle(c[0], c[1], sinif)
                islenen_nesne += 1
    elif "<kml" in metin.lower():
        root = ET.fromstring(metin)
        for elem in root.iter():
            if elem.tag.endswith("coordinates") and elem.text:
                parcalar = elem.text.strip().split()
                sinif = 0 if len(parcalar) > 2 else 1
                for p in parcalar:
                    xyz = p.split(",")
                    if len(xyz) >= 2:
                        koordinat_isle(float(xyz[0]), float(xyz[1]), sinif)
                islenen_nesne += 1

    return yakit_grid, islenen_nesne


# 5. ROTHERMEL + HÜCRESEL OTOMAT YANGIN YAYILIM SİMÜLASYONU (0 - 12 SAAT)
def yangin_simule_et(enlem: float, boylam: float, hava: dict, dem: dict, bitki: dict) -> dict:
    boyut = 25
    hucre_m = 400.0
    merkez = boyut // 2

    yakit = bitki["yakit_grid"]
    elev = dem["yukselti_grid"]
    fm100 = hava["olu_yakit_nemi_100saat"]

    temel_hiz = {0: 0.0, 1: 14.5, 2: 11.0, 3: 6.5, 4: 8.5}
    nem_katsayisi = float(np.clip(1.65 - (fm100 / 22.0), 0.25, 1.60))

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

    ziyaret = np.zeros((boyut, boyut), dtype=bool)
    for _ in range(boyut * boyut):
        adaylar = np.where(~ziyaret, yanma_saati, 999.0)
        r, c = np.unravel_index(np.argmin(adaylar), adaylar.shape)
        mevcut_t = yanma_saati[r, c]
        if mevcut_t >= 12.0:
            break
        ziyaret[r, c] = True

        saat_indeks = min(11, int(mevcut_t))
        ruzgar_hiz = hava["gelecek_12saat_ruzgar_hiz"][saat_indeks]
        ruzgar_gidis_yon = (hava["gelecek_12saat_ruzgar_yon"][saat_indeks] + 180.0) % 360.0

        for dr, dc, mesafe, yon_azimut in komsular:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < boyut and 0 <= nc < boyut) or ziyaret[nr, nc]:
                continue

            hedef_yakit = int(yakit[nr, nc])
            r0 = temel_hiz.get(hedef_yakit, 8.0)

            if hedef_yakit == 0:
                if ruzgar_hiz >= 32.0 and fm100 < 12.0:
                    r0 = 2.2
                else:
                    continue

            aci_fark = math.radians(yon_azimut - ruzgar_gidis_yon)
            ruzgar_etkisi = 1.0 + (ruzgar_hiz / 18.0) * max(-0.4, math.cos(aci_fark))

            kot_farki = elev[nr, nc] - elev[r, c]
            yerel_egim_tan = max(0.0, kot_farki / mesafe)
            egim_etkisi = 1.0 + 5.2 * (yerel_egim_tan ** 1.4)

            yayilim_hizi_m_dk = max(0.4, r0 * nem_katsayisi * ruzgar_etkisi * egim_etkisi)
            gecis_saati = (mesafe / yayilim_hizi_m_dk) / 60.0
            yeni_t = mevcut_t + gecis_saati

            if yeni_t < yanma_saati[nr, nc]:
                yanma_saati[nr, nc] = yeni_t

    yari_cap = 0.045
    min_lat, max_lat = enlem - yari_cap, enlem + yari_cap
    min_lon, max_lon = boylam - yari_cap, boylam + yari_cap

    def saat_poligonu_uret(hedef_saat: float) -> list:
        yanan_r, yanan_c = np.where(yanma_saati <= hedef_saat)
        if len(yanan_r) < 3:
            d = 0.0025 * hedef_saat
            return [[
                [round(boylam + d * math.cos(math.radians(a)), 5), round(enlem + d * math.sin(math.radians(a)), 5)]
                for a in range(0, 361, 15)
            ]]

        sektor_sayisi = 36
        max_uzaklik = np.full(sektor_sayisi, 0.0018)
        for rr, cc in zip(yanan_r, yanan_c):
            la = min_lat + (rr / (boyut - 1)) * (max_lat - min_lat)
            lo = min_lon + (cc / (boyut - 1)) * (max_lon - min_lon)
            dlat = la - enlem
            dlon = lo - boylam
            mes = math.hypot(dlat, dlon) + 0.0018
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
        alan_hektar = round(yanan_hucre * 16.0, 1)
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
        alan_ha = round(yanan_hucre * 16.0, 1)
        yaricap_m = math.sqrt((alan_ha * 10000.0) / math.pi)
        ort_hiz = round(yaricap_m / (s * 60.0), 1)
        risk = "Yüksek (Tepe Yangını Riski)" if (fm100 < 12.0 and hava["anlik_ruzgar_hizi"] > 20.0) else ("Orta" if fm100 < 18.0 else "Düşük")
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
