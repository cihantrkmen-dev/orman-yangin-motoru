import re
import math
import heapq
import json
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET
import requests
import numpy as np


def _doymus_buhar_basinci_kpa(sicaklik_c: float) -> float:
    """Magnus-Tetens formülü ile doymuş su buharı basıncı (kPa)."""
    return 0.61078 * math.exp((17.27 * sicaklik_c) / (sicaklik_c + 237.3))


def _buhar_basinci_acigi_vpd(sicaklik_c: float, bagil_nem: float) -> float:
    """Buhar Basıncı Açığı (VPD - kPa)."""
    es = _doymus_buhar_basinci_kpa(sicaklik_c)
    rh = max(1.0, min(100.0, bagil_nem))
    return max(0.0, es * (1.0 - rh / 100.0))


def _denge_nemi_emc(yakit_sicaklik_c: float, yakit_bagil_nem: float, kuruma_fazi: bool = True) -> float:
    """
    Simard (1968) denklemi + Viney (1991) / Nelson (2000) Yakıt Nemi Histerezisi.
    Kuruma (desorpsiyon) eğrisi, ıslanma (adsorpsiyon) eğrisinden ~%1.8 daha yüksek denge nemi tutar.
    """
    rh = max(1.0, min(100.0, yakit_bagil_nem))
    tf = yakit_sicaklik_c
    if rh < 10.0:
        emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * tf
    elif rh < 50.0:
        emc = 2.22749 + 0.160107 * rh - 0.01478 * tf
    else:
        emc = 21.0606 + 0.005565 * (rh ** 2) - 0.00035 * rh * tf - 0.483199 * rh

    if kuruma_fazi:
        emc += 1.8
    return max(2.0, min(35.0, emc))


def _astronomik_gunes_durumu(enlem: float, boylam: float, utc_zaman: datetime) -> tuple[int, float]:
    """Koordinat ve UTC zamana göre gerçek gece/gündüz (is_day) ve güneş radyasyonu (W/m2)."""
    gun_no = utc_zaman.timetuple().tm_yday
    deklinasyon = math.radians(23.44 * math.sin(math.radians((360.0 / 365.0) * (gun_no - 81))))
    yerel_gunes_saati = (utc_zaman.hour + utc_zaman.minute / 60.0 + boylam / 15.0) % 24.0
    saat_acisi = math.radians(15.0 * (yerel_gunes_saati - 12.0))
    enlem_rad = math.radians(enlem)

    sin_yukseklik = (
        math.sin(enlem_rad) * math.sin(deklinasyon)
        + math.cos(enlem_rad) * math.cos(deklinasyon) * math.cos(saat_acisi)
    )
    if sin_yukseklik > 0.02:
        radyasyon = round(920.0 * (sin_yukseklik ** 1.15), 1)
        return 1, radyasyon
    return 0, 0.0


def _saatlik_nem_adimi(
    fm1: float, fm10: float, fm100: float, fm240: float,
    t_air: float, rh_air: float, yagis: float, radyasyon: float, ruzgar: float
) -> tuple[float, float, float, float]:
    """Tek bir saat dilimi için histerezisli 4 katmanlı ölü yakıt nemi güncellemesi."""
    if radyasyon > 40.0:
        t_fuel = t_air + (0.020 * radyasyon) / (1.0 + 0.14 * ruzgar)
    else:
        t_fuel = t_air - 0.6

    es_air = _doymus_buhar_basinci_kpa(t_air)
    es_fuel = _doymus_buhar_basinci_kpa(t_fuel)
    rh_fuel = min(98.0, rh_air * (es_air / max(0.05, es_fuel)))

    emc_ham = _denge_nemi_emc(t_fuel, rh_fuel, kuruma_fazi=False)
    kuruma_mi = emc_ham < fm1
    emc = _denge_nemi_emc(t_fuel, rh_fuel, kuruma_fazi=kuruma_mi)

    if rh_air >= 76.0 and radyasyon < 60.0:
        emc = min(32.0, emc + (rh_air - 74.0) * 0.15)

    if yagis > 0.15:
        fm1 = min(42.0, fm1 + min(16.0, 4.5 + yagis * 3.2))
        fm10 = min(38.0, fm10 + min(9.0, 2.0 + yagis * 1.5))
        fm100 = min(34.0, fm100 + min(4.5, 0.8 + yagis * 0.6))
        fm240 = min(40.0, fm240 + min(3.0, 0.5 + yagis * 0.4))
    elif not kuruma_mi:
        # Islanma / nem geri kazanımı (adsorption)
        fm1 = emc + (fm1 - emc) * math.exp(-0.52)
        fm10 = emc + (fm10 - emc) * math.exp(-0.07)
        fm100 = emc + (fm100 - emc) * math.exp(-0.01)
        fm240 = emc + (fm240 - emc) * math.exp(-0.004)
    else:
        # Kuruma fazı (desorption)
        radyasyon_carpan = 0.88 + (radyasyon / 900.0) * 0.50
        fm1 = emc + (fm1 - emc) * math.exp(-min(0.86, radyasyon_carpan))
        fm10 = emc + (fm10 - emc) * math.exp(-radyasyon_carpan / 10.0)
        fm100 = emc + (fm100 - emc) * math.exp(-radyasyon_carpan / 100.0)
        fm240 = emc + (fm240 - emc) * math.exp(-radyasyon_carpan / 240.0)

    return fm1, fm10, fm100, fm240


# 1. GEÇMİŞ KURUMA + SAATLİK DİNAMİK 12 SAATLİK METEOROLOJİ VE YAKIT NEMİ MODELİ
def hava_durumu_getir(enlem: float, boylam: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params_tam = {
        "latitude": round(enlem, 4),
        "longitude": round(boylam, 4),
        "past_days": 10,
        "forecast_days": 2,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,shortwave_radiation,wind_speed_10m,wind_direction_10m,is_day",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,is_day",
        "timezone": "auto"
    }
    params_hafif = {
        "latitude": round(enlem, 4),
        "longitude": round(boylam, 4),
        "past_days": 5,
        "forecast_days": 2,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,shortwave_radiation,wind_speed_10m,wind_direction_10m,is_day",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,is_day",
        "timezone": "auto"
    }

    veri = None
    for prm in (params_tam, params_hafif):
        try:
            r = requests.get(url, params=prm, timeout=10)
            if r.status_code == 200:
                veri = r.json()
                break
        except Exception:
            continue

    simdi_utc = datetime.now(timezone.utc)

    if veri and "hourly" in veri and "current" in veri:
        saatlik = veri["hourly"]
        anlik = veri["current"]

        anlik_zaman = anlik.get("time", "")
        saat_listesi = saatlik.get("time", [])
        baslangic_idx = 0
        if anlik_zaman and saat_listesi:
            saat_on_eki = anlik_zaman[:13]
            for idx, t_str in enumerate(saat_listesi):
                if t_str.startswith(saat_on_eki):
                    baslangic_idx = idx
                    break
        if baslangic_idx == 0:
            baslangic_idx = max(0, len(saat_listesi) - 36)

        gecmis_saat = baslangic_idx
        t_arr = saatlik["temperature_2m"][:gecmis_saat]
        rh_arr = saatlik["relative_humidity_2m"][:gecmis_saat]
        p_arr = saatlik["precipitation"][:gecmis_saat]
        rad_arr = saatlik.get("shortwave_radiation", [320.0] * gecmis_saat)[:gecmis_saat]
        w_arr = saatlik["wind_speed_10m"][:gecmis_saat]

        fm1, fm10, fm100, fm240 = 14.0, 15.0, 16.0, 17.0
        for i in range(len(t_arr)):
            t_air = float(t_arr[i] if t_arr[i] is not None else 18.0)
            rh_air = float(rh_arr[i] if rh_arr[i] is not None else 50.0)
            yagis = float(p_arr[i] if p_arr[i] is not None else 0.0)
            radyasyon = float(rad_arr[i] if i < len(rad_arr) and rad_arr[i] is not None else 0.0)
            ruzgar = float(w_arr[i] if i < len(w_arr) and w_arr[i] is not None else 12.0)
            fm1, fm10, fm100, fm240 = _saatlik_nem_adimi(fm1, fm10, fm100, fm240, t_air, rh_air, yagis, radyasyon, ruzgar)

        bilesik_yakit_nemi = round(0.45 * fm1 + 0.30 * fm10 + 0.15 * fm100 + 0.10 * fm240, 1)
        ort_sicaklik = round(float(np.mean([x for x in t_arr if x is not None])) if t_arr else 20.0, 1)
        ort_nem = round(float(np.mean([x for x in rh_arr if x is not None])) if rh_arr else 50.0, 1)
        toplam_yagis = round(float(np.sum([x for x in p_arr if x is not None])) if p_arr else 0.0, 1)
        ort_radyasyon = round(float(np.mean([x for x in rad_arr if x is not None])) if rad_arr else 350.0, 0)

        lfmc = round(float(np.clip(92.0 + (toplam_yagis * 0.6) - max(0.0, ort_sicaklik - 17.0) * 1.4, 72.0, 126.0)), 1)

        anlik_t = round(float(anlik["temperature_2m"]), 1)
        anlik_rh = round(float(anlik["relative_humidity_2m"]), 1)
        anlik_vpd = round(_buhar_basinci_acigi_vpd(anlik_t, anlik_rh), 2)

        # Gelecek 12 saatin her biri için dinamik meteoroloji + evrilen yakıt nemi dizisi
        gelecek_12saat = []
        is_day_arr = saatlik.get("is_day", [])
        all_rad = saatlik.get("shortwave_radiation", [])
        all_prec = saatlik.get("precipitation", [])
        f1_s, f10_s, f100_s, f240_s = fm1, fm10, fm100, fm240

        for offset in range(12):
            h_idx = baslangic_idx + offset
            astro_day, astro_rad = _astronomik_gunes_durumu(enlem, boylam, simdi_utc + timedelta(hours=offset))
            if h_idx < len(saatlik["temperature_2m"]):
                ht = float(saatlik["temperature_2m"][h_idx] if saatlik["temperature_2m"][h_idx] is not None else anlik_t)
                hrh = float(saatlik["relative_humidity_2m"][h_idx] if saatlik["relative_humidity_2m"][h_idx] is not None else anlik_rh)
                hw = float(saatlik["wind_speed_10m"][h_idx] if saatlik["wind_speed_10m"][h_idx] is not None else anlik["wind_speed_10m"])
                hwd = int(saatlik["wind_direction_10m"][h_idx] if saatlik["wind_direction_10m"][h_idx] is not None else anlik["wind_direction_10m"])
                hrad = float(all_rad[h_idx]) if h_idx < len(all_rad) and all_rad[h_idx] is not None else astro_rad
                hprec = float(all_prec[h_idx]) if h_idx < len(all_prec) and all_prec[h_idx] is not None else 0.0
                hday = int(is_day_arr[h_idx]) if h_idx < len(is_day_arr) and is_day_arr[h_idx] is not None else astro_day
            else:
                ht, hrh, hw, hwd, hrad, hprec, hday = anlik_t, anlik_rh, float(anlik["wind_speed_10m"]), int(anlik["wind_direction_10m"]), astro_rad, 0.0, astro_day

            if offset > 0:
                f1_s, f10_s, f100_s, f240_s = _saatlik_nem_adimi(f1_s, f10_s, f100_s, f240_s, ht, hrh, hprec, hrad, hw)

            saatlik_fm = round(0.45 * f1_s + 0.30 * f10_s + 0.15 * f100_s + 0.10 * f240_s, 1)

            gelecek_12saat.append({
                "sicaklik": round(ht, 1),
                "nem": round(hrh, 1),
                "ruzgar_hiz": round(hw, 1),
                "ruzgar_yon": hwd,
                "radyasyon": round(hrad, 1),
                "yagis_mm": round(hprec, 1),
                "is_day": hday,
                "vpd": round(_buhar_basinci_acigi_vpd(ht, hrh), 2),
                "yakit_nemi": saatlik_fm
            })

        return {
            "anlik_sicaklik": anlik_t,
            "anlik_nem": anlik_rh,
            "anlik_vpd_kpa": anlik_vpd,
            "anlik_ruzgar_hizi": round(float(anlik["wind_speed_10m"]), 1),
            "anlik_ruzgar_yonu": int(anlik["wind_direction_10m"]),
            "son_5_gun_ort_sicaklik": ort_sicaklik,
            "son_5_gun_ort_nem": ort_nem,
            "son_5_gun_yagis_mm": toplam_yagis,
            "son_10_gun_ort_radyasyon_wm2": ort_radyasyon,
            "olu_yakit_nemi_100saat": bilesik_yakit_nemi,
            "canli_yaprak_nemi_lfmc": lfmc,
            "gelecek_12saat": gelecek_12saat,
            "kaynak": "Open-Meteo (Canlı + 12 Saatlik Dinamik Evrim)"
        }

    # Yedek profilde dahi koordinatın astronomik gece/gündüz döngüsünü saatlik üret
    gelecek_12saat = []
    f1_s, f10_s, f100_s, f240_s = 11.5, 12.2, 13.0, 14.0
    for offset in range(12):
        astro_day, astro_rad = _astronomik_gunes_durumu(enlem, boylam, simdi_utc + timedelta(hours=offset))
        ht = round(26.5 if astro_day == 1 else 16.5, 1)
        hrh = round(35.0 if astro_day == 1 else 68.0, 1)
        hw = round(20.0 if astro_day == 1 else 12.0, 1)
        if offset > 0:
            f1_s, f10_s, f100_s, f240_s = _saatlik_nem_adimi(f1_s, f10_s, f100_s, f240_s, ht, hrh, 0.0, astro_rad, hw)
        saatlik_fm = round(0.45 * f1_s + 0.30 * f10_s + 0.15 * f100_s + 0.10 * f240_s, 1)
        gelecek_12saat.append({
            "sicaklik": ht,
            "nem": hrh,
            "ruzgar_hiz": hw,
            "ruzgar_yon": 45,
            "radyasyon": astro_rad,
            "yagis_mm": 0.0,
            "is_day": astro_day,
            "vpd": round(_buhar_basinci_acigi_vpd(ht, hrh), 2),
            "yakit_nemi": saatlik_fm
        })

    ilk = gelecek_12saat[0]
    return {
        "anlik_sicaklik": ilk["sicaklik"],
        "anlik_nem": ilk["nem"],
        "anlik_vpd_kpa": ilk["vpd"],
        "anlik_ruzgar_hizi": ilk["ruzgar_hiz"],
        "anlik_ruzgar_yonu": 45,
        "son_5_gun_ort_sicaklik": 23.0,
        "son_5_gun_ort_nem": 44.0,
        "son_5_gun_yagis_mm": 0.0,
        "son_10_gun_ort_radyasyon_wm2": 440.0,
        "olu_yakit_nemi_100saat": ilk["yakit_nemi"],
        "canli_yaprak_nemi_lfmc": 88.0,
        "gelecek_12saat": gelecek_12saat,
        "kaynak": "Astronomik Günlük Döngü Yedek Profili"
    }


# 2. ÇİFT KAYNAKLI KESİNTİSİZ COPERNICUS / SRTM DEM (80x80 IZGARA, 50M ÇÖZÜNÜRLÜK)
def arazi_dem_getir(enlem: float, boylam: float, boyut: int = 80) -> dict:
    yari_cap = 0.020
    ornek_n = 7
    lats = np.linspace(enlem - yari_cap, enlem + yari_cap, ornek_n)
    lons = np.linspace(boylam - yari_cap, boylam + yari_cap, ornek_n)

    lat_list, lon_list, ciftler = [], [], []
    for la in lats:
        for lo in lons:
            lat_list.append(f"{la:.5f}")
            lon_list.append(f"{lo:.5f}")
            ciftler.append(f"{la:.5f},{lo:.5f}")

    elev_7 = None
    dem_kaynak = "Copernicus DEM 30m"

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": ",".join(lat_list), "longitude": ",".join(lon_list)},
            timeout=9
        )
        if r.status_code == 200 and "elevation" in r.json():
            elev_7 = np.array(r.json()["elevation"], dtype=float).reshape((ornek_n, ornek_n))
    except Exception:
        pass

    if elev_7 is None:
        try:
            r2 = requests.get(
                "https://api.opentopodata.org/v1/srtm90m",
                params={"locations": "|".join(ciftler)},
                timeout=9
            )
            if r2.status_code == 200 and "results" in r2.json():
                vals = [float(item["elevation"] or 0.0) for item in r2.json()["results"]]
                if len(vals) == ornek_n * ornek_n:
                    elev_7 = np.array(vals, dtype=float).reshape((ornek_n, ornek_n))
                    dem_kaynak = "SRTM 90m DEM"
        except Exception:
            pass

    if elev_7 is None:
        elev_7 = np.full((ornek_n, ornek_n), 450.0)
        dem_kaynak = "DEM Yedek"

    y_old = np.linspace(0, 1, ornek_n)
    x_old = np.linspace(0, 1, ornek_n)
    y_new = np.linspace(0, 1, boyut)
    x_new = np.linspace(0, 1, boyut)

    temp = np.array([np.interp(x_new, x_old, row) for row in elev_7])
    elev_grid = np.array([np.interp(y_new, y_old, temp[:, c]) for c in range(boyut)]).T

    hucre_m = 50.0
    dy, dx = np.gradient(elev_grid, hucre_m)
    egim_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    egim_derece = np.degrees(egim_rad)
    baki_derece = (np.degrees(np.arctan2(-dx, dy)) + 360.0) % 360.0

    pad_elev = np.pad(elev_grid, 4, mode="edge")
    komsu_ort = np.zeros_like(elev_grid)
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            komsu_ort += pad_elev[4 + dr:4 + dr + boyut, 4 + dc:4 + dc + boyut]
    komsu_ort /= 81.0
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
        "kaynak": dem_kaynak
    }


def _cizgi_izgaraya_isle(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, deger,
    carpan_grid: np.ndarray = None, carpan: float = 1.0,
    cbh_grid: np.ndarray = None, cbh: float = 3.5,
    kap_grid: np.ndarray = None, kap: int = 2
):
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
            if kap_grid is not None:
                kap_grid[rr, cc] = kap


def _poligon_icini_doldur(
    grid: np.ndarray, halka: list[tuple[int, int]], deger: int,
    carpan_grid: np.ndarray, carpan: float,
    cbh_grid: np.ndarray, cbh: float,
    kap_grid: np.ndarray, kap: int
):
    if len(halka) < 3:
        return
    for idx in range(len(halka) - 1):
        _cizgi_izgaraya_isle(
            grid, halka[idx][0], halka[idx][1], halka[idx + 1][0], halka[idx + 1][1],
            deger, carpan_grid, carpan, cbh_grid, cbh, kap_grid, kap
        )

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
                kap_grid[rr, cc] = kap


# 3. AKDENİZ VE EGE ORTALAMASI: KIZILÇAM + MAKİ TABAN MODELİ, KAPALILIK & ENGELLER
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
    cbh_grid = np.where(yakit_grid == 1, 2.2, np.where(yakit_grid == 2, 0.8, np.where(yakit_grid == 3, 5.5, 8.0)))
    # Kapalılık: 3 = Tam Kapalı, 2 = Orta Kapalı, 1 = Gevşek, 0 = Açık Alan
    kapalilik_grid = np.where(yakit_grid == 1, 2, np.where(yakit_grid == 2, 3, np.where(yakit_grid == 3, 2, 0)))

    engel_sayisi = 0
    osm_basarili = False

    query = f"""
    [out:json][timeout:10];
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
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            nodes = {el["id"]: (el["lat"], el["lon"]) for el in data.get("elements", []) if el["type"] == "node"}
            for el in data.get("elements", []):
                if el["type"] != "way":
                    continue
                tags = el.get("tags", {})
                sinif = None
                cbh_val = 2.5
                kap_val = 2
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
                    kap_val = 3
                elif tags.get("landuse") in ("farmland", "orchard"):
                    sinif = 4
                    cbh_val = 9.0
                    kap_val = 0
                elif tags.get("natural") == "wood" or tags.get("landuse") == "forest":
                    sinif = 3 if tags.get("leaf_type") == "broadleaved" else 1
                    cbh_val = 5.0 if sinif == 3 else 2.2
                    kap_val = 2

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
                        _cizgi_izgaraya_isle(
                            yakit_grid, r0, c0, r1, c1, sinif,
                            cbh_grid=cbh_grid, cbh=cbh_val,
                            kap_grid=kapalilik_grid, kap=kap_val
                        )
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
        "kapalilik_grid": kapalilik_grid,
        "baskin_bitki_ortusu": baskin_ad,
        "tespit_edilen_engel_hatti": engel_sayisi,
        "kaynak": "Akdeniz-Ege Kızılçam/Maki + OSM" if osm_basarili else "Akdeniz-Ege Kızılçam/Maki Modeli"
    }


def _mescere_rumuzu_coz(etiket_metni: str, varsayilan_poligon: bool = True) -> tuple[int, float, float, int]:
    """Meşcere tipi, yaş sınıfı (a,b,c,d) ve kapalılık (1,2,3) -> (yakıt_sınıfı, hız_çarpanı, CBH_m, kapalilik)."""
    temiz = re.sub(r"<[^>]+>", " ", etiket_metni).lower()

    if any(k in temiz for k in ["yey", "emniyet", "orman yolu", "yol", "dere", "akarsu", "gölet", "baraj", "su", "kaya", "iskân", "iskan"]):
        return 0, 0.0, 99.0, 0

    eslesme = re.search(r"\b(çz|cz|çk|ck|mb|md|m|bçz|bçk|kz|kn|me|z|ot|a|t)([abcd]{1,2})?([0123])?\b", temiz)
    if eslesme:
        tur = eslesme.group(1)
        cag = eslesme.group(2) or ""
        kap_str = eslesme.group(3) or ""

        if tur in ("çz", "cz", "bçz", "kz"):
            sinif = 1
        elif tur in ("m", "mb", "md"):
            sinif = 2
        elif tur in ("çk", "ck", "bçk", "kn", "me"):
            sinif = 3
        else:
            sinif = 4

        if "a" in cag and "b" not in cag:
            cag_carpan, cbh = 1.30, 1.0
        elif "a" in cag and "b" in cag:
            cag_carpan, cbh = 1.24, 1.5
        elif "b" in cag and "c" not in cag:
            cag_carpan, cbh = 1.15, 2.2
        elif "b" in cag and "c" in cag:
            cag_carpan, cbh = 1.08, 3.2
        elif "c" in cag and "d" not in cag:
            cag_carpan, cbh = 1.00, 4.8
        elif "d" in cag:
            cag_carpan, cbh = 0.85, 7.5
        else:
            cag_carpan = 0.85 if tur == "mb" else 1.00
            cbh = 0.8 if sinif == 2 else (5.5 if sinif == 3 else 2.5)

        if kap_str == "3":
            kap_carpan, kap_int = 1.12, 3
        elif kap_str == "1":
            kap_carpan, kap_int = 0.88, 1
            cbh += 1.2
        elif kap_str == "0":
            kap_carpan, kap_int = 0.65, 0
            cbh += 3.0
        else:
            kap_carpan, kap_int = 1.00, (1 if tur == "mb" else 2)

        return sinif, round(cag_carpan * kap_carpan, 2), cbh, kap_int

    if "maki" in temiz or "çalı" in temiz or "cali" in temiz:
        return 2, 1.0, 0.8, 3
    if "gençlik" in temiz or "sıklık" in temiz:
        return 1, 1.30, 1.0, 3
    if "sırıklık" in temiz:
        return 1, 1.15, 2.2, 3
    if "ağaçlık" in temiz or "yaşlı" in temiz:
        return 1, 0.85, 7.5, 2
    if "zeytin" in temiz or "tarım" in temiz or "açıklık" in temiz:
        return 4, 0.80, 9.0, 0

    return (1, 1.0, 2.5, 2) if varsayilan_poligon else (0, 0.0, 99.0, 0)


# 4. İSTEĞE BAĞLI KML / GEOJSON MEŞCERE VE YOL KATMAN ENTEGRASYONU
def ozel_katman_uygula(
    yakit_grid: np.ndarray, enlem: float, boylam: float, yari_cap: float, dosya_icerik: str,
    carpan_grid: np.ndarray = None, cbh_grid: np.ndarray = None, kap_grid: np.ndarray = None
) -> tuple[np.ndarray, int]:
    boyut = yakit_grid.shape[0]
    if carpan_grid is None:
        carpan_grid = np.ones((boyut, boyut), dtype=float)
    if cbh_grid is None:
        cbh_grid = np.full((boyut, boyut), 2.5, dtype=float)
    if kap_grid is None:
        kap_grid = np.full((boyut, boyut), 2, dtype=int)

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
                sinif, carpan, cbh, kap = _mescere_rumuzu_coz(props_str, varsayilan_poligon=False)
                for idx in range(len(coords) - 1):
                    r0, c0 = koordinat_noktasi(coords[idx][0], coords[idx][1])
                    r1, c1 = koordinat_noktasi(coords[idx + 1][0], coords[idx + 1][1])
                    _cizgi_izgaraya_isle(yakit_grid, r0, c0, r1, c1, sinif, carpan_grid, carpan, cbh_grid, cbh, kap_grid, kap)
                islenen_nesne += 1
            elif gtype == "Polygon" and coords:
                sinif, carpan, cbh, kap = _mescere_rumuzu_coz(props_str, varsayilan_poligon=True)
                halka = [koordinat_noktasi(c[0], c[1]) for c in coords[0]]
                _poligon_icini_doldur(yakit_grid, halka, sinif, carpan_grid, carpan, cbh_grid, cbh, kap_grid, kap)
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
            sinif, carpan, cbh, kap = _mescere_rumuzu_coz(tum_etiket, varsayilan_poligon=poligon_var)

            for alt in pm.iter():
                if alt.tag.endswith("coordinates") and alt.text:
                    parcalar = alt.text.strip().split()
                    noktalar = []
                    for p in parcalar:
                        xyz = p.split(",")
                        if len(xyz) >= 2:
                            noktalar.append(koordinat_noktasi(float(xyz[0]), float(xyz[1])))
                    if len(noktalar) >= 3 and poligon_var:
                        _poligon_icini_doldur(yakit_grid, noktalar, sinif, carpan_grid, carpan, cbh_grid, cbh, kap_grid, kap)
                        islenen_nesne += 1
                    elif len(noktalar) >= 2:
                        for idx in range(len(noktalar) - 1):
                            _cizgi_izgaraya_isle(yakit_grid, noktalar[idx][0], noktalar[idx][1], noktalar[idx + 1][0], noktalar[idx + 1][1], sinif, carpan_grid, carpan, cbh_grid, cbh, kap_grid, kap)
                        islenen_nesne += 1

    return yakit_grid, islenen_nesne


# 5. ROTHERMEL (1972) + ALBINI WAF (1979) + VAN WAGNER (1977) + RYAN/SHARPLES TOPOGRAFYA
def _rothermel_hesapla(
    yakit_sinif: int, mf_yuzde: float, efektif_ruzgar_kmh: float,
    cos_ruzgar_aci: float, egim_tan: float, is_day: int,
    kapalilik: int, cbh_m: float, lfmc: float, vpd_kpa: float
) -> tuple[float, float, bool]:
    """
    Rothermel (1972) yüzey yayılımı + Albini & Baughman (1979) tepe kapalılığı rüzgar sönümlenmesi (WAF)
    + Van Wagner (1977) tepe yangını geçişi.
    """
    parametreler = {
        1: (1.85, 5700.0, 16.0, 24.5),  # Kızılçam + Maki alt örtüsü
        2: (1.60, 6200.0, 18.0, 23.5),  # Saf Maki ve Sık Çalılık
        3: (1.35, 5100.0, 19.0, 25.0),  # Karaçam / Yüksek Rakım Ormanı
        4: (0.65, 7800.0, 8.5, 21.0),   # Zeytinlik / Ot ve Tarım
    }
    w0, sigma, rho_b, mx = parametreler.get(yakit_sinif, (1.5, 5600.0, 16.0, 24.0))

    if mf_yuzde >= mx:
        return 0.0, 0.0, False

    # Albini & Baughman (1979): Meşcere kapalılığına bağlı rüzgar sönümlenme katsayısı (WAF)
    waf_haritasi = {3: 0.24, 2: 0.38, 1: 0.54, 0: 0.68}
    waf = waf_haritasi.get(kapalilik, 0.38)

    r_m = mf_yuzde / mx
    eta_m = max(0.0, 1.0 - 2.59 * r_m + 5.11 * (r_m ** 2) - 3.52 * (r_m ** 3))
    eta_s = 0.417
    h_yanma = 18700.0
    w_n = w0 * 0.9445

    sigma_imp = sigma / 3.28084
    rho_p = 513.0
    beta = rho_b / rho_p
    beta_op = 3.348 * (sigma_imp ** -0.8189)
    beta_oran = beta / beta_op

    gamma_max = (sigma_imp ** 1.5) / (495.0 + 0.0594 * (sigma_imp ** 1.5))
    a_exp = 133.0 * (sigma_imp ** -0.7913)
    gamma_prime = gamma_max * (beta_oran ** a_exp) * math.exp(a_exp * (1.0 - beta_oran))

    i_r = gamma_prime * w_n * h_yanma * eta_m * eta_s
    xi = math.exp((0.792 + 0.681 * math.sqrt(sigma_imp)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma_imp)
    epsilon = math.exp(-138.0 / sigma_imp)
    q_ig = 581.0 + 2594.0 * (mf_yuzde / 100.0)

    r0 = (i_r * xi) / max(1.0, rho_b * epsilon * q_ig)

    c_wind = 7.47 * math.exp(-0.133 * (sigma_imp ** 0.55))
    b_wind = 0.02526 * (sigma_imp ** 0.54)
    e_wind = 0.715 * math.exp(-3.59e-4 * sigma_imp)

    def _hiz_ve_siddet(aktif_waf: float) -> tuple[float, float]:
        u_mid_ft_min = (efektif_ruzgar_kmh * aktif_waf * 1000.0 / 60.0) * 3.28084
        phi_w_head = c_wind * (max(0.1, u_mid_ft_min) ** b_wind) * (beta_oran ** -e_wind)
        if cos_ruzgar_aci >= 0.0:
            phi_w = phi_w_head * (cos_ruzgar_aci ** 1.35)
        else:
            phi_w = -0.55 * abs(cos_ruzgar_aci)

        if egim_tan >= 0.0:
            gunduz_egim_carpan = 1.25 if is_day == 1 else 0.65
            phi_s = 5.275 * (beta ** -0.3) * (egim_tan ** 1.85) * gunduz_egim_carpan
        else:
            if is_day == 0:
                phi_s = min(0.65, 1.1 * abs(egim_tan))
            else:
                phi_s = max(-0.65, -2.0 * abs(egim_tan))

        ros_val = max(0.06, r0 * max(0.15, 1.0 + phi_w + phi_s))
        i_r_kw_m2 = i_r / 60.0
        kalma_suresi_dk = 384.0 / sigma_imp
        byram_val = i_r_kw_m2 * (ros_val * kalma_suresi_dk)
        return ros_val, byram_val

    ros, byram_ib = _hiz_ve_siddet(waf)

    # Van Wagner (1977) Tepe Yangını Geçiş Eşiği + Albini WAF=1.0 Açık Rüzgar Sıçraması
    tepe_aktif = False
    i0_kritik = (0.010 * cbh_m * (460.0 + 25.9 * lfmc)) ** 1.5
    if byram_ib >= i0_kritik and yakit_sinif in (1, 2, 3) and (is_day == 1 or vpd_kpa > 2.2):
        tepe_aktif = True
        ros_tepe, byram_tepe = _hiz_ve_siddet(1.0)  # Ağaç tepesine çıkan alev 10m rüzgarının %100'ünü alır
        ros = max(ros * 1.45, ros_tepe * 1.35)
        byram_ib = max(byram_ib * 1.6, byram_tepe * 1.4)

    return ros, byram_ib, tepe_aktif


def yangin_simule_et(enlem: float, boylam: float, hava: dict, dem: dict, bitki: dict) -> dict:
    yakit = bitki["yakit_grid"]
    carpan_grid = bitki.get("yakit_carpan_grid", np.ones_like(yakit, dtype=float))
    cbh_grid = bitki.get("cbh_grid", np.full_like(yakit, 2.5, dtype=float))
    kap_grid = bitki.get("kapalilik_grid", np.full_like(yakit, 2, dtype=int))
    elev = dem["yukselti_grid"]
    baki = dem["baki_grid"]
    tpi = dem["tpi_grid"]
    boyut = yakit.shape[0]
    hucre_m = dem["hucre_m"]
    yari_cap = dem["yari_cap"]
    merkez = boyut // 2

    lfmc = hava.get("canli_yaprak_nemi_lfmc", 90.0)
    gelecek_12 = hava["gelecek_12saat"]
    dy_elev, dx_elev = np.gradient(elev, hucre_m)

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
    saatlik_durum_etiketi = [""] * 12
    kor_sicrama_sayaci = set()

    kuyruk = [(0.0, merkez, merkez)]
    while kuyruk:
        mevcut_t, r, c = heapq.heappop(kuyruk)
        if mevcut_t > yanma_saati[r, c] or mevcut_t >= 12.0:
            continue

        saat_indeks = min(11, int(mevcut_t))
        saat_hava = gelecek_12[saat_indeks]
        sinoptik_hiz = saat_hava["ruzgar_hiz"]
        sinoptik_gidis_rad = math.radians((saat_hava["ruzgar_yon"] + 180.0) % 360.0)
        is_day = saat_hava["is_day"]
        saat_vpd = saat_hava["vpd"]
        saat_fm = saat_hava["yakit_nemi"]

        # Saatlik yağış varsa sönme kontrolü
        if saat_hava.get("yagis_mm", 0.0) >= 1.5 and saat_fm >= 23.5:
            saatlik_durum_etiketi[saat_indeks] = "Yağış Sönümlemesi"
            continue

        ux = math.sin(sinoptik_gidis_rad)
        uy = math.cos(sinoptik_gidis_rad)

        gx = float(dx_elev[r, c])
        gy = float(dy_elev[r, c])
        egim_mag = math.hypot(gx, gy)
        egim_derece_yerel = math.degrees(math.atan(egim_mag))

        # RYAN (1977) SIRT ARKASI PERDELEMESİ & SHARPLES (2012) GİRDAP KAYNAKLI YANAL YAYILIM (VLS)
        yerel_sinoptik_hiz = sinoptik_hiz
        sirt_arkasi_aktif = False
        vls_girdap_aktif = False
        kontur_x, kontur_y = 0.0, 0.0

        if egim_mag > 0.015 and sinoptik_hiz > 0.5:
            sx = gx / egim_mag
            sy = gy / egim_mag
            yon_uyumu = ux * sx + uy * sy

            if yon_uyumu < -0.70 and egim_derece_yerel >= 18.0 and sinoptik_hiz >= 25.0:
                # Sharples vd. (2012): Dik sırt arkasında şiddetli rüzgarda yamaç konturuna paralel yanal girdap
                vls_girdap_aktif = True
                kontur_x, kontur_y = -sy, sx
                yerel_sinoptik_hiz = sinoptik_hiz * 0.72
            elif yon_uyumu < -0.10:
                # Ryan (1977): Sırt arkası rüzgar perdelemesi (%45 - %78 azalma)
                sirt_arkasi_aktif = True
                kalkan = max(0.22, 1.0 - 3.2 * abs(yon_uyumu) * min(0.30, egim_mag))
                if tpi[r, c] < -3.0:
                    kalkan *= 0.72
                yerel_sinoptik_hiz = sinoptik_hiz * kalkan
            elif yon_uyumu > 0.15:
                hizlanma = 1.0 + 0.50 * yon_uyumu * min(0.30, egim_mag)
                yerel_sinoptik_hiz = sinoptik_hiz * hizlanma

        wx = yerel_sinoptik_hiz * ux
        wy = yerel_sinoptik_hiz * uy

        # Gündüz Anabatik (Eğim Yukarı) / Gece Katabatik (Eğim Aşağı) Termal Rüzgar
        egim_siddet = min(0.5, egim_mag)
        if is_day == 1:
            termal_hiz = 16.0 * egim_siddet * min(1.0, saat_hava["radyasyon"] / 500.0)
            if egim_mag > 1e-5:
                wx += termal_hiz * (gx / egim_mag)
                wy += termal_hiz * (gy / egim_mag)
            if vls_girdap_aktif:
                saatlik_durum_etiketi[saat_indeks] = "VLS Yanal Girdap"
            elif sirt_arkasi_aktif:
                saatlik_durum_etiketi[saat_indeks] = "Sırt Arkası Yavaş"
            else:
                saatlik_durum_etiketi[saat_indeks] = "Gündüz (Eğim Yukarı)"
        else:
            katabatik_hiz = 14.0 * egim_siddet
            if egim_mag > 1e-5:
                wx -= katabatik_hiz * (gx / egim_mag)
                wy -= katabatik_hiz * (gy / egim_mag)
            saatlik_durum_etiketi[saat_indeks] = "Gece (Katabatik Yavaş)"

        efektif_ruzgar_hiz = math.hypot(wx, wy)
        efektif_ruzgar_yon_deg = (math.degrees(math.atan2(wx, wy)) + 360.0) % 360.0

        # Gece VPD ve Nem Geri Kazanımı Freni
        gece_nem_eklemesi = 0.0
        vpd_freni = 1.0
        if is_day == 0:
            gece_nem_eklemesi = max(1.5, (saat_hava["nem"] - 50.0) * 0.14)
            if -5.0 <= tpi[r, c] <= 18.0:
                gece_nem_eklemesi *= 0.55
            elif tpi[r, c] < -10.0:
                gece_nem_eklemesi += 2.2

            if saat_vpd < 0.45:
                vpd_freni = 0.22
            elif saat_vpd < 0.80:
                vpd_freni = 0.45
            else:
                vpd_freni = 0.65

        for dr, dc, mesafe, yon_azimut in komsular:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < boyut and 0 <= nc < boyut):
                continue

            hedef_yakit = int(yakit[nr, nc])
            if hedef_yakit == 0:
                if elev[nr, nc] <= 1.0:
                    continue
                if saatlik_max_byram[saat_indeks] > 1800.0 and efektif_ruzgar_hiz >= 25.0 and int(yakit[r, c]) in (1, 2):
                    yayilim_hizi_m_dk = 2.1
                    gecis_saati = (mesafe / yayilim_hizi_m_dk) / 60.0
                    yeni_t = mevcut_t + gecis_saati
                    if yeni_t < yanma_saati[nr, nc] and yeni_t <= 12.0:
                        yanma_saati[nr, nc] = yeni_t
                        heapq.heappush(kuyruk, (yeni_t, nr, nc))
                continue

            baki_rad = math.radians(float(baki[nr, nc]))
            gunes_faktoru = min(1.0, saat_hava["radyasyon"] / 650.0) if is_day == 1 else 0.0
            baki_nem_farki = 1.4 * math.cos(baki_rad) * gunes_faktoru
            yerel_mf = max(3.5, saat_fm + gece_nem_eklemesi + baki_nem_farki)

            cos_aci = math.cos(math.radians(yon_azimut - efektif_ruzgar_yon_deg))
            egim_tan = (elev[nr, nc] - elev[r, c]) / mesafe

            ros_yuzey, byram_ib, tepe_mi = _rothermel_hesapla(
                hedef_yakit, yerel_mf, efektif_ruzgar_hiz, cos_aci, egim_tan,
                is_day, int(kap_grid[nr, nc]), float(cbh_grid[nr, nc]), lfmc, saat_vpd
            )
            if ros_yuzey <= 0.0:
                continue

            # VLS Yanal Girdap Yönelim Çarpanı (Kontur boyunca sağa-sola hızlanma)
            vls_carpan = 1.0
            if vls_girdap_aktif:
                komsu_rad = math.radians(yon_azimut)
                kx, ky = math.sin(komsu_rad), math.cos(komsu_rad)
                kontur_uyumu = abs(kx * kontur_x + ky * kontur_y)
                if kontur_uyumu > 0.65:
                    vls_carpan = 1.65

            # Viegas (2005) Dar Dere / Vadi Baca Etkisi
            hedef_tpi = float(tpi[nr, nc])
            if is_day == 1 and hedef_tpi < -8.0 and egim_tan > 0.04:
                topo_carpan = min(1.90, 1.0 + abs(hedef_tpi) / 28.0)
            elif is_day == 0 and hedef_tpi < -6.0 and egim_tan < -0.03:
                topo_carpan = min(1.35, 1.0 + abs(hedef_tpi) / 45.0)
            else:
                topo_carpan = 1.0

            # Viegas vd. (2012) Alev Cephesi Birleşme Vakumu (Junction Fire Acceleration)
            yanan_komsu_sayisi = 0
            for kdr, kdc, _, _ in komsular:
                kr, kc = nr + kdr, nc + kdc
                if 0 <= kr < boyut and 0 <= kc < boyut and yanma_saati[kr, kc] <= mevcut_t:
                    yanan_komsu_sayisi += 1
            birlesme_carpan = 1.85 if yanan_komsu_sayisi >= 5 else 1.0

            ros_etkin = ros_yuzey * float(carpan_grid[nr, nc]) * topo_carpan * vls_carpan * birlesme_carpan * vpd_freni
            byram_etkin = byram_ib * float(carpan_grid[nr, nc]) * topo_carpan * birlesme_carpan

            if tepe_mi:
                saatlik_tepe_aktif[saat_indeks] = True

            if byram_etkin > saatlik_max_byram[saat_indeks]:
                saatlik_max_byram[saat_indeks] = byram_etkin

            # Sardoy vd. (2007) / Albini (1979) Uzun Mesafeli Kozalak Sıçraması (200m - 900m)
            if (
                byram_etkin > 2200.0 and efektif_ruzgar_hiz >= 25.0
                and hedef_yakit in (1, 2) and (r + c + saat_indeks) % 4 == 0
                and (r, c, saat_indeks) not in kor_sicrama_sayaci
            ):
                kor_sicrama_sayaci.add((r, c, saat_indeks))
                sicrama_m = min(900.0, 150.0 + 0.14 * math.sqrt(byram_etkin) * (efektif_ruzgar_hiz / 10.0))
                adim_hucre = max(3, int(round(sicrama_m / hucre_m)))
                s_rad = math.radians(efektif_ruzgar_yon_deg)
                sr = r + int(round(adim_hucre * math.cos(s_rad)))
                sc = c + int(round(adim_hucre * math.sin(s_rad)))
                if 0 <= sr < boyut and 0 <= sc < boyut and int(yakit[sr, sc]) > 0:
                    spot_t = mevcut_t + 0.25
                    if spot_t < yanma_saati[sr, sc] and spot_t <= 12.0:
                        yanma_saati[sr, sc] = spot_t
                        heapq.heappush(kuyruk, (spot_t, sr, sc))

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
            d = 0.00045
            return [[
                [round(boylam + d * math.cos(math.radians(a)), 5), round(enlem + d * math.sin(math.radians(a)), 5)]
                for a in range(0, 361, 20)
            ]]

        sektor_sayisi = 36
        max_uzaklik = np.full(sektor_sayisi, 0.00045)
        for rr, cc in zip(yanan_r, yanan_c):
            la = min_lat + (rr / (boyut - 1)) * (max_lat - min_lat)
            lo = min_lon + (cc / (boyut - 1)) * (max_lon - min_lon)
            dlat = la - enlem
            dlon = lo - boylam
            mes = math.hypot(dlat, dlon) + 0.00045
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
    onceki_alan_ha = 0.0
    for s in range(1, 13):
        idx = s - 1
        sh = gelecek_12[idx]
        yanan_hucre = int(np.sum(yanma_saati <= s))
        alan_ha = round(yanan_hucre * hucre_hektar, 1)
        byram_val = round(saatlik_max_byram[idx], 0)

        if sh["yakit_nemi"] >= 24.5 or yanan_hucre <= 1:
            anlik_hiz = 0.0
            risk = "Yok (Sönme Eşiği)"
        else:
            r_onceki = math.sqrt((onceki_alan_ha * 10000.0) / math.pi)
            r_simdi = math.sqrt((alan_ha * 10000.0) / math.pi)
            anlik_hiz = round(max(0.0, (r_simdi - r_onceki) / 60.0), 1)
            mod_etiket = saatlik_durum_etiketi[idx] or ("Gündüz" if sh["is_day"] == 1 else "Gece")
            if saatlik_tepe_aktif[idx]:
                risk = f"Tepe Yangını ({int(byram_val)} kW/m)"
            else:
                risk = f"{mod_etiket} ({int(byram_val)} kW/m)"

        onceki_alan_ha = alan_ha
        saatlik_ozet.append({
            "saat": s,
            "alan_hektar": alan_ha,
            "yayilim_hizi_m_dk": anlik_hiz,
            "tepe_yangini_riski": risk,
            "saatlik_hava": {
                "sicaklik": sh["sicaklik"],
                "nem": sh["nem"],
                "ruzgar_hiz": sh["ruzgar_hiz"],
                "ruzgar_yon": sh["ruzgar_yon"],
                "yakit_nemi": sh["yakit_nemi"],
                "is_day": sh["is_day"]
            }
        })

    return {
        "izokronlar": izokronlar,
        "saatlik_ozet": saatlik_ozet,
        "kuruluk_katsayisi": round(hava["anlik_vpd_kpa"], 2)
    }
