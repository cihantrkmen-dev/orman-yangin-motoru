from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor import (
    hava_durumu_getir,
    arazi_dem_getir,
    bitki_ve_engeller_getir,
    ozel_katman_uygula,
    yangin_simule_et
)

app = FastAPI(
    title="Orman Yangını Davranış Analizi ve Simülasyon Motoru",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulasyonIstegi(BaseModel):
    enlem: float
    boylam: float
    ozel_katman_icerik: Optional[str] = None


@app.get("/")
def ana_sayfa():
    return {
        "durum": "aktif",
        "motor": "Orman Yangını Davranış ve Simülasyon Motoru v1.0",
        "veri_kaynaklari": [
            "Open-Meteo (Son 5 Günlük + Anlık Hava)",
            "Copernicus DEM (Arazi Eğimi ve Bakı)",
            "OpenStreetMap + Akdeniz Ekolojik Katmanı",
            "İsteğe Bağlı KML / GeoJSON Katmanı"
        ]
    }


@app.post("/simule-et")
def simule_et_endpoint(istek: SimulasyonIstegi):
    hava = hava_durumu_getir(istek.enlem, istek.boylam)
    dem = arazi_dem_getir(istek.enlem, istek.boylam)
    bitki = bitki_ve_engeller_getir(istek.enlem, istek.boylam, dem["yukselti_grid"])

    ozel_nesne_sayisi = 0
    if istek.ozel_katman_icerik:
        bitki["yakit_grid"], ozel_nesne_sayisi = ozel_katman_uygula(
            bitki["yakit_grid"],
            istek.enlem,
            istek.boylam,
            istek.ozel_katman_icerik
        )

    simulasyon = yangin_simule_et(istek.enlem, istek.boylam, hava, dem, bitki)

    return {
        "koordinat": {"enlem": istek.enlem, "boylam": istek.boylam},
        "ozet_kartlar": {
            "son_5_gun": {
                "ort_sicaklik": hava["son_5_gun_ort_sicaklik"],
                "ort_nem": hava["son_5_gun_ort_nem"],
                "toplam_yagis_mm": hava["son_5_gun_yagis_mm"],
                "olu_yakit_nemi_100saat": hava["olu_yakit_nemi_100saat"]
            },
            "anlik_hava": {
                "sicaklik": hava["anlik_sicaklik"],
                "nem": hava["anlik_nem"],
                "ruzgar_hizi": hava["anlik_ruzgar_hizi"],
                "ruzgar_yonu": hava["anlik_ruzgar_yonu"]
            },
            "bitki_ortusu": {
                "baskin_tur": bitki["baskin_bitki_ortusu"],
                "engel_sayisi": bitki["tespit_edilen_engel_hatti"],
                "ozel_katman_nesne": ozel_nesne_sayisi,
                "kaynak": bitki["kaynak"]
            },
            "arazi_yapisi": {
                "rakim_m": dem["merkez_rakim_m"],
                "ortalama_egim": dem["ortalama_egim_derece"],
                "baki": dem["merkez_baki"]
            }
        },
        "simulasyon": simulasyon
    }
