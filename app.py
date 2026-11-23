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
    version="3.0.0"
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
        "motor": "Orman Yangını Fizik Motoru v3.0 (Rothermel + Byram + Van Wagner + VPD)",
        "denklemler": [
            "Rothermel (1972) Yüzey Yayılım Denklemi",
            "Byram (1959) Yangın Hattı Şiddeti (kW/m)",
            "Van Wagner (1977) Tepe Yangını Geçiş Eşiği",
            "Balch vd. (2022) Gece/Gündüz VPD Bariyeri",
            "Viegas (2005) Dere/Kanyon Patlama Modeli",
            "Albini (1979) Kozalak Sıçraması & McAlpine (1991) Nokta Hızlanma"
        ]
    }


@app.post("/simule-et")
def simule_et_endpoint(istek: SimulasyonIstegi):
    hava = hava_durumu_getir(istek.enlem, istek.boylam)
    dem = arazi_dem_getir(istek.enlem, istek.boylam)
    bitki = bitki_ve_engeller_getir(istek.enlem, istek.boylam, dem)

    ozel_nesne_sayisi = 0
    if istek.ozel_katman_icerik:
        bitki["yakit_grid"], ozel_nesne_sayisi = ozel_katman_uygula(
            bitki["yakit_grid"],
            istek.enlem,
            istek.boylam,
            dem["yari_cap"],
            istek.ozel_katman_icerik,
            bitki["yakit_carpan_grid"],
            bitki["cbh_grid"]
        )

    simulasyon = yangin_simule_et(istek.enlem, istek.boylam, hava, dem, bitki)

    return {
        "koordinat": {"enlem": istek.enlem, "boylam": istek.boylam},
        "ozet_kartlar": {
            "son_5_gun": {
                "ort_sicaklik": hava["son_5_gun_ort_sicaklik"],
                "ort_nem": hava["son_5_gun_ort_nem"],
                "toplam_yagis_mm": hava["son_5_gun_yagis_mm"],
                "olu_yakit_nemi_100saat": hava["olu_yakit_nemi_100saat"],
                "canli_yaprak_nemi_lfmc": hava["canli_yaprak_nemi_lfmc"]
            },
            "anlik_hava": {
                "sicaklik": hava["anlik_sicaklik"],
                "nem": hava["anlik_nem"],
                "vpd_kpa": hava["anlik_vpd_kpa"],
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
