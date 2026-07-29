# %% [markdown]
# # Trafik Kazası Şiddeti Tahmini
#
# Bir kaza haberi duyduğumuzda ilk sorduğumuz soru hep aynıdır: "ağır mı?".
# Bu projede kazanın gerçekleştiği koşullara (hava durumu, yol yüzeyi, saat,
# sürücü/araç bilgileri gibi) bakarak kazanın şiddetini (hafif / ağır / ölümlü)
# önceden tahmin etmeye çalışıyoruz. Üç farklı sınıflandırma algoritmasını
# (Random Forest, KNN, LightGBM) hiper-parametre aramasıyla eğitip
# karşılaştıracağız ve en iyi modelin hangi özelliklere dayandığını SHAP ile
# açıklayacağız.
#
# **Not (veri seti seçimi):** Projenin taslağında "US Accidents" veri seti
# öneriliyordu; ancak o veri seti tek başına birkaç GB büyüklüğünde ve 8 GB
# RAM'li bir makinede işlemek risklidir. Bunun yerine Kaggle'daki
# `saurabhshahane/road-traffic-accidents` (Addis Ababa trafik kaza şiddeti
# verisi, ~12.300 satır, ~4 MB) kullanıldı — hedef değişken zaten
# `Accident_severity` olarak hazır geliyor ve hafiflik/hız gerekçesiyle bu
# ikame tercih edildi.

# %% [markdown]
# ## 1. Kütüphaneler ve veri yükleme

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)
from lightgbm import LGBMClassifier

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import shap

pd.set_option("display.max_columns", 40)
np.random.seed(42)

VERI_DIZINI = Path("veri")
GORSEL_DIZINI = Path("gorseller")
GORSEL_DIZINI.mkdir(exist_ok=True)

df = pd.read_csv(VERI_DIZINI / "RTA Dataset.csv")
print("Veri boyutu:", df.shape)
df.head(3)

# %% [markdown]
# ## 2. Renk paleti ve grafik teması
#
# Tüm grafiklerde tutarlı bir tema kullanıyoruz: sabit kategorik renk sırası
# (mavi, turuncu, camgöbeği, sarı, macenta, yeşil, mor, kırmızı), kaza şiddeti
# için sabit bir renk eşlemesi (Hafif = camgöbeği, Ağır = sarı/turuncu,
# Ölümlü = kırmızı — böylece hangi grafikte olursa olsun "ölümlü" hep aynı
# kırmızıyla gösteriliyor) ve yoğunluk haritaları için tek tonlu (mavi)
# artan-koyulaşan bir skala.

# %%
KATEGORIK_PALET = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SIDDET_RENK = {
    "Slight Injury": "#1baf7a",   # camgöbeği - hafif
    "Serious Injury": "#eda100",  # sarı/turuncu - ağır
    "Fatal injury": "#e34948",    # kırmızı - ölümlü
}
SIDDET_TR = {
    "Slight Injury": "Hafif Yaralanma",
    "Serious Injury": "Ağır Yaralanma",
    "Fatal injury": "Ölümlü",
}
# Türkçe etiketli grafikler için aynı renk eşlemesinin Türkçe anahtarlı hali
SIDDET_RENK_TR = {SIDDET_TR[k]: v for k, v in SIDDET_RENK.items()}
MAVI_SKALA = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

PLOTLY_TEMA = go.layout.Template()
PLOTLY_TEMA.layout = go.Layout(
    paper_bgcolor="#fcfcfb",
    plot_bgcolor="#fcfcfb",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color="#0b0b0b", size=13),
    title=dict(font=dict(size=18, color="#0b0b0b")),
    xaxis=dict(gridcolor="#e1e0d9", linecolor="#c3c2b7", zerolinecolor="#c3c2b7"),
    yaxis=dict(gridcolor="#e1e0d9", linecolor="#c3c2b7", zerolinecolor="#c3c2b7"),
    colorway=KATEGORIK_PALET,
)
pio.templates["trafik_tema"] = PLOTLY_TEMA
pio.templates.default = "trafik_tema"


def gorsel_kaydet(fig, dosya_adi, genislik=1100, yukseklik=750):
    """Bir plotly figürünü hem PNG hem HTML olarak gorseller/ klasörüne kaydeder.
    genislik/yukseklik: bazı grafiklerde (ör. SHAP) Türkçe eksen etiketleri daha
    uzun olduğu için varsayılandan geniş bir tuval gerekebilir."""
    png_yolu = GORSEL_DIZINI / f"{dosya_adi}.png"
    html_yolu = GORSEL_DIZINI / f"{dosya_adi}.html"
    fig.write_image(str(png_yolu), width=genislik, height=yukseklik, scale=2)
    fig.write_html(str(html_yolu), include_plotlyjs="cdn")
    print(f"Kaydedildi: {png_yolu} ({png_yolu.stat().st_size/1024:.0f} KB), {html_yolu} ({html_yolu.stat().st_size/1024:.0f} KB)")


# %% [markdown]
# ## 3. İlk bakış: eksik veri ve hedef değişken
#
# Veri setinde 32 sütun var; bazı sütunlarda (özellikle araç kusuru, araç yaşı,
# yaralının işi/uygunluğu gibi alanlarda) hatırı sayılır oranda eksik veri var.
# Bunları kategorik bir "Bilinmiyor" sınıfı olarak ele alacağız (silmek yerine,
# çünkü eksikliğin kendisi de bilgi taşıyabilir — örn. kaza raporunda o alan
# doldurulmamışsa bu da bir örüntü olabilir).

# %%
eksik = df.isnull().sum().sort_values(ascending=False)
eksik = eksik[eksik > 0]
print("Eksik veri içeren sütun sayısı:", len(eksik))
print((eksik / len(df) * 100).round(1).astype(str) + "%")

# %%
print(df["Accident_severity"].value_counts())
print()
print((df["Accident_severity"].value_counts(normalize=True) * 100).round(2).astype(str) + "%")

# %% [markdown]
# Hedef değişken ciddi biçimde dengesiz: kazaların ~%84.6'sı hafif, ~%14.2'si
# ağır, sadece ~%1.3'ü ölümlü. Bu yüzden bu projede **accuracy tek başına
# yanıltıcıdır** — %84 accuracy'yi "her şeyi hafif tahmin et" diyen aptal bir
# model bile yakalar. Değerlendirmede macro precision/recall/F1'e ve
# confusion matrix'e ağırlık vereceğiz, ayrıca modellerde `class_weight`
# dengesini kullanacağız.

# %%
siddet_sayim = df["Accident_severity"].value_counts().reset_index()
siddet_sayim.columns = ["Siddet", "Adet"]
siddet_sayim["Siddet_TR"] = siddet_sayim["Siddet"].map(SIDDET_TR)

fig = px.treemap(
    siddet_sayim, path=["Siddet_TR"], values="Adet",
    color="Siddet_TR", color_discrete_map=SIDDET_RENK_TR,
    title="Kaza Şiddeti Sınıflarının Payı (12.316 kaza)",
)
fig.update_traces(
    textinfo="label+value+percent root", textfont_size=16,
    hovertemplate="<b>%{label}</b><br>Kaza sayısı: %{value}<br>Toplamın %{percentRoot:.1%}'i<extra></extra>",
)
gorsel_kaydet(fig, "01_siddet_siniflari_treemap")
fig.show()

# %% [markdown]
# ## 4. Zaman özelliği: kazalar ne zaman yoğunlaşıyor?
#
# `Time` sütunu "SS:DD:SS" formatında metin olarak geliyor; buradan saat
# (0-23) bilgisini çıkarıp haftanın günüyle birlikte bir ısı haritasında
# inceliyoruz.

# %%
df["Saat"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

GUN_SIRASI_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GUN_TR = {
    "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
    "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar",
}

isi_matrisi = (
    df.groupby(["Day_of_week", "Saat"]).size()
    .reset_index(name="Adet")
    .pivot(index="Day_of_week", columns="Saat", values="Adet")
    .reindex(GUN_SIRASI_EN)
    .fillna(0)
)
isi_matrisi.index = [GUN_TR[g] for g in isi_matrisi.index]

fig = px.imshow(
    isi_matrisi,
    color_continuous_scale=MAVI_SKALA,
    labels=dict(x="Saat", y="Haftanın Günü", color="Kaza Adedi"),
    title="Saat × Haftanın Günü — Kaza Yoğunluğu Isı Haritası",
    aspect="auto",
)
fig.update_xaxes(dtick=1)
gorsel_kaydet(fig, "02_saat_gun_isi_haritasi")
fig.show()

# %% [markdown]
# **Gözlem:** Isı haritası iki net tepe gösteriyor — sabah 07:00-09:00 ve
# öğleden sonra 15:00-19:00 — ve bu iki bant özellikle hafta içi günlerde
# (Pazartesi-Cuma) koyulaşıyor; hafta sonu (Cumartesi-Pazar) tepe noktaları
# daha geç saatlere (öğleden sonra) kayıyor ve genel yoğunluk daha düşük.
# Bu, mesai gidiş-geliş trafiğiyle örtüşen bir örüntü (yazılmamış ama veride
# açıkça görünen bir kural: "trafik kazası riski, mesai saatleriyle birlikte
# iki ayrı zirve yapıyor").

# %% [markdown]
# ## 5. Coğrafi kaza haritası — ATLANDI
#
# **Not:** Bu veri setinde enlem/boylam (lat/lon) bilgisi bulunmuyor; kaza
# yeri sadece "Area_accident_occured" (ör. "Residential areas", "Office
# areas") gibi kategorik/genel bir alan olarak veriliyor, gerçek koordinat
# yok. Bu yüzden Folium ile nokta bazlı bir coğrafi harita üretmek mümkün
# değil ve bu adım bilinçli olarak atlandı. Yerine, kaza alanı türünü
# aşağıdaki hava durumu grafiğine benzer bir kırılımda (bkz. bölüm 6) metinsel
# olarak inceliyoruz.

# %%
alan_sayim = (
    df["Area_accident_occured"].fillna("Bilinmiyor").value_counts().head(10)
)
print("En sık 10 kaza alanı türü:")
print(alan_sayim)

# %% [markdown]
# Gerçek koordinat olmasa da elimizdeki "en yakın coğrafi bilgi" kaza alanı
# türü (`Area_accident_occured`) — bunu, eksik haritanın yerine geçecek altıncı
# görsel olarak alan türü × şiddet kırılımında bir icicle grafiğiyle inceliyoruz.
# Nadir alan türlerini (50 örneğin altı) yine "Diğer" altında topluyoruz.

# %%
# Kaza alanı türü (Area_accident_occured) kategorilerinin görselde gösterilecek
# Türkçe karşılığı — sütun adı ve ham veri değerleri koddan/modelden etkilenmez,
# sadece bu grafikte yazılan etiket metni değişiyor. "Other" veri setinin kendi
# kategorisi; bizim "Diğer (az örnek)" toplama grubumuzla karışmaması için ayrı
# çevrildi.
ALAN_TR = {
    "Other": "Diğer Alan Türü",
    "Office areas": "Ofis Alanları",
    "Residential areas": "Yerleşim Alanları",
    "Church areas": "Kilise Çevresi",
    "Industrial areas": "Sanayi Alanları",
    "School areas": "Okul Çevresi",
    "Recreational areas": "Rekreasyon Alanları",
    "Outside rural areas": "Kırsal Alan Dışı",
    "Hospital areas": "Hastane Çevresi",
    "Market areas": "Pazar/Çarşı Alanları",
    "Rural village areas": "Kırsal Köy Alanları",
    "Bilinmiyor": "Bilinmiyor",
}

df["Alan_Turu"] = df["Area_accident_occured"].fillna("Bilinmiyor").str.strip()
alan_ham_sayim = df["Alan_Turu"].value_counts()
sik_alan = alan_ham_sayim[alan_ham_sayim >= 50].index.tolist()
df["Alan_Grup"] = df["Alan_Turu"].apply(
    lambda x: ALAN_TR.get(x, x) if x in sik_alan else "Diğer (az örnek)"
)
df["Siddet_TR"] = df["Accident_severity"].map(SIDDET_TR)

alan_siddet = df.groupby(["Alan_Grup", "Siddet_TR"]).size().reset_index(name="Adet")

fig = px.icicle(
    alan_siddet, path=["Alan_Grup", "Siddet_TR"], values="Adet",
    color="Siddet_TR", color_discrete_map=SIDDET_RENK_TR,
    title="Kaza Alanı Türüne Göre Şiddet Dağılımı (Coğrafi Harita Yerine)",
)
fig.update_traces(
    hovertemplate="<b>%{label}</b><br>Kaza sayısı: %{value}<br>Üst kırılıma göre oran: %{percentParent:.1%}<extra></extra>",
)
gorsel_kaydet(fig, "03_alan_turu_siddet_icicle")
fig.show()

# %% [markdown]
# ## 6. Hava durumuna göre şiddet dağılımı
#
# Nadir görülen hava durumu kategorilerini ("Fog or mist", "Raining and
# Windy", "Snow" gibi 100 örneğin altında kalanları) tek başına yorumlamak
# istatistiksel olarak anlamsız olur, bu yüzden bunları "Diğer (az örnek)"
# altında birleştirip sunburst grafiğine öyle taşıyoruz.

# %%
# Hava durumu (Weather_conditions) kategorilerinin görselde gösterilecek Türkçe
# karşılığı — aynı mantık: "Other" veri setinin kendi kategorisi, "Diğer (az
# örnek)" toplama grubumuzla karışmaması için ayrı çevrildi.
HAVA_TR = {
    "Normal": "Normal",
    "Raining": "Yağmurlu",
    "Other": "Diğer Hava Durumu",
    "Unknown": "Bilinmiyor",
    "Cloudy": "Bulutlu",
}

hava_sayim = df["Weather_conditions"].value_counts()
sik_hava = hava_sayim[hava_sayim >= 100].index.tolist()
df["Hava_Grup"] = df["Weather_conditions"].apply(
    lambda x: HAVA_TR.get(x, x) if x in sik_hava else "Diğer (az örnek)"
)

hava_siddet = (
    df.groupby(["Hava_Grup", "Siddet_TR"]).size().reset_index(name="Adet")
)

fig = px.sunburst(
    hava_siddet, path=["Hava_Grup", "Siddet_TR"], values="Adet",
    color="Siddet_TR", color_discrete_map=SIDDET_RENK_TR,
    title="Hava Durumuna Göre Kaza Şiddeti Dağılımı",
)
fig.update_traces(
    hovertemplate="<b>%{label}</b><br>Kaza sayısı: %{value}<br>Üst kırılıma göre oran: %{percentParent:.1%}<extra></extra>",
)
gorsel_kaydet(fig, "04_hava_durumu_siddet_sunburst")
fig.show()

# %% [markdown]
# **Gözlem:** Kazaların büyük çoğunluğu zaten "Normal" hava koşullarında
# gerçekleşiyor (Addis Ababa'nın iklimi göz önüne alındığında beklenir); yağmurlu
# koşullarda ağır/ölümlü payının hafifçe arttığı görülüyor ama örneklem küçük
# olduğu için bu gözlemi HYPOTHESIS (güven: MEDIUM) olarak not düşüyoruz, kesin
# kural değil.

# %% [markdown]
# ## 7. Ön işleme
#
# **Not (veri sızıntısı kararı):** `Casualty_severity` sütunu, yaralının kendi
# şiddet skorunu taşıyor ve kavramsal olarak hedef değişkenle (`Accident_severity`)
# aynı anda, aynı olayın sonucunu ölçüyor — bunu özellik olarak kullanmak
# hedefin bir başka ölçümüyle hedefi tahmin etmek anlamına gelir (sızıntı).
# Bu yüzden bilinçli olarak özellik setinden ÇIKARILDI. Diğer yaralı-bazlı
# sütunlar (yaş, cinsiyet, meslek vb.) kaza anında/raporunda kayda geçen
# betimleyici bilgiler olarak tutuldu; ancak bunların da "kaza olmadan önce"
# bilinen bilgiler olmadığını, kaza raporunun bir parçası olduğunu dürüstçe
# belirtmek gerekir.

# %%
HEDEF = "Accident_severity"
SIZINTI_SUTUNLARI = ["Casualty_severity"]
SAYISAL_OZELLIKLER = ["Number_of_vehicles_involved", "Number_of_casualties", "Saat"]

# object/str dtype sütunları elle topla (pandas 3.0'da string dtype ayrımı için
# select_dtypes yerine pd.api.types.is_string_dtype kullanıyoruz).
# Not: Hava_Grup / Alan_Turu / Alan_Grup sadece görselleştirme için türetilen
# yardımcı sütunlar; Area_accident_occured ve Weather_conditions ile aynı
# bilgiyi taşıdıkları için modelde tekrar kullanılmıyor (gereksiz çoğullama).
GORSEL_YARDIMCI_SUTUNLARI = ["Hava_Grup", "Alan_Turu", "Alan_Grup", "Siddet_TR"]
kategorik_ozellikler = [
    c for c in df.columns
    if c not in SAYISAL_OZELLIKLER + [HEDEF, "Time"] + SIZINTI_SUTUNLARI + GORSEL_YARDIMCI_SUTUNLARI
    and pd.api.types.is_string_dtype(df[c])
]

print("Sayısal özellik sayısı:", len(SAYISAL_OZELLIKLER))
print("Kategorik özellik sayısı:", len(kategorik_ozellikler))

X = df[SAYISAL_OZELLIKLER + kategorik_ozellikler].copy()
for c in kategorik_ozellikler:
    X[c] = X[c].fillna("Bilinmiyor")

y_ham = df[HEDEF]
le = LabelEncoder()
y = le.fit_transform(y_ham)
print("Sınıf eşlemesi:", dict(zip(le.classes_, range(len(le.classes_)))))

on_isleme = ColumnTransformer(
    transformers=[
        ("sayisal", StandardScaler(), SAYISAL_OZELLIKLER),
        ("kategorik", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kategorik_ozellikler),
    ]
)

X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Eğitim boyutu:", X_egitim.shape, "Test boyutu:", X_test.shape)

cv_semasi = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
# ## 8. Model 1 — Random Forest (RandomizedSearchCV, 5-fold CV)

# %%
rf_pipeline = Pipeline([
    ("on_isleme", on_isleme),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42)),
])

rf_izgara = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}

rf_arama = RandomizedSearchCV(
    rf_pipeline, rf_izgara, n_iter=20, cv=cv_semasi, scoring="f1_macro",
    random_state=42, n_jobs=-1, verbose=0,
)
rf_arama.fit(X_egitim, y_egitim)
print("RF en iyi parametreler:", rf_arama.best_params_)
print("RF CV f1_macro (en iyi):", round(rf_arama.best_score_, 4))
rf_en_iyi = rf_arama.best_estimator_

# %% [markdown]
# ## 9. Model 2 — KNN (GridSearchCV, 5-fold CV)
#
# KNN mesafe tabanlı olduğu için tek-sıcak (one-hot) kodlanmış kategorik
# değişkenlerle boyut biraz şişer; bu yüzden `n_neighbors` aralığını daha
# geniş tutuyoruz (küçük k değerleri gürültüye çok duyarlı olur).

# %%
knn_pipeline = Pipeline([
    ("on_isleme", on_isleme),
    ("model", KNeighborsClassifier()),
])

knn_izgara = {
    "model__n_neighbors": [5, 9, 15, 21, 31],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["minkowski", "manhattan"],
}

knn_arama = GridSearchCV(
    knn_pipeline, knn_izgara, cv=cv_semasi, scoring="f1_macro", n_jobs=-1, verbose=0,
)
knn_arama.fit(X_egitim, y_egitim)
print("KNN en iyi parametreler:", knn_arama.best_params_)
print("KNN CV f1_macro (en iyi):", round(knn_arama.best_score_, 4))
knn_en_iyi = knn_arama.best_estimator_

# %% [markdown]
# ## 10. Model 3 — LightGBM (RandomizedSearchCV, 5-fold CV)
#
# Üçüncü ve en gelişmiş model olarak LightGBM'i seçtik (XGBoost yerine):
# çok sınıflı dengesizlikte `class_weight="balanced"` parametresini doğrudan
# destekliyor, bu da manuel `sample_weight` hesaplama karmaşasını ortadan
# kaldırıyor.

# %%
lgbm_pipeline = Pipeline([
    ("on_isleme", on_isleme),
    ("model", LGBMClassifier(
        class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1,
    )),
])

lgbm_izgara = {
    "model__n_estimators": [100, 300, 500],
    "model__num_leaves": [15, 31, 63],
    "model__max_depth": [-1, 5, 10, 15],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__min_child_samples": [5, 10, 20],
}

lgbm_arama = RandomizedSearchCV(
    lgbm_pipeline, lgbm_izgara, n_iter=20, cv=cv_semasi, scoring="f1_macro",
    random_state=42, n_jobs=-1, verbose=0,
)
lgbm_arama.fit(X_egitim, y_egitim)
print("LightGBM en iyi parametreler:", lgbm_arama.best_params_)
print("LightGBM CV f1_macro (en iyi):", round(lgbm_arama.best_score_, 4))
lgbm_en_iyi = lgbm_arama.best_estimator_

# %% [markdown]
# ## 11. Test seti karşılaştırması
#
# Üç modeli de test setinde (eğitim sırasında hiç görmedikleri veri) tek tek
# değerlendiriyoruz. Accuracy'nin yanına macro precision/recall/F1 ekliyoruz —
# dengesiz sınıflarda asıl belirleyici olan bunlar.

# %%
modeller = {
    "Random Forest": rf_en_iyi,
    "KNN": knn_en_iyi,
    "LightGBM": lgbm_en_iyi,
}

sonuclar = []
tahminler = {}
for isim, model in modeller.items():
    y_tahmin = model.predict(X_test)
    tahminler[isim] = y_tahmin
    acc = accuracy_score(y_test, y_tahmin)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_tahmin, average="macro", zero_division=0)
    sonuclar.append({
        "Model": isim, "Accuracy": acc,
        "Macro Precision": prec, "Macro Recall": rec, "Macro F1": f1,
    })

sonuc_tablosu = pd.DataFrame(sonuclar).sort_values("Macro F1", ascending=False).reset_index(drop=True)
print(sonuc_tablosu.round(4))

en_iyi_model_adi = sonuc_tablosu.iloc[0]["Model"]
print("\nEn iyi model (Macro F1'e göre):", en_iyi_model_adi)

# %%
print(f"\n--- {en_iyi_model_adi} sınıflandırma raporu ---")
print(classification_report(y_test, tahminler[en_iyi_model_adi], target_names=le.classes_, zero_division=0))

# %% [markdown]
# ## 12. Confusion matrix (en iyi model)

# %%
cm = confusion_matrix(y_test, tahminler[en_iyi_model_adi])
etiketler_tr = [SIDDET_TR[c] for c in le.classes_]

fig = px.imshow(
    cm, x=etiketler_tr, y=etiketler_tr,
    color_continuous_scale=MAVI_SKALA,
    labels=dict(x="Tahmin Edilen", y="Gerçek Sınıf", color="Adet"),
    title=f"Karışıklık Matrisi — {en_iyi_model_adi} (Test Seti)",
    text_auto=True,
)
fig.update_xaxes(side="bottom")
gorsel_kaydet(fig, "05_confusion_matrix_en_iyi_model")
fig.show()

# %% [markdown]
# ## 13. Özellik önemi ve SHAP özeti
#
# SHAP açıklaması için ağaç tabanlı modeller (Random Forest / LightGBM)
# arasında test setinde Macro F1'i daha yüksek olanı seçiyoruz — KNN için
# `TreeExplainer` kullanılamaz (ağaç yapısı yok) ve `KernelExplainer` bu
# boyuttaki tek-sıcak kodlanmış veri için çok yavaş kalır, bu yüzden SHAP
# analizini ağaç tabanlı en iyi modelle sınırlıyoruz.

# %%
agac_modeller = sonuc_tablosu[sonuc_tablosu["Model"].isin(["Random Forest", "LightGBM"])]
shap_model_adi = agac_modeller.sort_values("Macro F1", ascending=False).iloc[0]["Model"]
shap_pipeline = modeller[shap_model_adi]
print("SHAP için seçilen model:", shap_model_adi)

on_isleme_fit = shap_pipeline.named_steps["on_isleme"]
model_fit = shap_pipeline.named_steps["model"]

ozellik_isimleri = (
    SAYISAL_OZELLIKLER
    + list(on_isleme_fit.named_transformers_["kategorik"].get_feature_names_out(kategorik_ozellikler))
)

X_test_donusturulmus = on_isleme_fit.transform(X_test)
# SHAP hesaplama maliyetini sınırlamak için test setinden 500 örneklik bir alt küme
ornek_sayisi = min(500, X_test_donusturulmus.shape[0])
np.random.seed(42)
ornek_idx = np.random.choice(X_test_donusturulmus.shape[0], ornek_sayisi, replace=False)
X_shap = X_test_donusturulmus[ornek_idx]

aciklayici = shap.TreeExplainer(model_fit)
shap_degerleri = aciklayici.shap_values(X_shap)

# çok sınıflı çıktıda shap_values bir liste/3B dizi döner; sınıflar üzerinden
# ortalama mutlak etkiyi alarak genel özellik önemini çıkarıyoruz
if isinstance(shap_degerleri, list):
    shap_array = np.abs(np.stack(shap_degerleri, axis=0)).mean(axis=0)
elif shap_degerleri.ndim == 3:
    shap_array = np.abs(shap_degerleri).mean(axis=2)
else:
    shap_array = np.abs(shap_degerleri)

ortalama_etki = shap_array.mean(axis=0)
onem_df = pd.DataFrame({"Ozellik": ozellik_isimleri, "Ortalama_SHAP_Etki": ortalama_etki})
onem_df = onem_df.sort_values("Ortalama_SHAP_Etki", ascending=False).head(15)

# --- Bu grafiğe özel Türkçeleştirme yardımcıları ---
# Gerçek sütun/değişken adları (Age_band_of_driver, Cause_of_accident vb.)
# koddan ve modelden etkilenmeden İngilizce/orijinal kalır; sadece SHAP
# grafiğinin ekseninde görünen metne parantez içinde kısa bir Türkçe açıklama
# ekleniyor (ör. "Cause_of_accident_Overtaking (Kaza nedeni: Sollama)").
SUTUN_ACIKLAMA_TR = {
    "Day_of_week": "Haftanın günü", "Age_band_of_driver": "Sürücü yaş grubu",
    "Sex_of_driver": "Sürücü cinsiyeti", "Educational_level": "Eğitim düzeyi",
    "Vehicle_driver_relation": "Sürücünün araçla ilişkisi", "Driving_experience": "Sürüş deneyimi",
    "Type_of_vehicle": "Araç türü", "Owner_of_vehicle": "Araç sahipliği",
    "Service_year_of_vehicle": "Aracın kullanım yaşı", "Defect_of_vehicle": "Araç arızası",
    "Area_accident_occured": "Kaza alanı türü", "Lanes_or_Medians": "Şerit/refüj türü",
    "Road_allignment": "Yol güzergahı", "Types_of_Junction": "Kavşak türü",
    "Road_surface_type": "Yol yüzeyi türü", "Road_surface_conditions": "Yol yüzeyi durumu",
    "Light_conditions": "Işık/aydınlatma durumu", "Weather_conditions": "Hava durumu",
    "Type_of_collision": "Çarpışma türü", "Vehicle_movement": "Araç hareketi",
    "Casualty_class": "Yaralı sınıfı", "Sex_of_casualty": "Yaralı cinsiyeti",
    "Age_band_of_casualty": "Yaralı yaş grubu", "Work_of_casuality": "Yaralının mesleği",
    "Fitness_of_casuality": "Yaralının sağlık durumu", "Pedestrian_movement": "Yaya hareketi",
    "Cause_of_accident": "Kaza nedeni",
    "Number_of_vehicles_involved": "Kazaya karışan araç sayısı",
    "Number_of_casualties": "Yaralı/kayıp sayısı",
}
DEGER_ACIKLAMA_TR = {
    "Unknown": "Bilinmiyor", "unknown": "bilinmiyor", "Other": "Diğer", "other": "diğer",
    "na": "Yok/Uygulanamaz", "No defect": "Arıza yok",
    "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba", "Thursday": "Perşembe",
    "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar",
    "Male": "Erkek", "Female": "Kadın",
    "Under 18": "18 yaş altı", "Over 51": "51 yaş üstü",
    "Above high school": "Lise üstü", "Junior high school": "Ortaokul",
    "Elementary school": "İlkokul", "High school": "Lise", "Illiterate": "Okuma yazma bilmiyor",
    "Writing & reading": "Okuma yazma biliyor",
    "Employee": "Çalışan", "Owner": "Sahibi",
    "Above 10yr": "10 yıldan fazla", "5-10yr": "5-10 yıl", "5-10yrs": "5-10 yıl",
    "2-5yr": "2-5 yıl", "2-5yrs": "2-5 yıl", "1-2yr": "1-2 yıl", "Below 1yr": "1 yıldan az",
    "No Licence": "Ehliyetsiz",
    "Automobile": "Otomobil", "Public (> 45 seats)": "Toplu taşıma (45+ koltuk)",
    "Lorry (41?100Q)": "Kamyon (41-100Q)", "Public (13?45 seats)": "Toplu taşıma (13-45 koltuk)",
    "Lorry (11?40Q)": "Kamyon (11-40Q)", "Long lorry": "Büyük kamyon (TIR)",
    "Public (12 seats)": "Toplu taşıma (12 koltuk)", "Taxi": "Taksi",
    "Pick up upto 10Q": "Kamyonet (10Q'a kadar)", "Stationwagen": "Steyşın",
    "Ridden horse": "Binek at", "Bajaj": "Bajaj (üç tekerlekli motorlu)",
    "Turbo": "Turbo (üç tekerlekli motorlu)", "Motorcycle": "Motosiklet",
    "Special vehicle": "Özel araç", "Bicycle": "Bisiklet",
    "Governmental": "Devlet", "Organization": "Kurum/Şirket",
    "Undivided Two way": "Bölünmemiş iki yönlü", "Double carriageway (median)": "Bölünmüş yol (refüjlü)",
    "One way": "Tek yönlü",
    "Two-way (divided with solid lines road marking)": "İki yönlü (düz çizgi ile ayrılmış)",
    "Two-way (divided with broken lines road marking)": "İki yönlü (kesikli çizgi ile ayrılmış)",
    "Tangent road with flat terrain": "Düz arazide düz yol",
    "Tangent road with mild grade and flat terrain": "Hafif eğimli düz arazi yolu",
    "Escarpments": "Sarp yamaç", "Tangent road with rolling terrain": "Dalgalı arazide düz yol",
    "Gentle horizontal curve": "Hafif yatay viraj",
    "Tangent road with mountainous terrain and": "Dağlık arazide düz yol",
    "Steep grade downward with mountainous terrain": "Dağlık arazide dik iniş",
    "Sharp reverse curve": "Keskin ters viraj",
    "Steep grade upward with mountainous terrain": "Dağlık arazide dik yokuş",
    "No junction": "Kavşak yok", "Y Shape": "Y kavşağı", "Crossing": "Dört yol ağzı",
    "O Shape": "O (dönel) kavşak", "T Shape": "T kavşağı", "X Shape": "X kavşağı",
    "Asphalt roads": "Asfalt yol", "Earth roads": "Toprak yol",
    "Asphalt roads with some distress": "Bozuk asfalt yol", "Gravel roads": "Çakıl yol",
    "Dry": "Kuru", "Wet or damp": "Islak/nemli", "Snow": "Karlı",
    "Flood over 3cm. deep": "3 cm üstü su birikintisi",
    "Daylight": "Gündüz", "Darkness - lights lit": "Karanlık - aydınlatma açık",
    "Darkness - no lighting": "Karanlık - aydınlatma yok",
    "Darkness - lights unlit": "Karanlık - aydınlatma kapalı",
    "Normal": "Normal", "Raining": "Yağmurlu", "Raining and Windy": "Yağmurlu ve rüzgarlı",
    "Cloudy": "Bulutlu", "Windy": "Rüzgarlı", "Fog or mist": "Sisli",
    "Collision with roadside-parked vehicles": "Yol kenarında park halindeki araca çarpma",
    "Vehicle with vehicle collision": "Araç-araç çarpışması",
    "Collision with roadside objects": "Yol kenarındaki cisme çarpma",
    "Collision with animals": "Hayvana çarpma", "Rollover": "Takla atma",
    "Fall from vehicles": "Araçtan düşme", "Collision with pedestrians": "Yayaya çarpma",
    "With Train": "Trenle çarpışma",
    "Going straight": "Düz gidiş", "U-Turn": "U dönüşü", "Moving Backward": "Geri gidiş",
    "Turnover": "Devrilme", "Waiting to go": "Hareket için bekleme", "Getting off": "İnme",
    "Reversing": "Geri manevra", "Parked": "Park halinde", "Stopping": "Durma",
    "Overtaking": "Sollama", "Entering a junction": "Kavşağa girme",
    "Driver or rider": "Sürücü", "Pedestrian": "Yaya", "Passenger": "Yolcu",
    "Driver": "Şoför", "Unemployed": "İşsiz", "Self-employed": "Serbest meslek", "Student": "Öğrenci",
    "Deaf": "İşitme engelli", "Blind": "Görme engelli", "NormalNormal": "Normal",
    "Not a Pedestrian": "Yaya değil",
    "Crossing from driver's nearside": "Sürücünün yakın tarafından geçiş",
    "Unknown or other": "Bilinmiyor/diğer",
    "Walking along in carriageway, back to traffic": "Yolda trafiğe sırtı dönük yürüme",
    "Walking along in carriageway, facing traffic": "Yolda trafiğe dönük yürüme",
    "Changing lane to the left": "Sola şerit değiştirme",
    "Changing lane to the right": "Sağa şerit değiştirme",
    "Overloading": "Aşırı yükleme", "No priority to vehicle": "Araca öncelik vermeme",
    "No priority to pedestrian": "Yayaya öncelik vermeme", "No distancing": "Takip mesafesi bırakmama",
    "Getting off the vehicle improperly": "Araçtan hatalı inme", "Improper parking": "Hatalı park",
    "Overspeed": "Hız sınırını aşma", "Driving carelessly": "Dikkatsiz sürüş",
    "Driving at high speed": "Yüksek hızla sürüş", "Driving to the left": "Soldan sürüş",
    "Overturning": "Devrilme", "Driving under the influence of drugs": "Uyuşturucu etkisiyle sürüş",
    "Drunk driving": "Alkollü sürüş",
}


def ozellik_adi_turkce(ohe_adi, kategorik_sutunlar):
    """Tek-sıcak kodlanmış özellik adını, sütun adı İngilizce kalacak şekilde,
    parantez içinde kısa bir Türkçe açıklamayla döndürür. 'Saat' zaten Türkçe
    olduğu için olduğu gibi bırakılır."""
    if ohe_adi == "Saat":
        return ohe_adi
    eslesen_sutun = max(
        (s for s in kategorik_sutunlar if ohe_adi.startswith(s + "_")),
        key=len, default=None,
    )
    if eslesen_sutun is None:
        aciklama = SUTUN_ACIKLAMA_TR.get(ohe_adi)
        return f"{ohe_adi} ({aciklama})" if aciklama else ohe_adi
    deger = ohe_adi[len(eslesen_sutun) + 1:]
    sutun_tr = SUTUN_ACIKLAMA_TR.get(eslesen_sutun, eslesen_sutun)
    deger_tr = DEGER_ACIKLAMA_TR.get(deger)
    aciklama = f"{sutun_tr}: {deger_tr}" if deger_tr else sutun_tr
    return f"{ohe_adi} ({aciklama})"


onem_df["Ozellik_Goster"] = onem_df["Ozellik"].apply(
    lambda o: ozellik_adi_turkce(o, kategorik_ozellikler)
)

fig = px.bar(
    onem_df.sort_values("Ortalama_SHAP_Etki"),
    x="Ortalama_SHAP_Etki", y="Ozellik_Goster", orientation="h",
    color_discrete_sequence=[KATEGORIK_PALET[0]],
    title=f"SHAP Özellik Önemi (Ortalama |Etki|) — {shap_model_adi}",
    labels={"Ortalama_SHAP_Etki": "Ortalama |SHAP değeri|", "Ozellik_Goster": ""},
)
fig.update_yaxes(automargin=True)
fig.update_traces(
    hovertemplate="<b>%{y}</b><br>Ortalama |SHAP değeri|: %{x:.4f}<extra></extra>",
)
gorsel_kaydet(fig, "06_shap_ozellik_onemi", genislik=1700, yukseklik=800)
fig.show()

# %% [markdown]
# **Gözlem:** En etkili özelliklerin başında saat (`Saat`), kazaya karışan
# araç/yaralı sayısı ve çarpışma türü geliyor — bu, "kaç aracın işin içinde
# olduğu ve ne zaman olduğu" bilgisinin şiddeti belirlemede hava durumundan
# daha güçlü bir sinyal taşıdığını gösteriyor (HYPOTHESIS, güven: MEDIUM-HIGH,
# tek veri setine dayanıyor, çapraz doğrulama için başka bir kaza veri seti
# gerekir).

# %% [markdown]
# ## 14. Sonuç
#
# Üç model de aynı 5-fold çapraz doğrulama ve aynı ön işleme hattından geçti;
# aralarındaki fark yalnızca algoritma ve hiper-parametrelerden geliyor. Nihai
# tabloyu ve dosya çıktısını aşağıda özetliyoruz.

# %%
print("=== NİHAİ KARŞILAŞTIRMA TABLOSU ===")
print(sonuc_tablosu.round(4).to_string(index=False))

sonuc_tablosu.round(4).to_csv(GORSEL_DIZINI.parent / "sonuc_tablosu.csv", index=False)
print("\nÜretilen görseller:")
for f in sorted(GORSEL_DIZINI.glob("*")):
    print(" -", f.name, f"({f.stat().st_size/1024:.0f} KB)")
