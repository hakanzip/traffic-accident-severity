# Trafik Kazası Şiddeti Tahmini

Bir kaza haberi duyduğumuzda ilk merak ettiğimiz şey hep aynıdır: "ağır mı?"
Bu projede kazanın gerçekleştiği koşullara bakarak (hava durumu, saat, yol
yüzeyi, sürücü ve araç bilgileri gibi) kazanın şiddetini -hafif, ağır ya da
ölümlü- önceden tahmin etmeye çalıştık.

## Veri seti

Taslakta "US Accidents" veri seti öneriliyordu, ama o veri seti tek başına
birkaç GB büyüklüğünde ve 8 GB RAM'li bir makinede işlemek riskli olurdu. Bu
yüzden Kaggle'daki `saurabhshahane/road-traffic-accidents` veri setini
kullandık: Addis Ababa'da kayıtlı 12.316 kaza, ~4 MB, hedef değişken zaten
`Accident_severity` adıyla hazır geliyor. Daha küçük ve tamamen tabular olduğu
için hem hızlı işleniyor hem de bu makinede risksiz.

Hedef değişken ciddi ölçüde dengesiz: kazaların %84,6'sı hafif, %14,2'si ağır,
sadece %1,3'ü ölümlü. Bu yüzden accuracy tek başına yanıltıcı - her şeyi
"hafif" tahmin eden bir model bile %84 civarı accuracy'ye ulaşır. Değerlendirmeyi
macro precision/recall/F1 ve confusion matrix üzerinden yaptık, modellerde de
`class_weight="balanced"` kullandık.

**Veri sızıntısı kararı:** `Casualty_severity` sütunu yaralının kendi şiddet
skorunu taşıyor ve kavramsal olarak hedefle aynı olayı ölçüyor; özellik
setinden bilinçli olarak çıkarıldı. Diğer yaralı-bazlı sütunlar (yaş, meslek
vb.) tutuldu, ama bunların da kaza *anında* kayda geçen bilgiler olduğunu,
kaza öncesinden bilinen bir şey olmadığını belirtmek gerekiyor.

## Modeller

Üç algoritmayı aynı ön işleme hattından (eksik kategorik değerler
"Bilinmiyor" ile dolduruldu, sayısallar StandardScaler, kategorikler
OneHotEncoder) geçirip 5 katlı çapraz doğrulamayla (StratifiedKFold) karşılaştırdık:

- **Random Forest** - RandomizedSearchCV (20 kombinasyon), `class_weight="balanced"`
- **KNN** - GridSearchCV (tam ızgara), mesafe/komşu sayısı/ağırlık taraması
- **LightGBM** - RandomizedSearchCV (20 kombinasyon), `class_weight="balanced"`

Arama skoru olarak `f1_macro` kullandık çünkü asıl önemsediğimiz şey nadir
görülen ama kritik olan "ölümlü" sınıfını kaçırmamak; accuracy bu konuda kör.

## Sonuçlar

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| LightGBM | 0.737 | 0.467 | 0.473 | 0.464 |
| Random Forest | 0.795 | 0.448 | 0.411 | 0.424 |
| KNN | 0.833 | 0.420 | 0.345 | 0.334 |

En yüksek accuracy'ye KNN ulaşıyor (0.833) ama macro F1'de en düşük performansı
o veriyor - çünkü büyük olasılıkla çoğunluk sınıfını (hafif) ezici biçimde
tahmin edip nadir sınıflarda başarısız oluyor. LightGBM, accuracy'de üçüncü
sırada kalmasına rağmen macro F1'de en iyisi, çünkü ölümlü ve ağır sınıflarda
diğerlerinden daha dengeli tahmin veriyor. Bu yüzden **LightGBM'i nihai model
seçtik** - bu projenin amacı zaten "nadir ama kritik olan ölümlü kazayı
gözden kaçırmamak", ham accuracy değil.

Dürüst olmak gerekirse üç modelin de macro F1'i mütevazı (0,33-0,46 arası) -
"ölümlü" sınıfında sadece 158 örnek olması ve hava/yol koşullarının kazanın
şiddetini tek başına belirlememesi (birçok dışsal faktör de var - hız, sürücü
tepkisi, kaza anı detayları veri setinde yok) bu tavanı zorluyor. Bu bir
tarama/önceliklendirme aracı olarak faydalı olabilir ama tek başına karar
mercii olarak kullanılmamalı.

## Görseller

`gorseller/` klasöründe hem PNG hem interaktif HTML olarak duruyor:

1. `01_siddet_siniflari_treemap` - kaza şiddeti sınıflarının payı
2. `02_saat_gun_isi_haritasi` - saat × haftanın günü kaza yoğunluğu ısı
   haritası. İki net zirve var: sabah 07:00-09:00 ve öğleden sonra
   15:00-19:00, özellikle hafta içi günlerde koyulaşıyor - mesai gidiş-geliş
   trafiğiyle örtüşen bir örüntü.
3. `03_alan_turu_siddet_icicle` - **coğrafi harita yerine geçen görsel.** Veri
   setinde enlem/boylam bilgisi yok (kaza yeri sadece "Office areas",
   "Residential areas" gibi genel bir kategori olarak veriliyor), bu yüzden
   Folium ile nokta bazlı harita üretmek mümkün değildi; bunun yerine kaza
   alanı türüne göre şiddet dağılımını inceledik.
4. `04_hava_durumu_siddet_sunburst` - hava durumuna göre şiddet dağılımı.
   Kazaların büyük çoğunluğu zaten "Normal" hava koşulunda oluyor.
5. `05_confusion_matrix_en_iyi_model` - LightGBM'in test setindeki confusion matrix'i
6. `06_shap_ozellik_onemi` - SHAP özellik önemi (LightGBM üzerinden; KNN ağaç
   yapısında olmadığı için SHAP açıklaması ağaç tabanlı en iyi modelle sınırlı tutuldu)

## Notebook

`proje.py` önce düz Python script olarak yazılıp çalıştırıldı, `jupytext` ile
`proje.ipynb`'ye çevrildi, sonra `jupyter nbconvert --execute` ile gerçekten
baştan sona çalıştırılıp hücre çıktıları üretildi.

## Kullanılan kütüphaneler

- [pandas](https://pandas.pydata.org/docs/) - veri işleme
- [numpy](https://numpy.org/doc/) - sayısal işlemler
- [scikit-learn](https://scikit-learn.org/stable/) - Random Forest, KNN, ön işleme, GridSearchCV/RandomizedSearchCV
- [LightGBM](https://lightgbm.readthedocs.io/) - gradyan artırımlı ağaç modeli
- [SHAP](https://shap.readthedocs.io/) - model açıklanabilirliği
- [Plotly](https://plotly.com/python/) - tüm görselleştirmeler
- [Kaleido](https://github.com/plotly/Kaleido) - Plotly grafiklerini PNG'ye aktarma
- [Jupytext](https://jupytext.readthedocs.io/) - .py ↔ .ipynb dönüşümü
- [nbconvert](https://nbconvert.readthedocs.io/) - notebook'u çalıştırıp kaydetme
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) - veri setini indirmek için
