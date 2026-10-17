# Penguen Sınıflandırma

Palmer takımadalarında (Antarktika) ölçülen penguenleri gaga ölçüleri, yüzgeç uzunluğu, kilo, ada ve cinsiyet bilgisine göre üç türe ayırır: Adelie, Chinstrap, Gentoo.

Iris projesinden farklı olarak bu projede eksik değerler, kategorik özellikler ve farklı ölçeklerdeki sayısal özellikler var. Ön işleme scikit-learn ile, model PyTorch ile yazıldı.

## Veri

- **Kaynak:** [Palmer Archipelago (Antarctica) Penguin Data – Kaggle](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data) (CC0)
- **Dosya:** `penguins_size.csv`, 344 satır

| Sütun | Açıklama |
|---|---|
| `species` | Hedef: Adelie (152), Gentoo (124), Chinstrap (68) |
| `island` | Biscoe, Dream, Torgersen |
| `culmen_length_mm`, `culmen_depth_mm` | Gaga uzunluğu ve derinliği |
| `flipper_length_mm` | Yüzgeç uzunluğu |
| `body_mass_g` | Kilo |
| `sex` | MALE / FEMALE |

## Veri temizliği

- 10 satırda eksik değer vardı: 2 satırda tüm ölçümler, 8 satırda sadece cinsiyet boştu.
- 1 satırda cinsiyet `.` olarak girilmişti. `dropna` bunu yakalamadığı için ayrıca silindi.
- Toplam 11 satır (%3,2) silindi, kalan veri 333 satır. Silinen satırlar Adelie ve Gentoo'dan geliyordu, en küçük sınıf olan Chinstrap etkilenmedi.

## Ön işleme

`ColumnTransformer` ile:
- **Sayısal sütunlar → `StandardScaler`.** Ölçekler çok farklı (gaga derinliği ~15–20, kilo ~3000–6000), ölçeklenmezse model kiloya aşırı ağırlık verir.
- **`island`, `sex` → `OneHotEncoder(drop="if_binary")`.** Ada 3 sütuna, cinsiyet tek 0/1 sütununa dönüşür. Sinir ağı sayıları büyüklük olarak yorumladığı için label encoding yerine one-hot kullanıldı.
- **Hedef → `LabelEncoder`** (`CrossEntropyLoss` sınıf indeksi bekler).

Veri sınıf oranları korunarak %80/%20 ayrıldı (266 / 67). Ön işleyici yalnızca eğitim verisine `fit` edildi.

## Model

| | |
|---|---|
| Mimari | MLP: 8 → 32 → 32 → 3, ReLU |
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam, lr = 0.01 |
| Epoch | 100 |

`lr = 0.001` ile model 100 epoch sonunda hâlâ yakınsamamıştı. `lr = 0.01` ile ~20 epoch'ta %98'in üzerine çıktı.

## Sonuç

- **Test doğruluğu: %98,5** (67 örnekten 66'sı doğru).
- 40. epoch'tan sonra train loss düşmeye devam ederken test loss hafifçe yükseldi. Bu hafif bir overfitting işareti, daha az epoch yeterli olurdu.

## Kaydedilen dosyalar

`models/` klasöründe:

| Dosya | İçerik |
|---|---|
| `penguin_classification.pth` | Model ağırlıkları (`state_dict`) |
| `model_config.joblib` | Modelin yapısı (`in_features`, `hidden_units`, `out_features`) |
| `preprocessor.joblib` | Fit edilmiş `ColumnTransformer` |
| `label_encoder.joblib` | Sınıf adları |

Yeni bir penguen için tahmin:

```python
import joblib, torch, pandas as pd

pre = joblib.load("models/preprocessor.joblib")
le = joblib.load("models/label_encoder.joblib")
config = joblib.load("models/model_config.joblib")

model = PenguinClassifier(**config)   # sınıf tanımı notebook'ta
model.load_state_dict(torch.load("models/penguin_classification.pth"))
model.eval()

penguin = pd.DataFrame([{
    "island": "Biscoe", "culmen_length_mm": 48.0, "culmen_depth_mm": 14.8,
    "flipper_length_mm": 219, "body_mass_g": 5300, "sex": "MALE",
}])

with torch.inference_mode():
    x = torch.tensor(pre.transform(penguin), dtype=torch.float32)
    pred = model(x).argmax(dim=1)

print(le.classes_[pred.item()])   # Gentoo
```
