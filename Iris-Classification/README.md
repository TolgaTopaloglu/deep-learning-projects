# Iris Sınıflandırma

Iris çiçeklerini dört ölçüme göre (çanak yaprak uzunluğu/genişliği, taç yaprak uzunluğu/genişliği) üç türe ayırır: setosa, versicolor, virginica.

## Adımlar
1. pandas ve seaborn ile veri keşfi
2. Etiket kodlama (LabelEncoder), sınıf oranları korunarak %80/%20 eğitim/test ayrımı
3. PyTorch ile MLP: 4 → 10 → 10 → 3, ReLU aktivasyonu
4. `CrossEntropyLoss` ve Adam (lr=0.01, 100 epoch) ile eğitim
5. Değerlendirme: doğruluk, loss eğrileri, karışıklık matrisi (torchmetrics)

## Sonuç
Test doğruluğu yaklaşık %97–100. Test setinde yalnızca 30 örnek olduğundan her örnek doğruluğu yaklaşık %3,3 değiştirir.

## Veri
`iris.csv`: klasik Iris veri seti (150 örnek, her sınıftan 50).
