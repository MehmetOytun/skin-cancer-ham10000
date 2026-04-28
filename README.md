# HAM10000 Cilt Lezyonu Sınıflandırması

EfficientNet-B0, ROI segmentasyonu ve Optuna ile hiperparametre optimizasyonu kullanan 7 sınıflı dermatoskopi görüntü sınıflandırma projesi.

## Genel Bakış

Bu proje, HAM10000 veri setindeki cilt lezyonlarını 7 tanı kategorisinde sınıflandırır. Klasik bilgisayarlı görü tekniklerini (ROI segmentasyonu) modern transfer öğrenme (EfficientNet-B0) ve otomatik hiperparametre optimizasyonu (Optuna) ile birleştiren bir derin öğrenme pipeline'ı kullanır.

## Pipeline

1. **Veri hazırlama** — HAM10000 metadata yükleme + lezyon bazlı stratified split (aynı lezyonun farklı görüntülerinin train/test arasına dağılmasını önler, data leakage'ı engeller)
2. **ROI segmentasyonu** — Otsu eşikleme + morfolojik işlemler + kontur tespiti → %15 padding ile bounding box crop
3. **ROI cache** — Önceden hesaplanmış ROI'ler `.npy` formatında MD5 hash ile saklanır; Optuna trial'larında tekrar tekrar hesaplanmaz
4. **Model** — EfficientNet-B0 (ImageNet pretrained) + özelleştirilmiş classifier head
5. **Hiperparametre optimizasyonu** — Optuna + TPE sampler + Median pruner (15 trial × 6 epoch, kısıtlı search space)
6. **İki aşamalı eğitim:**
   - Aşama 1: Sadece classifier head (backbone donuk) — 50 epoch
   - Aşama 2: Son 3 feature bloğunun fine-tuning'i, discriminative learning rate ile — 30 epoch
7. **Değerlendirme** — Test accuracy, precision, recall, F1, confusion matrix, classification report

## Veri Seti

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 7 sınıfta 10.015 dermatoskopi görüntüsü:

| Kod | Sınıf | Açıklama |
|-----|-------|----------|
| `akiec` | Aktinik Keratoz | Pre-kanseröz lezyon |
| `bcc` | Bazal Hücreli Karsinom | Yaygın cilt kanseri |
| `bkl` | Benign Keratoz | Selim lezyon |
| `df` | Dermatofibroma | Selim cilt nodülü |
| `mel` | Melanom | Malign melanom |
| `nv` | Melanositik Nevus | Yaygın benler (veri setinin ~%67'si) |
| `vasc` | Vasküler Lezyon | Damar kaynaklı lezyonlar |

## Teknoloji Yığını

- **PyTorch** + **torchvision** — Model ve eğitim
- **OpenCV** — ROI segmentasyonu (Otsu + morfoloji)
- **Optuna** — Hiperparametre optimizasyonu
- **scikit-learn** — Stratified split, metrikler
- **pandas / numpy** — Veri işleme

## Sınıf Dengesizliği Yönetimi

HAM10000 oldukça dengesiz bir veri seti (`nv` ~%67, `df` ~%1). İki strateji uygulandı:
- Dengeli batch'ler için `WeightedRandomSampler`
- Label smoothing ile `CrossEntropyLoss(weight=class_weights)`

## Optuna Search Space (Kısıtlı)

| Hiperparametre | Aralık |
|----------------|--------|
| Learning rate | 1e-4 → 1e-2 (log) |
| Dropout | 0.2 → 0.5 |
| Weight decay | 1e-5 → 1e-3 (log) |
| Batch size | {16, 32, 64} |

`TPESampler` ile 15 trial, `MedianPruner` kötü trial'ları erken keser.

## Nasıl Çalıştırılır?

Notebook **Kaggle** ortamı için tasarlandı (`/kaggle/input/` ve `/kaggle/working/` path'lerini kullanır).

1. Kaggle'da `notebook448638b0d6.ipynb` dosyasını aç
2. [HAM10000 veri setini](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) input olarak ekle
3. GPU'yu aktif et (T4 veya P100)
4. Tüm hücreleri çalıştır

T4 GPU'da yaklaşık çalışma süreleri:
- ROI cache oluşturma: ~5–10 dk (tek sefer)
- Optuna study (15 trial): ~45–60 dk
- Final eğitim (Aşama 1 + Aşama 2): ~60–90 dk

## Çıktılar

Çalıştırma sonrası `/kaggle/working/` klasörüne kaydedilen dosyalar:
- `best_efficientnet_b0.pth` — En iyi model ağırlıkları
- `probabilities_efficientnet_b0.npy` — Test seti olasılıkları
- `test_labels_efficientnet_b0.npy` — Test seti gerçek etiketler
- `confusion_matrix_efficientnet_b0.jpg` — Confusion matrix
- `optuna_study.pkl` — Optuna study objesi
- `roi_examples.png` — ROI segmentasyon örnekleri
- `optuna_results.png` — Trial geçmişi + parametre önemleri

## Önemli Tasarım Tercihleri

- **Lezyon bazlı split** (image bazlı değil) — Aynı lezyonun birden fazla görüntüsünün hem train hem test'te yer almasını engeller; bu olmasaydı data leakage yüzünden test skorları yanıltıcı şekilde yüksek çıkardı
- **ROI cache** — Segmentasyon 10K görüntü için bir kez yapılır, sonra trial'lar diskten okur (HPO için büyük hızlanma)
- **İki aşamalı eğitim** — Önce donuk backbone üzerinde classifier'ın yakınsamasına izin verilir, sonra katmanlar açılır; catastrophic forgetting önlenir
- **Discriminative learning rate** — Fine-tuning sırasında backbone `lr/5` ile, classifier tam `lr` ile eğitilir
