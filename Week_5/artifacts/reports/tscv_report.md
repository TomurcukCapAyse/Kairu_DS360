# Time Series Cross-Validation Raporu

**Tarih:** 2025-11-23 17:26:14
**Method:** Rolling-Origin Cross-Validation
**Model:** LightGBM Regressor

## 🎯 Amaç

Zaman serisi için uygun çapraz doğrulama ile model performansını robust şekilde ölçmek.

## ⚠️ Neden Shuffle CV Olmaz?

1. **Temporal Leakage**: Gelecek verisi ile geçmiş tahmin edilir (data leakage)
2. **Pattern Bozukluk**: Zaman bağımlı pattern'ler parçalanır
3. **Gerçekçi Olmama**: Production'da shuffle yok, sadece geçmiş var

## 📊 Cross-Validation Yapısı

- **Validation Horizon:** 28 gün
- **Toplam Fold:** 3 (başarılı)
- **Yaklaşım:** Rolling-Origin (Expanding window train)

### Fold Detayları

| Fold | Train Başlangıç | Train Bitiş | Valid Başlangıç | Valid Bitiş | Train Gün | Valid Gün |
|------|-----------------|-------------|-----------------|-------------|-----------|-----------|
| 0 | 2011-01-29 | 2016-03-27 | 2016-03-28 | 2016-04-24 | 1885 | 28 |
| 1 | 2011-01-29 | 2016-02-28 | 2016-02-29 | 2016-03-27 | 1857 | 28 |
| 2 | 2011-01-29 | 2016-01-31 | 2016-02-01 | 2016-02-28 | 1829 | 28 |

## 📈 Performans Sonuçları

### Özet Metrikler

| Metrik | Ortalama | Std Sapma | Min | Max |
|--------|----------|-----------|-----|-----|
| MAE | 8.94 | 0.73 | 7.91 | 9.46 |
| RMSE | 13.42 | 1.07 | 12.55 | 14.93 |
| MAPE | 29.81 | 4.19 | 24.53 | 34.79 |
| sMAPE | 34.10 | 5.32 | 27.86 | 40.87 |

### Fold Bazında Detaylar

| Fold | MAE | RMSE | sMAPE (%) | Model Iterasyon |
|------|-----|------|-----------|-----------------|
| 0 | 9.46 | 12.77 | 33.55 | 108 |
| 1 | 7.91 | 12.55 | 40.87 | 128 |
| 2 | 9.44 | 14.93 | 27.86 | 52 |

## 🔍 Analiz ve Yorumlar

### Model Tutarlılığı
- **Orta tutarlılık**: sMAPE standart sapması orta (5.32%)

- **En iyi fold**: Fold 2 (sMAPE: 27.86%)
- **En kötü fold**: Fold 1 (sMAPE: 40.87%)

### Production Önerileri

1. **Model Güvenilirliği**: CV sonuçları model performansının robust bir ölçümünü sağlar
2. **Temporal Validation**: Rolling-origin yaklaşımı production senaryosunu yansıtır
3. **Performance Beklentisi**: Ortalama sMAPE 34.10% ±5.32%

### Sınırlamalar

- Basit iteratif forecasting kullanıldı (production için iyileştirilebilir)
- Sadece LightGBM test edildi (ensemble modeller denenebilir)
- Sabit validation horizon (adaptive horizon test edilebilir)

---
*Bu rapor otomatik olarak oluşturulmuştur - 2025-11-23 17:26:14*
