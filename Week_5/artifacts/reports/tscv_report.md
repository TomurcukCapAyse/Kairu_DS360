# Time Series Cross-Validation Raporu

**Tarih:** 2025-11-24 02:23:39
**Method:** Rolling-Origin Cross-Validation

## Neden Shuffle CV Olmaz?

1. **Temporal Leakage**: Gelecek verisi ile geçmiþ tahmin edilir
2. **Pattern Bozukluk**: Zaman baðýmlý pattern'ler parçalanýr
3. **Gerçekçi Olmama**: Production'da shuffle yok, sadece geçmiþ var

## Sonuçlar

**Fold Sayýsý:** 3
**Validation Horizon:** 28 gün

### Ortalama Metrikler

| Metrik | Ortalama | Std Sapma |
|--------|----------|-----------|
| MAE | 7.11 | 0.50 |
| RMSE | 12.70 | 0.77 |
| sMAPE | 21.02 | 5.61 |

### Fold Detaylarý

| Fold | MAE | RMSE | sMAPE (%) |
|------|-----|------|-----------|
| 0 | 7.12 | 13.75 | 17.43 |
| 1 | 6.48 | 11.95 | 16.69 |
| 2 | 7.72 | 12.40 | 28.95 |