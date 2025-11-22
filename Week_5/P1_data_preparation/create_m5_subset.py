#!/usr/bin/env python3
"""
M5 Veri Seti Küçük Çalışma Seti Üretici

Bu script M5 veri setinden küçük bir alt-küme oluşturur:
- CA eyaleti, CA_1 mağazası, FOODS kategorisi
- En yüksek satışlı 5 ürün
- Günlük zaman serisi formatında
- Train/Validation split (son 28 gün validation)

Kullanım: python create_m5_subset.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_m5_subset():
    """M5 veri setinden küçük çalışma seti oluştur"""
    
    print("🎯 M5 Küçük Çalışma Seti Oluşturucu")
    print("=" * 50)
    
    # Çıktı klasörlerini oluştur
    os.makedirs('./artifacts/datasets', exist_ok=True)
    os.makedirs('./artifacts/figures', exist_ok=True)
    
    # 1. Veri dosyalarını oku
    print("\n📁 1. Veri dosyaları okunuyor...")
    
    try:
        # Sales verisi
        print("   • sales_train_validation.csv okunuyor...")
        sales_df = pd.read_csv('/Users/eyyupcap/Desktop/Ayşe/VS Code/Kairu_DS360/Week_5/data/sales_train_validation.csv')
        print(f"   ✓ Satış verisi: {sales_df.shape}")
        
        # Calendar verisi
        print("   • calendar.csv okunuyor...")
        calendar_df = pd.read_csv('/Users/eyyupcap/Desktop/Ayşe/VS Code/Kairu_DS360/Week_5/data/calendar.csv')
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        print(f"   ✓ Takvim verisi: {calendar_df.shape}")
        
        # Prices verisi (opsiyonel, kullanmayacağız ama kontrol edelim)
        try:
            prices_df = pd.read_csv('/Users/eyyupcap/Desktop/Ayşe/VS Code/Kairu_DS360/Week_5/data/sell_prices.csv')
            print(f"   ✓ Fiyat verisi: {prices_df.shape}")
        except FileNotFoundError:
            print("   ⚠️  Fiyat verisi bulunamadı (isteğe bağlı)")
            
    except FileNotFoundError as e:
        print(f"   ❌ Veri dosyası bulunamadı: {e}")
        print("   💡 Önce create_sample_data.py çalıştırın veya gerçek M5 verisini indirin")
        return None, None, None
    
    # 2. CA_1 mağazası ve FOODS kategorisini filtrele
    print("\n🏪 2. CA_1 mağazası ve FOODS kategorisi filtreleniyor...")
    
    # CA_1 mağazası filtresi
    ca1_mask = (sales_df['store_id'] == 'CA_1')
    ca1_sales = sales_df[ca1_mask].copy()
    print(f"   • CA_1 mağazası ürün sayısı: {len(ca1_sales)}")
    
    # FOODS kategorisi filtresi
    # M5'te kategori 'cat_id' sütununda, FOODS genelde FOODS ile başlar
    foods_mask = ca1_sales['cat_id'].str.contains('FOOD', case=False, na=False)
    foods_sales = ca1_sales[foods_mask].copy()
    print(f"   • FOODS kategorisi ürün sayısı: {len(foods_sales)}")
    
    if len(foods_sales) == 0:
        print("   ⚠️  FOODS kategorisi bulunamadı, tüm kategorileri kullanıyoruz...")
        foods_sales = ca1_sales.copy()
    
        # 3. Veriyi günlük formata dönüştür ve en yüksek satışlı 5 ürünü bul
    print("\n📊 3. Veriler günlük formata dönüştürülüyor...")
    
    # Satış verilerini uzun formata dönüştür
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_long = pd.melt(foods_sales, 
                        id_vars=id_vars,
                        var_name='d', 
                        value_name='sales')
    
    # d_1, d_2 gibi sütunları takvim ile eşleştir
    sales_long = sales_long.merge(calendar_df[['d', 'date']], on='d', how='left')
    
    # En yüksek satışlı 5 ürünü bul
    total_sales_by_item = sales_long.groupby('item_id')['sales'].sum().sort_values(ascending=False)
    top_5_items = total_sales_by_item.head(5).index.tolist()
    print(f"   • En çok satılan 5 ürün: {', '.join(top_5_items)}")
    
    # Sadece en yüksek satışlı 5 ürünü filtrele
    top_sales = sales_long[sales_long['item_id'].isin(top_5_items)].copy()
    
    # 4. Veriyi temizle ve hazırla
    print("\n🧹 4. Veriler düzenleniyor...")
    
    # Günlük zaman serisi formatına dönüştür
    complete_df = top_sales[['date', 'item_id', 'sales']].copy()
    
    # Tüm olası tarih-ürün kombinasyonlarını oluştur
    all_dates = calendar_df['date'].unique()
    all_dates_sorted = sorted(all_dates)
    
    # Train/validation split için tarihler
    split_date = all_dates_sorted[-28]  # Son 28 gün validation
    train_end_date = all_dates_sorted[-29]  # Train sonu
    
    print(f"   • Toplam {len(all_dates_sorted)} gün, split tarihi: {split_date.strftime('%Y-%m-%d')}")
    
    # 5. Train ve validation setlerini oluştur
    print("\n🔄 5. Train ve validation setleri oluşturuluyor...")
    
    # Train ve validation setleri
    train_df = complete_df[complete_df['date'] <= train_end_date].copy()
    valid_df = complete_df[complete_df['date'] >= split_date].copy()
    
    print(f"   • Train: {train_df['date'].min()} - {train_df['date'].max()} ({len(train_df)} satır)")
    print(f"   • Valid: {valid_df['date'].min()} - {valid_df['date'].max()} ({len(valid_df)} satır)")
    
    # Index'i tarih yap
    train_df = train_df.set_index('date')
    valid_df = valid_df.set_index('date')
    
    # 7. Çıktıları kaydet
    print("\n💾 6. Sonuçlar kaydediliyor...")
    
    # CSV dosyaları
    train_path = './artifacts/datasets/train.csv'
    valid_path = './artifacts/datasets/valid.csv'
    
    train_df.to_csv(train_path)
    valid_df.to_csv(valid_path)
    
    print(f"   ✓ Train verisi: {train_path}")
    print(f"   ✓ Valid verisi: {valid_path}")
    
    # 8. Görselleştirme
    print("\n📊 7. Günlük toplam satış grafiği oluşturuluyor...")
    
    # Günlük toplam satış hesapla
    daily_total = complete_df.groupby('date')['sales'].sum().reset_index()
    
    # Grafik oluştur
    plt.figure(figsize=(15, 8))
    
    # Train ve validation bölgelerini ayır
    train_dates = train_df.reset_index()['date'].unique()
    valid_dates = valid_df.reset_index()['date'].unique()
    
    train_total = daily_total[daily_total['date'].isin(train_dates)]
    valid_total = daily_total[daily_total['date'].isin(valid_dates)]
    
    # Train verisi
    plt.plot(train_total['date'], train_total['sales'], 
             label='Train', color='blue', linewidth=2)
    
    # Validation verisi
    plt.plot(valid_total['date'], valid_total['sales'], 
             label='Validation', color='red', linewidth=2)
    
    # Split çizgisi
    plt.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7, 
                label=f'Train/Valid Split ({split_date.strftime("%Y-%m-%d")})')
    
    # Grafik düzenlemeleri
    plt.title('M5 Seçilen 5 Ürün - Günlük Toplam Satış\n' + 
              f'CA_1 Mağazası, FOODS Kategorisi', fontsize=16, fontweight='bold')
    plt.xlabel('Tarih', fontsize=12)
    plt.ylabel('Günlük Toplam Satış', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # X ekseni etiketlerini döndür
    plt.xticks(rotation=45)
    
    # Layout ayarla
    plt.tight_layout()
    
    # Kaydet
    figure_path = './artifacts/figures/overall_daily_sales.png'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Grafik: {figure_path}")
    
    plt.close()
    
    # 9. Özet bilgiler
    print("\n📋 ÖZET BİLGİLER")
    print("=" * 50)
    print(f"• Seçilen ürünler: {', '.join(complete_df['item_id'].unique())}")
    print(f"• Toplam gün sayısı: {len(all_dates_sorted)}")
    print(f"• Train gün sayısı: {len(train_df.reset_index()['date'].unique())}")
    print(f"• Validation gün sayısı: {len(valid_df.reset_index()['date'].unique())}")
    print(f"• Ortalama günlük satış: {daily_total['sales'].mean():.1f}")
    print(f"• Maksimum günlük satış: {daily_total['sales'].max()}")
    print(f"• Minimum günlük satış: {daily_total['sales'].min()}")
    
    # Ürün bazında istatistikler
    print(f"\n📊 ÜRÜN BAZINDA İSTATİSTİKLER:")
    item_stats = complete_df.groupby('item_id')['sales'].agg(['sum', 'mean', 'std', 'max']).round(2)
    for item_id, stats in item_stats.iterrows():
        print(f"• {item_id}: Toplam={stats['sum']:,.0f}, Ort={stats['mean']:.1f}, "
              f"Std={stats['std']:.1f}, Max={stats['max']:.0f}")
    
    print(f"\n✅ İşlem tamamlandı!")
    print(f"📁 Çıktılar: ./artifacts/ klasöründe")
    
    return train_df, valid_df, daily_total

def main():
    """run_modular.py için wrapper fonksiyonu"""
    result = create_m5_subset()
    if result is None or (isinstance(result, tuple) and result[0] is None):
        print(f"❌ Veri dosyası bulunamadı. Sample data kullanın.")
        return False
    else:
        print(f"✅ M5 CA_1 FOODS subset created successfully!")
        return True

if __name__ == "__main__":
    try:
        result = create_m5_subset()
        if result is None or (isinstance(result, tuple) and result[0] is None):
            print(f"\n❌ Veri dosyası bulunamadı. Script durduruluyor.")
        else:
            train_data, valid_data, daily_sales = result
            print(f"\n🎉 M5 küçük çalışma seti başarıyla oluşturuldu!")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()