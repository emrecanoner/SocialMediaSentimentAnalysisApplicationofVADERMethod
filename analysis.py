import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

# Veriyi oku
df = pd.read_csv('dataset/sentimentdataset.csv')

# Veri temizleme işlemleri
# Tüm metin sütunlarındaki boşlukları temizle
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Boş değerleri kontrol et
print("Veri setindeki boş değerler:")
print(df.isnull().sum())

# Tekrarlanan verileri kontrol et
print("\nTekrarlanan veri sayısı:")
print(df.duplicated().sum())

# Tekrarlanan verileri temizle
df = df.drop_duplicates()

# Sentiment sütunundaki değerleri standartlaştır (büyük/küçük harf tutarlılığı)
df['Sentiment'] = df['Sentiment'].str.capitalize()

# Veri setindeki benzersiz duyguları kontrol et
print("\nVeri setindeki benzersiz duygular:")
print(df['Sentiment'].unique())

# Duyguların frekansını göster
print("\nDuyguların dağılımı:")
print(df['Sentiment'].value_counts())

# İlk Grafik: En çok görülen 10 duygu
plt.figure(figsize=(12, 6))
sentiment_counts = df['Sentiment'].value_counts().head(10)

sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Blues_r')
plt.title('En Çok Görülen 10 Duygu', fontsize=14, pad=20)
plt.xlabel('Duygular', fontsize=12)
plt.ylabel('Paylaşım Sayısı', fontsize=12)
plt.xticks(rotation=45)

# Sayıları çubukların üzerine ekle
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# İkinci Grafik: Temel duygular (Positive, Negative, Neutral)
plt.figure(figsize=(10, 6))

basic_sentiments = df[df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]['Sentiment'].value_counts()

sns.barplot(x=basic_sentiments.index, y=basic_sentiments.values, palette='Set2')
plt.title('Temel Duygu Dağılımı', fontsize=14, pad=20)
plt.xlabel('Duygu', fontsize=12)
plt.ylabel('Paylaşım Sayısı', fontsize=12)

# Sayıları çubukların üzerine ekle
for i, v in enumerate(basic_sentiments.values):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# En çok görülen 5 duyguyu seç
top_5_sentiments = df['Sentiment'].value_counts().head(5)

# Pasta grafiği
plt.figure(figsize=(10, 8))
plt.pie(top_5_sentiments.values, 
        labels=top_5_sentiments.index, 
        autopct='%1.1f%%',
        textprops={'fontsize': 12})
plt.title('En Çok Görülen 5 Duygu Dağılımı', fontsize=14, pad=20)
plt.axis('equal')
plt.show()

# Temel duyguların istatistiklerini yazdır
print("\nTemel duyguların dağılımı:")
print(basic_sentiments)

# Veri seti hakkında genel bilgiler
print("\nVeri seti hakkında genel bilgiler:")
print(df.info())