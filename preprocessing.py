import pandas as pd
import re
import emoji
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK gerekli dosyaları indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Veriyi oku
df = pd.read_csv('dataset/sentimentdataset.csv')

def preprocess_text(text):
    """Metin ön işleme fonksiyonu"""
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Emojileri metne çevir
    text = emoji.demojize(text)
    
    # Özel karakterleri ve fazla boşlukları temizle
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Sayıları kaldır
    text = re.sub(r'\d+', '', text)
    
    return text.strip()

def extract_hashtags(text):
    """Metinden hashtag'leri çıkarma"""
    if not isinstance(text, str):
        return ""
    hashtags = re.findall(r'#(\w+)', text)
    return ",".join(hashtags) if hashtags else ""  # Listeyi string'e çevir

# Veri temizleme işlemleri
print("Veri ön işleme başlıyor...")

# Text sütununu temizle
df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Hashtag'leri ayrı bir sütun olarak sakla (string formatında)
df['extracted_hashtags'] = df['Text'].apply(extract_hashtags)

# Sentiment sütununu standartlaştır
df['Sentiment'] = df['Sentiment'].str.strip()
df['Sentiment'] = df['Sentiment'].str.lower()

# Boş değerleri kontrol et ve temizle
print("\nBoş değerler:")
print(df.isnull().sum())

# Tekrarlanan verileri kontrol et ve temizle
duplicate_count = df.duplicated().sum()
print(f"\nTekrarlanan veri sayısı: {duplicate_count}")
df = df.drop_duplicates()

# Duygu sınıflarını kontrol et
print("\nBenzersiz duygu sınıfları:")
print(df['Sentiment'].unique())
print("\nDuygu sınıflarının dağılımı:")
print(df['Sentiment'].value_counts())

# Temizlenmiş veriyi kaydet
df.to_csv('dataset/cleaned_sentiment_dataset.csv', index=False)
print("\nTemizlenmiş veri kaydedildi: cleaned_sentiment_dataset.csv")

# Örnek temizlenmiş metinleri göster
print("\nÖrnek temizlenmiş metinler:")
print(df[['Text', 'cleaned_text', 'extracted_hashtags']].head()) 