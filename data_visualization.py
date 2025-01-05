import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter
import plotly.graph_objects as go
from datetime import datetime

# Temel görsel ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Temizlenmiş veriyi oku
df = pd.read_csv('dataset/cleaned_sentiment_dataset.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Ülke ve Platform isimlerini standartlaştır
df['Country'] = df['Country'].str.strip()
df['Platform'] = df['Platform'].str.strip()

# USA değerlerini birleştir
df['Country'] = df['Country'].replace({'USA': 'United States', 'USA ': 'United States'})

def create_sentiment_distribution():
    """Duygu dağılımı grafiği - Sadece en sık görülen 8 duygu"""
    sentiment_counts = df['Sentiment'].value_counts().head(8)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('En Sık Görülen 8 Duygu Sınıfının Dağılımı')
    plt.xticks(rotation=45)
    plt.ylabel('Tweet Sayısı')
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_platform_analysis():
    """Platform bazlı duygu analizi - En sık görülen 5 duygu"""
    # Platform verilerini grupla
    platform_counts = df['Platform'].value_counts()
    print("Platform dağılımı:")
    print(platform_counts)
    
    top_sentiments = df['Sentiment'].value_counts().head(5).index
    platform_data = df[df['Sentiment'].isin(top_sentiments)]
    
    plt.figure(figsize=(12, 6))
    platform_sentiment = pd.crosstab(platform_data['Platform'], platform_data['Sentiment'])
    platform_sentiment.plot(kind='bar', stacked=True)
    plt.title('Platform Bazlı Duygu Dağılımı (En Sık 5 Duygu)')
    plt.xlabel('Platform')
    plt.ylabel('Tweet Sayısı')
    plt.legend(title='Duygular', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('visualizations/platform_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_time_series_analysis():
    """Zaman serisi analizi - Aylık bazda en sık 3 duygu"""
    df['Month_Year'] = df['Timestamp'].dt.to_period('M')
    top_sentiments = df['Sentiment'].value_counts().head(3).index
    monthly_sentiment = df[df['Sentiment'].isin(top_sentiments)].groupby(['Month_Year', 'Sentiment']).size().unstack()
    
    plt.figure(figsize=(15, 6))
    monthly_sentiment.plot(kind='line', marker='o', markersize=4)
    plt.title('Aylık Duygu Değişimi (En Sık 3 Duygu)')
    plt.xlabel('Tarih')
    plt.ylabel('Tweet Sayısı')
    plt.legend(title='Duygular', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_wordcloud():
    """Duygu bazlı kelime bulutları - Sadece en sık 4 duygu"""
    top_sentiments = df['Sentiment'].value_counts().head(4).index
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, sentiment in enumerate(top_sentiments):
        text = ' '.join(df[df['Sentiment'] == sentiment]['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white',
                            max_words=50).generate(text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(f'{sentiment.capitalize()}')
    
    plt.tight_layout()
    plt.savefig('visualizations/wordcloud_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hourly_distribution():
    """Saatlik tweet dağılımı - En sık 3 duygu"""
    df['Hour'] = df['Timestamp'].dt.hour
    top_sentiments = df['Sentiment'].value_counts().head(3).index
    hourly_data = df[df['Sentiment'].isin(top_sentiments)]
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=hourly_data, x='Hour', hue='Sentiment', multiple="stack", bins=24)
    plt.title('Saatlik Tweet Dağılımı (En Sık 3 Duygu)')
    plt.xlabel('Saat')
    plt.ylabel('Tweet Sayısı')
    plt.legend(title='Duygular', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('visualizations/hourly_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_country_analysis():
    """Ülke bazlı duygu analizi - En sık 5 ülke ve 3 duygu"""
    # Ülke verilerini grupla
    country_counts = df['Country'].value_counts()
    print("\nÜlke dağılımı:")
    print(country_counts)
    
    top_countries = df['Country'].value_counts().head(5).index
    top_sentiments = df['Sentiment'].value_counts().head(3).index
    country_data = df[df['Country'].isin(top_countries) & df['Sentiment'].isin(top_sentiments)]
    
    plt.figure(figsize=(12, 6))
    country_sentiment = pd.crosstab(country_data['Country'], country_data['Sentiment'])
    country_sentiment.plot(kind='bar', stacked=True)
    plt.title('Ülke Bazlı Duygu Dağılımı (En Sık 5 Ülke, 3 Duygu)')
    plt.xlabel('Ülke')
    plt.ylabel('Tweet Sayısı')
    plt.legend(title='Duygular', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/country_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Visualizations klasörünü oluştur
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    print("Veri özeti:")
    print("\nToplam kayıt sayısı:", len(df))
    print("\nBenzersiz değerler:")
    print("Platform sayısı:", df['Platform'].nunique())
    print("Ülke sayısı:", df['Country'].nunique())
    print("Duygu sayısı:", df['Sentiment'].nunique())
    
    print("\nGörselleştirmeler oluşturuluyor...")
    
    # Tüm görselleştirmeleri oluştur
    create_sentiment_distribution()
    create_platform_analysis()
    create_time_series_analysis()
    create_wordcloud()
    create_hourly_distribution()
    create_country_analysis()
    
    print("\nGörselleştirmeler tamamlandı! 'visualizations' klasörünü kontrol ediniz.")

if __name__ == "__main__":
    main()