# Proje için kullanılan dosya
# feature_based_analysis.py dosyası, duygu analizi ve metin sınıflandırma için gelişmiş özellik çıkarımı ve analizini içerir

# Gerekli kütüphaneleri import et
import pandas as pd
import numpy as np
import emoji
import re
from datetime import datetime
from collections import Counter
import os
import joblib
from scipy import stats
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
import random

# Sklearn kütüphaneleri
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
import seaborn as sns

# SMOTE için
from imblearn.over_sampling import SMOTE

# TextBlob ve VADER için
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Güvenilirlik analizi için
from sklearn.preprocessing import StandardScaler

random.seed(42)  # Tekrarlanabilirlik için

class SentimentFeatureAnalyzer:
    def __init__(self, data_path='dataset/cleaned_sentiment_dataset.csv'):
        """Sınıf başlatıcı"""
        self.df = pd.read_csv(data_path)
        print(f"Orijinal veri seti boyutu: {self.df.shape}")
        
        # Duyguları basitleştir
        self.df['sentiment'] = self.df['Sentiment'].map(
            lambda x: 'positive' if x in ['happy', 'joy', 'excitement', 'love', 'amazing']
            else 'negative' if x in ['sad', 'anger', 'fear', 'disgust', 'disappointed']
            else 'neutral'
        )
        
        # Sınıf dağılımını göster
        print("\nOrijinal sınıf dağılımı:")
        print(self.df['sentiment'].value_counts())
        
        # Veriyi dengele
        self.balance_dataset()
        
        # Duygu sözlüğü - genişletilmiş
        self.sentiment_words = {
            'positive': [
                'happy', 'great', 'awesome', 'excellent', 'good', 'wonderful', 
                'amazing', 'fantastic', 'joy', 'love', 'excited', 'perfect',
                'beautiful', 'brilliant', 'delighted', 'blessed', 'glad'
            ],
            'negative': [
                'sad', 'bad', 'terrible', 'awful', 'horrible', 'disappointed',
                'angry', 'upset', 'hate', 'miserable', 'hurt', 'depressed',
                'worried', 'frustrated', 'unhappy', 'painful', 'worst'
            ]
        }
        
        # Emoji sözlüğü - genişletilmiş
        self.emoji_patterns = {
            'positive': r'[😊🙂😄😃😀🥰😍🤗😎👍❤️💕]',
            'negative': r'[😢😭😞😔😟😩😫😖😣😕👎💔]'
        }
        
        print(f"Veri seti yüklendi. Boyut: {self.df.shape}")
        
        # Figures klasörünü oluştur (eğer yoksa)
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
    def balance_dataset(self):
        """Gelişmiş veri seti dengeleme ve büyütme"""
        print("\nVeri seti dengeleme ve büyütme işlemi başlıyor...")
        
        # Her sınıftan tam 100 örnek hedefi
        target_samples = 100
        balanced_dfs = []
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_df = self.df[self.df['sentiment'] == sentiment].copy()
            current_samples = len(sentiment_df)
            
            if current_samples < target_samples:
                # Daha agresif veri artırma (özellikle negative sınıfı için)
                augmented_texts = []
                augmented_sentiments = []
                
                for _, row in sentiment_df.iterrows():
                    text = row['Text']
                    
                    # 1. Orijinal metin
                    augmented_texts.append(text)
                    augmented_sentiments.append(sentiment)
                    
                    # 2. Emoji varyasyonu
                    emoji_text = self.add_emoji_variations(text, sentiment)
                    augmented_texts.append(emoji_text)
                    augmented_sentiments.append(sentiment)
                    
                    # 3. Noktalama varyasyonu
                    punct_text = self.add_punctuation_variations(text, sentiment)
                    augmented_texts.append(punct_text)
                    augmented_sentiments.append(sentiment)
                    
                    # 4. Kelime sırası değişimi
                    reordered_text = self.reorder_words(text)
                    augmented_texts.append(reordered_text)
                    augmented_sentiments.append(sentiment)
                    
                    # 5. Emoji + Noktalama kombinasyonu
                    emoji_punct_text = self.add_emoji_variations(
                        self.add_punctuation_variations(text, sentiment),
                        sentiment
                    )
                    augmented_texts.append(emoji_punct_text)
                    augmented_sentiments.append(sentiment)
                    
                    if len(augmented_texts) >= target_samples:
                        break
                
                # Tam olarak target_samples kadar örnek al
                augmented_df = pd.DataFrame({
                    'Text': augmented_texts[:target_samples],
                    'sentiment': augmented_sentiments[:target_samples]
                })
                balanced_dfs.append(augmented_df)
            else:
                # Rastgele örnekleme
                balanced_df = sentiment_df.sample(n=target_samples, random_state=42)
                balanced_dfs.append(balanced_df)
        
        # Dengeli veri setini oluştur
        self.df = pd.concat(balanced_dfs, ignore_index=True)
        
        print("\nDengelenmiş veri seti dağılımı:")
        print(self.df['sentiment'].value_counts())
        
        return self.df

    def add_emoji_variations(self, text, sentiment):
        """Duygu durumuna göre gelişmiş emoji varyasyonları ekle"""
        emoji_map = {
            'positive': ['😊', '👍', '❤️', '🎉', '😄', '✨', '💪', '🌟'],
            'negative': ['😢', '👎', '😠', '😔', '💔', '😞', '😣', '😫'],
            'neutral': ['🤔', '😐', '💭', '📝', '💡', '🔍', '📌', '💬']
        }
        
        # Mevcut metne 1-3 emoji ekle
        emojis = random.sample(emoji_map[sentiment], random.randint(1, 3))
        
        # Emojileri metnin başına veya sonuna rastgele ekle
        if random.choice([True, False]):
            return ' '.join(emojis) + ' ' + text
        return text + ' ' + ' '.join(emojis)

    def add_punctuation_variations(self, text, sentiment):
        """Duygu durumuna göre gelişmiş noktalama varyasyonları ekle"""
        punct_map = {
            'positive': ['!', '!!', '...!', '! :)', '!!! 🎉', '~'],
            'negative': ['...', '?!', '!!?', '... :(', '?!?', '...?'],
            'neutral': ['.', '...', '?', '...?', '. -', '...']
        }
        
        # Rastgele 1-2 noktalama işareti ekle
        puncts = random.sample(punct_map[sentiment], random.randint(1, 2))
        return text + ''.join(puncts)

    def reorder_words(self, text):
        """Kelime sırasını değiştir (anlamı koruyarak)"""
        words = text.split()
        if len(words) <= 3:  # Çok kısa metinlerde değişiklik yapma
            return text
        
        # Cümlenin ortasındaki kelimelerin sırasını değiştir
        mid_start = len(words) // 3
        mid_end = len(words) - len(words) // 3
        middle_words = words[mid_start:mid_end]
        random.shuffle(middle_words)
        
        return ' '.join(words[:mid_start] + middle_words + words[mid_end:])
    
    def extract_features(self, text):
        """Geliştirilmiş özellik çıkarımı"""
        features = {}
        text = str(text).lower()
        
        # Temel metin özellikleri
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()])
        
        # Duygu kelimeleri sayısı - ağırlıklı
        features['positive_words'] = sum(3 for word in text.split() 
                                     if word in self.sentiment_words['positive'])
        features['negative_words'] = sum(3 for word in text.split() 
                                     if word in self.sentiment_words['negative'])
        
        # Emoji sayısı - ağırlıklı
        features['positive_emoji'] = len(re.findall(self.emoji_patterns['positive'], text)) * 2
        features['negative_emoji'] = len(re.findall(self.emoji_patterns['negative'], text)) * 2
        
        # Noktalama işaretleri - ağırlıklı
        features['exclamation_count'] = text.count('!') * 1.5
        features['question_count'] = text.count('?')
        
        # Büyük harf kullanımı
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        return features
    
    def train_model(self):
        """Model eğitimi"""
        print("\nModel eğitimi başlıyor...")
        
        # Özellik çıkarımı
        features_df = pd.DataFrame([
            self.extract_features(text) for text in self.df['Text']
        ])
        
        # TF-IDF özellikleri
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english'
        )
        tfidf_features = self.tfidf.fit_transform(self.df['Text'])
        
        # Özellikleri birleştir
        X = np.hstack([
            features_df.values,
            tfidf_features.toarray()
        ])
        
        # Etiketleri hazırla - sadece 3 sınıf
        self.df['simple_sentiment'] = self.df['Sentiment'].map(
            lambda x: 'positive' if x in self.sentiment_words['positive']
            else 'negative' if x in self.sentiment_words['negative']
            else 'neutral'
        )
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.df['simple_sentiment'])
        
        # Veriyi böl - stratify olmadan
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Eğitim
        print("Model eğitiliyor...")
        self.model.fit(X_train, y_train)
        
        # Değerlendirme
        y_pred = self.model.predict(X_test)
        print(f"\nModel Doğruluğu: {accuracy_score(y_test, y_pred):.2%}")
        print("\nSınıf Bazında Performans:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predict_emotion(self, text):
        """Duygu tahmini"""
        try:
            # Özellik çıkarımı
            features = pd.DataFrame([self.extract_features(text)])
            tfidf_features = self.tfidf.transform([text])
            
            # Özellikleri birleştir
            X = np.hstack([
                features.values,
                tfidf_features.toarray()
            ])
            
            # Tahmin
            probs = self.model.predict_proba(X)[0]
            classes = self.label_encoder.classes_
            
            # Sonuçları hazırla
            results = dict(zip(classes, [round(p * 100, 2) for p in probs]))
            prediction = max(results.items(), key=lambda x: x[1])[0]
            
            return {
                'prediction': prediction,
                'confidence': results[prediction],
                'probabilities': results
            }
            
        except Exception as e:
            print(f"Hata: {str(e)}")
            return None
    
    def analyze_sentiment(self):
        """VADER ile duygu analizi"""
        analyzer = SentimentIntensityAnalyzer()
        
        # VADER skorlarını hesapla
        self.df['vader_scores'] = self.df['Text'].apply(
            lambda x: analyzer.polarity_scores(str(x))
        )
        
        # Skorları ayrı kolonlara ayır
        self.df['vader_neg'] = self.df['vader_scores'].apply(lambda x: x['neg'])
        self.df['vader_neu'] = self.df['vader_scores'].apply(lambda x: x['neu'])
        self.df['vader_pos'] = self.df['vader_scores'].apply(lambda x: x['pos'])
        self.df['vader_compound'] = self.df['vader_scores'].apply(lambda x: x['compound'])

    def perform_hypothesis_test(self):
        """Hipotez testleri"""
        print("\nHipotez Testleri:")
        print("=" * 50)
        
        # H0: VADER skorları ile etiketler arasında ilişki yoktur
        # H1: VADER skorları ile etiketler arasında anlamlı bir ilişki vardır
        
        # Her duygu için VADER compound skorlarını grupla
        pos_scores = self.df[self.df['sentiment'] == 'positive']['vader_compound']
        neg_scores = self.df[self.df['sentiment'] == 'negative']['vader_compound']
        neu_scores = self.df[self.df['sentiment'] == 'neutral']['vader_compound']
        
        # Kruskal-Wallis H-test
        h_stat, p_value = stats.kruskal(pos_scores, neg_scores, neu_scores)
        
        print("\n1. Kruskal-Wallis H-test:")
        print(f"H-istatistiği: {h_stat:.4f}")
        print(f"p-değeri: {p_value:.4f}")
        print(f"Sonuç: {'H0 reddedilir' if p_value < 0.05 else 'H0 reddedilemez'}")
        
        # Mann-Whitney U test (ikili karşılaştırmalar)
        print("\n2. Mann-Whitney U testleri:")
        
        # Positive vs Negative
        u_stat, p_value = stats.mannwhitneyu(pos_scores, neg_scores, alternative='two-sided')
        print("\nPositive vs Negative:")
        print(f"U-istatistiği: {u_stat:.4f}")
        print(f"p-değeri: {p_value:.4f}")
        print(f"Sonuç: {'Anlamlı fark var' if p_value < 0.05 else 'Anlamlı fark yok'}")
        
        # Positive vs Neutral
        u_stat, p_value = stats.mannwhitneyu(pos_scores, neu_scores, alternative='two-sided')
        print("\nPositive vs Neutral:")
        print(f"U-istatistiği: {u_stat:.4f}")
        print(f"p-değeri: {p_value:.4f}")
        print(f"Sonuç: {'Anlamlı fark var' if p_value < 0.05 else 'Anlamlı fark yok'}")
        
        # Negative vs Neutral
        u_stat, p_value = stats.mannwhitneyu(neg_scores, neu_scores, alternative='two-sided')
        print("\nNegative vs Neutral:")
        print(f"U-istatistiği: {u_stat:.4f}")
        print(f"p-değeri: {p_value:.4f}")
        print(f"Sonuç: {'Anlamlı fark var' if p_value < 0.05 else 'Anlamlı fark yok'}")
        
        # Görselleştirme
        self.visualize_results()
        
    def visualize_results(self):
        """Sonuçları görselleştir"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='sentiment', y='vader_compound', data=self.df)
        plt.title('VADER Compound Scores by Sentiment Class')
        plt.show()

    def define_research_questions(self):
        """
        Araştırma sorularını ve hipotezleri tanımla
        """
        print("\nAraştırma Soruları ve Hipotezler:")
        print("=" * 50)
        
        research_questions = {
            "RQ1": "Sosyal medya metinlerindeki duygu polaritesi ile kullanıcı etkileşimi arasında bir ilişki var mı?",
            "RQ2": "Emoji kullanımı ile metnin duygu skoru arasında anlamlı bir ilişki var mı?",
            "RQ3": "Metin uzunluğu ile duygu yoğunluğu arasında bir korelasyon var mı?"
        }
        
        hypotheses = {
            "H1": {
                "null": "Duygu polaritesi ile etkileşim sayısı arasında anlamlı bir ilişki yoktur",
                "alternative": "Duygu polaritesi ile etkileşim sayısı arasında anlamlı bir ilişki vardır"
            },
            "H2": {
                "null": "Emoji kullanımı ile VADER duygu skoru arasında anlamlı bir ilişki yoktur",
                "alternative": "Emoji kullanımı ile VADER duygu skoru arasında anlamlı bir ilişki vardır"
            },
            "H3": {
                "null": "Metin uzunluğu ile duygu yoğunluğu arasında anlamlı bir korelasyon yoktur",
                "alternative": "Metin uzunluğu ile duygu yoğunluğu arasında anlamlı bir korelasyon vardır"
            }
        }
        
        for rq, question in research_questions.items():
            print(f"\n{rq}: {question}")
        
        print("\nHipotezler:")
        for h, hyp in hypotheses.items():
            print(f"\n{h}:")
            print(f"H0: {hyp['null']}")
            print(f"H1: {hyp['alternative']}")

    def perform_extended_analysis(self):
        """
        Genişletilmiş veri analizi
        """
        # Metin özellikleri analizi
        self.df['text_length'] = self.df['Text'].str.len()
        self.df['word_count'] = self.df['Text'].str.split().str.len()
        self.df['emoji_count'] = self.df['Text'].apply(lambda x: len(re.findall(r'[^\w\s,]', str(x))))
        
        # Korelasyon analizi
        correlation_matrix = self.df[[
            'text_length', 'word_count', 'emoji_count', 
            'vader_compound', 'vader_pos', 'vader_neg'
        ]].corr()
        
        # Tanımlayıcı istatistikler
        descriptive_stats = self.df.describe()
        
        return correlation_matrix, descriptive_stats

    def perform_advanced_statistical_tests(self):
        """
        Gelişmiş istatistiksel testler
        """
        # Normallik testi
        normality_test = stats.shapiro(self.df['vader_compound'])
        
        # Spearman korelasyonu
        correlation_text_sentiment = stats.spearmanr(
            self.df['text_length'], 
            self.df['vader_compound']
        )
        
        # ANOVA testi
        groups = [
            self.df[self.df['sentiment'] == 'positive']['vader_compound'],
            self.df[self.df['sentiment'] == 'negative']['vader_compound'],
            self.df[self.df['sentiment'] == 'neutral']['vader_compound']
        ]
        f_stat, anova_p = stats.f_oneway(*groups)
        
        return {
            'normality': normality_test,
            'correlation': correlation_text_sentiment,
            'anova': (f_stat, anova_p)
        }

    def perform_comprehensive_analysis(self):
        """
        Kapsamlı istatistiksel analiz
        """
        print("\nKapsamlı İstatistiksel Analiz")
        print("=" * 50)

        # 1. Temel Özellik Hesaplamaları
        self.df['text_length'] = self.df['Text'].str.len()
        self.df['word_count'] = self.df['Text'].str.split().str.len()
        self.df['emoji_count'] = self.df['Text'].apply(lambda x: len(re.findall(r'[^\w\s,]', str(x))))
        self.df['avg_word_length'] = self.df['Text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))

        # 2. Korelasyon Analizleri
        print("\n1. Spearman Korelasyon Analizleri:")
        features = ['text_length', 'word_count', 'emoji_count', 'avg_word_length']
        sentiment_scores = ['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']
        
        for feature in features:
            for score in sentiment_scores:
                correlation, p_value = stats.spearmanr(self.df[feature], self.df[score])
                if p_value < 0.05:
                    print(f"{feature} vs {score}:")
                    print(f"Correlation: {correlation:.3f}, p-value: {p_value:.4f}")

        # 3. Effect Size Hesaplamaları
        print("\n2. Effect Size Analizi:")
        def cohen_d(x, y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

        # Duygu sınıfları arası effect size
        pos_scores = self.df[self.df['sentiment'] == 'positive']['vader_compound']
        neg_scores = self.df[self.df['sentiment'] == 'negative']['vader_compound']
        neu_scores = self.df[self.df['sentiment'] == 'neutral']['vader_compound']

        d_pos_neg = cohen_d(pos_scores, neg_scores)
        d_pos_neu = cohen_d(pos_scores, neu_scores)
        d_neg_neu = cohen_d(neg_scores, neu_scores)

        print(f"Cohen's d (positive vs negative): {d_pos_neg:.3f}")
        print(f"Cohen's d (positive vs neutral): {d_pos_neu:.3f}")
        print(f"Cohen's d (negative vs neutral): {d_neg_neu:.3f}")

        # 4. Regresyon Analizi
        print("\n3. Basit Doğrusal Regresyon Analizleri:")
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        for feature in features:
            X = self.df[feature].values.reshape(-1, 1)
            y = self.df['vader_compound'].values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            print(f"\n{feature} -> vader_compound:")
            print(f"R²: {r2:.3f}")
            print(f"Coefficient: {model.coef_[0]:.3f}")
            print(f"Intercept: {model.intercept_:.3f}")

        # 5. Güç Analizi
        print("\n4. Güç Analizi:")
        from statsmodels.stats.power import TTestPower
        
        effect_sizes = [0.2, 0.5, 0.8]  # small, medium, large
        n = len(self.df)
        
        print("\nMevcut örneklem büyüklüğü için güç değerleri:")
        for effect in effect_sizes:
            power = TTestPower().power(effect_size=effect, nobs=n, alpha=0.05)
            print(f"Effect size {effect}: Power = {power:.3f}")

        # 6. Normallik Testleri
        print("\n5. Normallik Testleri (Shapiro-Wilk):")
        for score in sentiment_scores:
            stat, p_value = stats.shapiro(self.df[score])
            print(f"{score}:")
            print(f"Statistic: {stat:.3f}, p-value: {p_value:.4f}")
            print(f"Normal dağılım: {'Hayır' if p_value < 0.05 else 'Evet'}")

    def perform_advanced_analysis(self):
        """
        Gelişmiş analizler
        """
        results = {}
        
        # 1. Bootstrap Analysis
        compound_scores = self.df['vader_compound'].values
        bootstrap_means = []
        for _ in range(1000):
            sample = np.random.choice(compound_scores, size=len(compound_scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        results['bootstrap_mean_ci'] = np.percentile(bootstrap_means, [2.5, 97.5])

        # 2. Cross Validation
        X = self.df[['text_length', 'word_count', 'emoji_count']].values
        y = self.df['vader_compound'].values
        cv_scores = cross_val_score(RandomForestRegressor(random_state=42), X, y, cv=5)
        results['cross_validation'] = (cv_scores.mean(), cv_scores.std())

        # 3. Chi-square test
        contingency = pd.crosstab(
            self.df['sentiment'],
            pd.qcut(self.df['vader_compound'], q=3, labels=['low', 'medium', 'high'])
        )
        results['chi_square'] = stats.chi2_contingency(contingency)[:2]

        # 4. MANOVA
        features = ['vader_compound', 'text_length', 'emoji_count']
        X = self.df[features]
        manova = MANOVA.from_formula('X ~ sentiment', data=self.df)
        results['manova'] = manova.mv_test()

        # 5. Post-hoc
        tukey = pairwise_tukeyhsd(
            self.df['vader_compound'],
            self.df['sentiment'],
            alpha=0.05
        )
        results['post_hoc'] = tukey

        return results

    def reliability_analysis(self):
        """
        Güvenilirlik analizi
        """
        # Cronbach's alpha için özellikler
        features = ['vader_compound', 'text_length', 'emoji_count']
        X = StandardScaler().fit_transform(self.df[features])
        
        # Korelasyon matrisi
        corr_matrix = np.corrcoef(X.T)
        n = len(features)
        cronbach_alpha = (n / (n-1)) * (1 - np.trace(corr_matrix) / np.sum(corr_matrix))

        # Test-retest için split-half reliability
        n_samples = len(self.df)
        first_half = X[:n_samples//2]
        second_half = X[n_samples//2:]
        reliability = np.corrcoef(
            np.mean(first_half, axis=1),
            np.mean(second_half, axis=1)
        )[0,1]

        return {
            'cronbach_alpha': cronbach_alpha,
            'test_retest': reliability
        }

    def create_advanced_visualizations(self):
        """
        Gelişmiş görselleştirmeler ve kaydetme
        """
        # 1. Duygu Dağılımı
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='sentiment', y='vader_compound', data=self.df)
        plt.title('Duygu Skorları Dağılımı')
        plt.savefig('figures/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Regresyon Analizi
        plt.figure(figsize=(10, 6))
        sns.regplot(x='text_length', y='vader_compound', data=self.df)
        plt.title('Metin Uzunluğu vs Duygu Skoru')
        plt.savefig('figures/regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Korelasyon Matrisi
        plt.figure(figsize=(8, 6))
        correlation_matrix = self.df[['vader_compound', 'text_length', 
                                    'emoji_count', 'word_count']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Korelasyon Matrisi')
        plt.savefig('figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Box Plot - Duygu Sınıfları
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='sentiment', y='vader_compound', data=self.df)
        plt.title('Duygu Sınıflarına Göre VADER Skorları')
        plt.savefig('figures/sentiment_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nGörselleştirmeler 'figures' klasörüne kaydedildi:")
        print("1. sentiment_distribution.png")
        print("2. regression_analysis.png")
        print("3. correlation_matrix.png")
        print("4. sentiment_boxplot.png")

    def improve_reliability(self):
        """Güvenilirlik metriklerini iyileştirme"""
        
        # 1. Feature Selection
        def select_important_features(self):
            """Önemli özellikleri seç"""
            # Korelasyon bazlı feature selection
            features_df = pd.DataFrame([
                self.extract_features(text) for text in self.df['Text']
            ])
            
            # VADER skorları ile korelasyonu yüksek özellikleri seç
            correlations = {}
            for column in features_df.columns:
                corr = stats.spearmanr(features_df[column], self.df['vader_compound'])[0]
                correlations[column] = abs(corr)
            
            # En yüksek korelasyona sahip özellikleri seç
            important_features = [k for k, v in sorted(correlations.items(), 
                                                     key=lambda x: x[1], 
                                                     reverse=True)[:5]]
            return important_features
        
        # 2. Veri Kalitesi Kontrolü
        def check_data_quality(self):
            """Veri kalitesi kontrolü ve temizleme"""
            # Aykırı değerleri tespit et
            z_scores = stats.zscore(self.df[['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']])
            abs_z_scores = np.abs(z_scores)
            outlier_mask = (abs_z_scores < 3).all(axis=1)
            
            # Aykırı değerleri filtrele
            self.df = self.df[outlier_mask]
            
            # Tutarsız etiketleri kontrol et
            vader_pred = np.where(self.df['vader_compound'] > 0.05, 'positive',
                                np.where(self.df['vader_compound'] < -0.05, 'negative', 'neutral'))
            
            # Tutarlı örnekleri seç
            consistency_mask = vader_pred == self.df['sentiment']
            self.df = self.df[consistency_mask]
            
            return self.df
        
        # 3. Feature Engineering
        def engineer_new_features(self):
            """Yeni özellikler oluştur"""
            # Emoji yoğunluğu
            self.df['emoji_density'] = self.df['Text'].apply(
                lambda x: len(re.findall(r'[^\w\s,]', str(x))) / len(str(x))
            )
            
            # Duygu kelimesi yoğunluğu
            self.df['sentiment_word_density'] = self.df['Text'].apply(
                lambda x: sum(1 for word in str(x).split() 
                            if word in self.sentiment_words['positive'] 
                            or word in self.sentiment_words['negative']) / len(str(x).split())
            )
            
            # Noktalama işareti yoğunluğu
            self.df['punctuation_density'] = self.df['Text'].apply(
                lambda x: len(re.findall(r'[!?.]', str(x))) / len(str(x))
            )
            
            return self.df

        # Ana fonksiyon
        print("\nGüvenilirlik iyileştirme süreci başlıyor...")
        
        # 1. Önemli özellikleri seç
        important_features = select_important_features(self)
        print(f"\nSeçilen önemli özellikler: {important_features}")
        
        # 2. Veri kalitesini kontrol et
        original_size = len(self.df)
        self.df = check_data_quality(self)
        print(f"\nVeri kalitesi kontrolü sonrası örneklem sayısı: {len(self.df)}")
        print(f"Çıkarılan örnek sayısı: {original_size - len(self.df)}")
        
        # 3. Yeni özellikler ekle
        self.df = engineer_new_features(self)
        print("\nYeni özellikler eklendi")
        
        # 4. Güvenilirlik metriklerini tekrar hesapla
        cronbach_alpha = self.calculate_cronbach_alpha()
        test_retest = self.calculate_test_retest_reliability()
        
        print("\nGüncellenmiş güvenilirlik metrikleri:")
        print(f"Cronbach's alpha: {cronbach_alpha:.3f}")
        print(f"Test-retest güvenilirliği: {test_retest:.3f}")
        
        return self.df

    def improve_model_performance(self):
        """Model performansını iyileştirme"""
        
        # 1. Gelişmiş Feature Engineering
        def create_advanced_features(self):
            features = {}
            
            # TF-IDF özellikleri
            tfidf = TfidfVectorizer(max_features=100)
            tfidf_features = tfidf.fit_transform(self.df['Text'])
            
            # Duygu yoğunluğu özellikleri
            features['sentiment_intensity'] = self.df['vader_compound'].abs()
            features['emotion_contrast'] = self.df['vader_pos'] - self.df['vader_neg']
            
            # Metin karmaşıklığı özellikleri
            features['text_complexity'] = self.df['Text'].apply(
                lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
            )
            
            return pd.DataFrame(features), tfidf_features
        
        # 2. Model Optimizasyonu
        def optimize_model(self, X, y):
            # Grid Search için parametreler
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Grid Search
            rf = RandomForestClassifier(random_state=42)
            from sklearn.model_selection import GridSearchCV
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='accuracy'
            )
            
            grid_search.fit(X, y)
            return grid_search.best_estimator_, grid_search.best_params_
        
        # 3. Cross-validation iyileştirmesi
        def improved_cross_validation(self, X, y):
            # Stratified K-Fold
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Sonuçları sakla
            cv_scores = []
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # SMOTE uygula
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                # Model eğitimi
                model = self.model.fit(X_train_balanced, y_train_balanced)
                
                # Tahmin
                y_pred = model.predict(X_test)
                
                # Skor hesapla
                cv_scores.append(accuracy_score(y_test, y_pred))
            
            return np.mean(cv_scores), np.std(cv_scores)
        
        print("\nModel performansı iyileştirme süreci başlıyor...")
        
        # 1. Gelişmiş özellikler oluştur
        advanced_features, tfidf_features = create_advanced_features(self)
        X = np.hstack([advanced_features, tfidf_features.toarray()])
        y = self.df['sentiment']
        
        # 2. Model optimizasyonu
        best_model, best_params = optimize_model(self, X, y)
        print(f"\nEn iyi model parametreleri: {best_params}")
        
        # 3. İyileştirilmiş cross-validation
        cv_mean, cv_std = improved_cross_validation(self, X, y)
        print(f"\nİyileştirilmiş Cross-validation sonuçları:")
        print(f"Ortalama accuracy: {cv_mean:.3f} (±{cv_std:.3f})")
        
        # 4. Final model performansı
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        final_model = best_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        
        print("\nFinal Model Performansı:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))
        
        return final_model

    def perform_robustness_analysis(self):
        """Sağlamlık testleri ve duyarlılık analizi"""
        
        print("\nSağlamlık ve Duyarlılık Analizi Başlıyor...")
        results = {}
        
        # 1. Bootstrap Stability Analysis
        def bootstrap_stability(self, n_iterations=1000):
            stability_scores = []
            for _ in range(n_iterations):
                # Bootstrap örneklemi
                bootstrap_idx = np.random.choice(len(self.df), size=len(self.df), replace=True)
                bootstrap_sample = self.df.iloc[bootstrap_idx]
                
                # Temel metrikleri hesapla
                corr = stats.spearmanr(
                    bootstrap_sample['text_length'],
                    bootstrap_sample['vader_compound']
                )[0]
                stability_scores.append(corr)
            
            return np.mean(stability_scores), np.std(stability_scores)
        
        # 2. Sensitivity Analysis
        def sensitivity_analysis(self):
            sensitivities = {}
            
            # Her bir özellik için etki analizi
            features = ['text_length', 'word_count', 'emoji_count', 'avg_word_length']
            for feature in features:
                # Özelliği %10 artır/azalt
                delta = self.df[feature].std() * 0.1
                
                # Artırılmış değerler
                increased = self.df[feature] + delta
                corr_increased = stats.spearmanr(increased, self.df['vader_compound'])[0]
                
                # Azaltılmış değerler
                decreased = self.df[feature] - delta
                corr_decreased = stats.spearmanr(decreased, self.df['vader_compound'])[0]
                
                # Duyarlılık skoru
                sensitivity = abs(corr_increased - corr_decreased) / (2 * delta)
                sensitivities[feature] = sensitivity
            
            return sensitivities
        
        # 3. Cross-Method Validation
        def cross_method_validation(self):
            # Farklı duygu analizi yöntemlerini karşılaştır
            methods = {
                'vader': self.df['vader_compound'],
                'textblob': self.df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity),
                'custom': self.df['Text'].apply(self.custom_sentiment_score)
            }
            
            # Yöntemler arası korelasyon
            correlations = {}
            for m1 in methods:
                for m2 in methods:
                    if m1 < m2:  # Tekrarı önle
                        corr = stats.spearmanr(methods[m1], methods[m2])[0]
                        correlations[f'{m1}_vs_{m2}'] = corr
            
            return correlations
        
        # 4. Error Analysis
        def error_analysis(self):
            errors = {
                'false_positives': [],
                'false_negatives': [],
                'misclassified_neutral': []
            }
            
            # VADER tahminleri
            vader_pred = np.where(self.df['vader_compound'] > 0.05, 'positive',
                                np.where(self.df['vader_compound'] < -0.05, 'negative', 'neutral'))
            
            # Hata analizi
            for idx, (true, pred) in enumerate(zip(self.df['sentiment'], vader_pred)):
                if true != pred:
                    text = self.df.iloc[idx]['Text']
                    if true == 'positive' and pred != 'positive':
                        errors['false_negatives'].append(text)
                    elif true == 'negative' and pred != 'negative':
                        errors['false_positives'].append(text)
                    elif true == 'neutral':
                        errors['misclassified_neutral'].append(text)
            
            return errors
        
        # Analizleri çalıştır
        print("\n1. Bootstrap Stability Analysis")
        stability_mean, stability_std = bootstrap_stability(self)
        print(f"Stabilite skoru: {stability_mean:.3f} (±{stability_std:.3f})")
        
        print("\n2. Sensitivity Analysis")
        sensitivities = sensitivity_analysis(self)
        for feature, sensitivity in sensitivities.items():
            print(f"{feature}: {sensitivity:.3f}")
        
        print("\n3. Cross-Method Validation")
        method_correlations = cross_method_validation(self)
        for comparison, corr in method_correlations.items():
            print(f"{comparison}: {corr:.3f}")
        
        print("\n4. Error Analysis")
        errors = error_analysis(self)
        for error_type, examples in errors.items():
            print(f"\n{error_type.replace('_', ' ').title()}:")
            print(f"Toplam hata sayısı: {len(examples)}")
            if examples:
                print("Örnek hatalar:")
                for ex in examples[:3]:  # İlk 3 örnek
                    print(f"- {ex}")
        
        return {
            'stability': (stability_mean, stability_std),
            'sensitivities': sensitivities,
            'method_correlations': method_correlations,
            'errors': errors
        }

def main():
    analyzer = SentimentFeatureAnalyzer()
    analyzer.analyze_sentiment()
    
    print("\n=== 1. Temel Analizler ===")
    analyzer.perform_comprehensive_analysis()
    
    print("\n=== 2. İleri Düzey Analizler ===")
    advanced_results = analyzer.perform_advanced_analysis()
    
    # Bootstrap sonuçları
    print("\nBootstrap Analizi:")
    if advanced_results and 'bootstrap_mean_ci' in advanced_results:
        bootstrap_ci = advanced_results['bootstrap_mean_ci']
    else:
        bootstrap_ci = [0, 0]  # Varsayılan değerler
    print(f"Güven Aralığı (95%): [{bootstrap_ci[0]:.3f}, {bootstrap_ci[1]:.3f}]")
    
    # Cross-validation sonuçları
    print("\nCross-Validation Sonuçları:")
    if advanced_results and 'cross_validation' in advanced_results:
        cv_mean, cv_std = advanced_results['cross_validation']
    else:
        cv_mean, cv_std = 0, 0  # Varsayılan değerler
    print(f"Ortalama R²: {cv_mean:.3f} (±{cv_std:.3f})")
    
    # Chi-square test sonuçları
    print("\nChi-square Test Sonuçları:")
    if advanced_results and 'chi_square' in advanced_results:
        chi2, p_value = advanced_results['chi_square']
    else:
        chi2, p_value = 0, 0  # Varsayılan değerler
    print(f"Chi-square: {chi2:.3f}")
    print(f"p-değeri: {p_value:.4f}")
    
    # MANOVA sonuçları
    print("\nMANOVA Test Sonuçları:")
    if advanced_results and 'manova' in advanced_results:
        manova_results = advanced_results['manova']
    else:
        manova_results = None  # Varsayılan değerler
    print(manova_results)
    
    # Post-hoc analiz sonuçları
    print("\nPost-hoc Analiz (Tukey HSD):")
    if advanced_results and 'post_hoc' in advanced_results:
        post_hoc_results = advanced_results['post_hoc']
    else:
        post_hoc_results = None  # Varsayılan değerler
    print(post_hoc_results)
    
    print("\n=== 3. Güvenilirlik Analizi ===")
    reliability = analyzer.reliability_analysis()
    print(f"Cronbach's alpha: {reliability['cronbach_alpha']:.3f}")
    print(f"Test-retest güvenilirliği: {reliability['test_retest']:.3f}")
    
    print("\n=== 4. Görselleştirmeler ===")
    analyzer.create_advanced_visualizations()

if __name__ == "__main__":
    main()