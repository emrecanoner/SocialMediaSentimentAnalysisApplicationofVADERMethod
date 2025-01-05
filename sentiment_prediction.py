import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

class SentimentAnalyzer:
    def __init__(self, model_path='models/sentiment_model'):
        # Model ve tokenizer'ı yükle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Etiket sözlüğünü oluştur (model eğitiminde kullanılan sırayla aynı olmalı)
        self.label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    def predict(self, text):
        # Metni tokenize et
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
        
        # Olasılıkları al
        probabilities = predictions[0].cpu().numpy()
        
        # Sonuçları sözlük olarak döndür
        result = {
            'text': text,
            'sentiment': self.label_dict[predicted_label],
            'confidence': float(probabilities[predicted_label]),
            'probabilities': {
                label: float(prob)
                for label, prob in zip(self.label_dict.values(), probabilities)
            }
        }
        
        return result

def main():
    # Örnek metinler
    test_texts = [
        "Bu film gerçekten harikaydı, kesinlikle tavsiye ederim!",
        "Çok kötü bir deneyimdi, hiç memnun kalmadım.",
        "Film fena değildi, ama daha iyi olabilirdi.",
        "Harika bir gün geçirdim, her şey mükemmeldi!",
        "Maalesef beklentilerimi karşılamadı.",
    ]
    
    # Analiz yapacak sınıfı başlat
    analyzer = SentimentAnalyzer()
    
    # Her metin için tahmin yap ve sonuçları göster
    print("\nDuygu Analizi Sonuçları:")
    print("-" * 50)
    
    for text in test_texts:
        result = analyzer.predict(text)
        
        print(f"\nMetin: {result['text']}")
        print(f"Duygu: {result['sentiment']}")
        print(f"Güven Skoru: {result['confidence']:.2%}")
        print("\nTüm Olasılıklar:")
        for sentiment, prob in result['probabilities'].items():
            print(f"- {sentiment}: {prob:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    main() 