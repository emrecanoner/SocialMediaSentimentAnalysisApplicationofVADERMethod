import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# GPU kullanılabilir mi kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Veriyi oku
df = pd.read_csv('dataset/cleaned_sentiment_dataset.csv')

def prepare_data():
    """Veriyi model için hazırla"""
    # En sık görülen 3 duyguyu seç (örnek: positive, negative, neutral)
    top_sentiments = df['Sentiment'].value_counts().head(3).index
    filtered_df = df[df['Sentiment'].isin(top_sentiments)]
    
    # Etiketleri sayısala çevir
    label_dict = {label: i for i, label in enumerate(top_sentiments)}
    filtered_df['label'] = filtered_df['Sentiment'].map(label_dict)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        filtered_df['cleaned_text'],
        filtered_df['label'],
        test_size=0.2,
        random_state=42,
        stratify=filtered_df['label']
    )
    
    return X_train, X_test, y_train, y_test, label_dict

def tokenize_data(texts, tokenizer, max_length=128):
    """Metinleri tokenize et"""
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def create_dataloader(encodings, labels, batch_size=16):
    """DataLoader oluştur"""
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels.values)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, train_dataloader, val_dataloader, epochs=3):
    """Modeli eğit"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        
        # Eğitim
        for batch in tqdm(train_dataloader, desc="Eğitim"):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Validasyon
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validasyon"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                val_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        print(f"Ortalama eğitim kaybı: {train_loss/len(train_dataloader):.4f}")
        print(f"Ortalama validasyon kaybı: {val_loss/len(val_dataloader):.4f}")
        
        # Validasyon metrikleri
        print("\nValidasyon sonuçları:")
        print(classification_report(val_labels, val_preds))

def plot_confusion_matrix(y_true, y_pred, labels):
    """Karmaşıklık matrisini görselleştir"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()

def main():
    # Veriyi hazırla
    X_train, X_test, y_train, y_test, label_dict = prepare_data()
    
    # Tokenizer ve model
    model_name = "dbmdz/bert-base-turkish-cased"  # Türkçe BERT modeli
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_dict)
    ).to(device)
    
    # Veriyi tokenize et
    train_encodings = tokenize_data(X_train, tokenizer)
    test_encodings = tokenize_data(X_test, tokenizer)
    
    # DataLoader'ları oluştur
    train_dataloader = create_dataloader(train_encodings, y_train)
    test_dataloader = create_dataloader(test_encodings, y_test)
    
    # Modeli eğit
    print("Model eğitimi başlıyor...")
    train_model(model, train_dataloader, test_dataloader)
    
    # Test seti üzerinde tahminler
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            test_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Sonuçları görselleştir
    print("\nTest seti sonuçları:")
    print(classification_report(test_labels, test_preds))
    
    # Karmaşıklık matrisini çiz
    label_names = [k for k, v in sorted(label_dict.items(), key=lambda x: x[1])]
    plot_confusion_matrix(test_labels, test_preds, label_names)
    
    # Modeli kaydet
    model.save_pretrained('models/sentiment_model')
    tokenizer.save_pretrained('models/sentiment_model')
    print("\nModel kaydedildi: models/sentiment_model")

if __name__ == "__main__":
    main() 