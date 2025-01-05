import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def prepare_data():
    """Veriyi daha iyi hazırla"""
    df = pd.read_csv('dataset/cleaned_sentiment_dataset.csv')
    
    # Duygu etiketlerini dengele
    min_count = df['Sentiment'].value_counts().min()
    balanced_df = pd.concat([
        df[df['Sentiment'] == label].sample(min_count)
        for label in df['Sentiment'].unique()
    ])
    
    # Etiketleri sayısala çevir
    label_dict = {label: i for i, label in enumerate(balanced_df['Sentiment'].unique())}
    balanced_df['label'] = balanced_df['Sentiment'].map(label_dict)
    
    return balanced_df, label_dict

def compute_metrics(eval_pred):
    """Değerlendirme metrikleri"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Metrikleri hesapla
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }

def main():
    # Veriyi hazırla
    df, label_dict = prepare_data()
    
    # Model ve tokenizer
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['cleaned_text'], padding='max_length', truncation=True, max_length=128)
    
    # Dataset oluştur
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Eğitim parametreleri
    training_args = TrainingArguments(
        output_dir="models/improved_sentiment_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_dict)
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset.shuffle(seed=42).select(range(int(len(tokenized_dataset)*0.8))),
        eval_dataset=tokenized_dataset.shuffle(seed=42).select(range(int(len(tokenized_dataset)*0.8), len(tokenized_dataset))),
        compute_metrics=compute_metrics,
    )
    
    # Eğitim
    print("Model eğitimi başlıyor...")
    trainer.train()
    
    # En iyi modeli kaydet
    trainer.save_model("models/improved_sentiment_model")
    tokenizer.save_pretrained("models/improved_sentiment_model")
    
    # Etiket sözlüğünü kaydet
    import json
    with open("models/improved_sentiment_model/label_dict.json", "w") as f:
        json.dump(label_dict, f)
    
    print("İyileştirilmiş model kaydedildi!")

if __name__ == "__main__":
    main() 