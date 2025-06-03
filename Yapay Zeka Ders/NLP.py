import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from tqdm import tqdm
import os
import tkinter as tk
from tkinter import messagebox, font
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer  # LabelBinarizer eklendi

# NLTK'den gerekli veri setlerini indir
try:
    nltk.corpus.stopwords.words("turkish")  # Daha önce indirilmişse hata vermez
except LookupError:
    nltk.download("stopwords")
from nltk.corpus import stopwords

# Veri seti yolu (KENDI DOSYA YOLUNUZU BURAYA YAZIN)
train_data_path = "C:\\Users\\mamie\\Desktop\\Yapay Zeka Ders\\test.csv\\zenginlestirilmis_veri_seti.xlsx"  # Veri setinin yeni adı
test_data_path = "C:\\Users\\mamie\\Desktop\\Yapay Zeka Ders\\test.csv\\test2.xlsx"
valid_data_path = "C:\\Users\\mamie\\Desktop\\Yapay Zeka Ders\\test.csv\\valid2_2_converted.xlsx"  # Valid yolu eklendi

# Metin ve etiket sütunlarının adları (GEREKİRSE DÜZENLEYİN)
text_column = "text"
label_column = "label"
MODEL_PATH = "hate_speech_model_last.pth"  # Kaydedilecek modelin dosya adı
MAX_LEN = 128  # Sabit bir MAX_LEN değeri


# Metin normalizasyon fonksiyonu
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)  # URL'leri kaldır
        text = re.sub(
            r"[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]", "", text
        )  # Özel karakterleri kaldır (Türkçe karakterler eklendi)
        text = text.strip()
        # Stop words kaldırma (Türkçe)
        stop_words = set(stopwords.words("turkish"))
        text_tokens = text.split()
        filtered_tokens = [w for w in text_tokens if not w in stop_words]
        text = " ".join(filtered_tokens)
        return text
    return ""


# Veri setlerini yükleme
def load_data():
    try:
        train_df = pd.read_excel(train_data_path)
        test_df = pd.read_excel(test_data_path)
        valid_df = pd.read_excel(valid_data_path)  # valid_df'yi de yükle
        print("Veri setleri başarıyla yüklendi.")

        # ID sütununu düşürme
        for df in [train_df, test_df, valid_df]:
            if "id" in df.columns:
                df.drop("id", axis=1, inplace=True)
                print(f"Bir veri setinden 'id' sütunu kaldırıldı.")

        return train_df, test_df, valid_df  # valid_df'yi de döndür
    except FileNotFoundError as e:
        print(
            f"Veri seti dosyası bulunamadı: {e}. Lütfen dosya yollarını kontrol edin."
        )
        return None, None, None
    except Exception as e:
        print(f"Veri setini yüklerken bir hata oluştu: {e}")
        return None, None, None


# Bert Tokenizer yükleme
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Dataset sınıfı
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Ensure label is an integer before creating a tensor
        # This is the crucial part to address the TypeError
        if not isinstance(label, (int, float, np.integer, np.floating)):
            try:
                label = int(label)
            except ValueError:
                # Handle cases where label might be an unconvertible string (e.g., 'NaN', 'missing')
                # You might want to log this or assign a default value
                print(
                    f"Warning: Could not convert label '{label}' to int. Skipping or handling this item."
                )
                # Depending on your data, you might want to:
                # 1. Skip the item (e.g., raise an error or return None, but this needs handling in DataLoader)
                # 2. Assign a default label (e.g., 0 or -1 for 'unknown')
                # For now, let's assume it should always be convertible and raise if not
                raise

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Parametreler
EPOCHS = 16
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_LABELS = 0  # main fonksiyonunda belirlenecek, başlangıç değeri 0


# DataLoader oluşturma
def create_data_loaders(
    train_texts,
    train_labels,
    valid_texts,
    valid_labels,
    test_texts,
    test_labels,
    tokenizer,
    max_len,
    batch_size,
):
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, max_len)
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, max_len)
    valid_dataset = HateSpeechDataset(valid_texts, valid_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, valid_loader, test_loader


# Model yükleme
def load_model(
    num_labels,
    device,
    model_path="C:\\Users\\mamie\\Desktop\\Yapay Zeka Ders\\hate_speech_model_last.pth",
):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, hidden_dropout_prob=0.3
    )
    model.to(device)
    try:
        if os.path.exists(model_path):  # Check if model file exists
            model.load_state_dict(torch.load(model_path))
            print("Model başarıyla yüklendi.")
            return model
        else:
            print(f"Model dosyası bulunamadı: {model_path}. Model eğitilecek.")
            return None  # Return None if model file doesn't exist to trigger training
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        return None


# Eğitim fonksiyonu
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(data_loader, leave=True, desc="Eğitim")

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # İlerleme çubuğunu güncelle
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    return total_loss / len(data_loader), 100 * correct / total


# Değerlendirme fonksiyonu
def eval_model(
    model, data_loader, loss_fn, device, desc="Değerlendirme", return_probs=False
):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    loop = tqdm(data_loader, leave=True, desc=desc)

    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # İlerleme çubuğunu güncelle
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    if return_probs:
        return (
            total_loss / len(data_loader),
            100 * correct / total,
            all_preds,
            all_labels,
            all_probs,
        )
    else:
        return (
            total_loss / len(data_loader),
            100 * correct / total,
            all_preds,
            all_labels,
        )


# Eğitim döngüsü
def train_model(
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    epochs=16,
    early_stopping_patience=3,
    model_path="hate_speech_model_last.pth",
):
    best_val_loss = float("inf")
    early_stopping_counter = 0
    accuracies = []
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler
        )
        print(f"Eğitim Kaybı: {train_loss:.4f}, Eğitim Doğruluğu: {train_acc:.2f}%")

        val_loss, val_acc, val_preds, val_labels = eval_model(
            model, valid_loader, loss_fn, device, desc="Doğrulama"
        )
        print(f"Doğrulama Kaybı: {val_loss:.4f}, Doğrulama Doğruluğu: {val_acc:.2f}%")
        accuracies.append((epoch + 1, val_acc))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early Stopping kontrolü
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)  # En iyi modeli kaydetme yolu
            print(f"En iyi model kaydedildi: {model_path}")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Erken durdurma tetiklendi.")
            break
    return accuracies, history


def predict_hate_speech(text, model, tokenizer, device, max_len=128):
    """Verilen bir metnin nefret söylemi içerip içermediğini tahmin eder.

    Args:
        text (str): Tahmin edilecek metin.
        model (BertForSequenceClassification): Eğitilmiş BERT modeli.
        tokenizer (BertTokenizer): BERT tokenizer.
        device (torch.device): İşlem yapılacak cihaz (CPU veya GPU).
        max_len (int): Metinlerin maksimum uzunluğu.

    Returns:
        str: "Nefret Söylemi" veya "Nefret Söylemi Değil".
    """
    model.eval()  # Tahmin için modeli değerlendirme moduna alın

    cleaned_text = clean_text(text)  # Metni temizle

    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.softmax(outputs.logits, dim=1)  # Olasılık dağılımı al
        predicted_class = torch.argmax(predictions, dim=1).item()
    if predicted_class == 1:
        return "Nefret Söylemi"
    else:
        return "Nefret Söylemi Değil"


# --- Plotting Fonksiyonları ---


def plot_roc_curve(y_true_binarized, y_probs, class_names):
    """ROC eğrisini çizer ve AUC değerlerini gösterir."""
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Her sınıf için ROC eğrisini ve AUC alanını hesapla
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Mikro-ortalama ROC eğrisini ve AUC alanını hesapla
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # ROC eğrilerini çiz
    plt.figure(figsize=(8, 6))

    # Düzeltilmiş renk seçimi: Colormap'den renkleri direkt alıyoruz
    colors = [
        plt.colormaps["jet"](i / n_classes) for i in range(n_classes)
    ]  # Veya plt.get_cmap('jet')(i / n_classes)

    for i, color in zip(range(n_classes), cycle(colors)):  # Buraya cycle ekledik
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})"
            "".format(class_names[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Multi-Class")
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true_binarized, y_probs, class_names):
    """Precision-Recall eğrisini çizer ve AP değerlerini gösterir."""
    n_classes = len(class_names)
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Her sınıf için Precision-Recall eğrisini ve AP'yi hesapla
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_binarized[:, i], y_probs[:, i]
        )
        average_precision[i] = average_precision_score(
            y_true_binarized[:, i], y_probs[:, i]
        )

    # Mikro-ortalama Precision-Recall eğrisini hesapla
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_binarized.ravel(), y_probs.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_binarized, y_probs, average="micro"
    )

    # Precision-Recall eğrilerini çiz
    plt.figure(figsize=(8, 6))

    # Düzeltilmiş renk seçimi: Colormap'den renkleri direkt alıyoruz
    colors = [
        plt.colormaps["rainbow"](i / n_classes) for i in range(n_classes)
    ]  # Veya plt.get_cmap('rainbow')(i / n_classes)

    for i, color in zip(range(n_classes), cycle(colors)):  # Buraya cycle ekledik
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label="Precision-recall for class {0} (AP = {1:0.2f})"
            "".format(class_names[i], average_precision[i]),
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall Curve for Multi-Class")
    plt.legend(loc="best")
    plt.show()


def plot_training_history(history):
    """Eğitim ve doğrulama kaybı ile doğruluk grafiklerini çizer."""
    plt.figure(figsize=(12, 5))

    # Doğruluk Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Eğitim Doğruluğu")
    plt.plot(history["val_acc"], label="Doğrulama Doğruluğu")
    plt.title("Eğitim ve Doğrulama Doğruluğu")
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk (%)")
    plt.legend()
    plt.grid(True)

    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Eğitim Kaybı")
    plt.plot(history["val_loss"], label="Doğrulama Kaybı")
    plt.title("Eğitim ve Doğrulama Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Main Fonksiyonu ---
def main():
    # Veri setini yükle
    train_df, test_df, valid_df = load_data()  # valid_df'yi de yükle
    if train_df is None or test_df is None or valid_df is None:
        return  # Veri yüklenemezse çık

    # Metin temizleme
    train_df[text_column] = train_df[text_column].apply(clean_text)
    test_df[text_column] = test_df[text_column].apply(clean_text)
    valid_df[text_column] = valid_df[text_column].apply(
        clean_text
    )  # valid_df'yi de temizle

    # *** CRUCIAL FIX: Ensure label column is of integer type ***
    # Before proceeding with balancing and dataset creation,
    # convert the label column to integer type.
    try:
        train_df[label_column] = train_df[label_column].astype(int)
        test_df[label_column] = test_df[label_column].astype(int)
        valid_df[label_column] = valid_df[label_column].astype(int)
        print("Label sütunları başarıyla int türüne dönüştürüldü.")
    except ValueError as e:
        print(
            f"Hata: Etiket sütununda sayısal olmayan değerler var. Lütfen Excel dosyalarınızı kontrol edin. Hata: {e}"
        )
        return  # Exit if labels cannot be converted

    # Sınıf dağılımını dengeleme (oversampling)
    class_counts = train_df[label_column].value_counts()
    min_class = class_counts.idxmin()
    df_minority = train_df[train_df[label_column] == min_class]
    if class_counts.min() > 0:  # Bölme hatasını önlemek için kontrol
        replication_factor = class_counts.max() // class_counts.min() - 1
        if replication_factor > 0:
            train_df_balanced = pd.concat(
                [train_df] + [df_minority] * replication_factor,
                ignore_index=True,
            )
        else:  # Eğer max ve min sayılar eşitse veya min daha büyükse kopyalamaya gerek yok
            train_df_balanced = train_df.copy()
    else:  # Eğer bir sınıfın sayısı 0 ise, dengeleme yapma
        train_df_balanced = train_df.copy()
        print("Uyarı: Bir sınıfın örnek sayısı 0 olduğu için dengeleme atlandı.")

    # Veriyi ayırma (Artık train_test_split ile validasyon setine bölmeye gerek yok)
    train_texts = train_df_balanced[text_column].values
    train_labels = train_df_balanced[label_column].values
    test_texts = test_df[text_column].values
    test_labels = test_df[label_column].values
    valid_texts = valid_df[text_column].values  # Doğrudan valid_df'den al
    valid_labels = valid_df[label_column].values  # Doğrudan valid_df'den al

    # Etiket sayısını belirle
    global NUM_LABELS
    all_unique_labels = np.unique(
        np.concatenate((train_labels, test_labels, valid_labels))
    )
    NUM_LABELS = len(all_unique_labels)
    print(f"Tespit edilen sınıf sayısı (NUM_LABELS): {NUM_LABELS}")

    if NUM_LABELS < 2:
        print(
            "Uyarı: ROC ve Precision-Recall eğrileri için en az iki sınıf gereklidir. Veri setinizde yalnızca bir sınıf bulunuyor veya hatalı etiketleme yapılmış olabilir."
        )
        # Burada ROC ve PR eğrisi çizimini atlayabilir veya başka bir uyarı verebilirsiniz.

    # Tokenizer zaten globalde tanımlı

    # Data Loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_texts,
        train_labels,
        valid_texts,
        valid_labels,
        test_texts,
        test_labels,
        tokenizer,
        MAX_LEN,
        BATCH_SIZE,
    )
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli yükle
    model = load_model(NUM_LABELS, device, MODEL_PATH)

    # Sınıf ağırlıkları (hem eğitim hem de değerlendirme için kullanılabilir)
    class_counts_balanced = pd.Series(train_labels).value_counts().sort_index().values
    # Ensure class_weights has correct shape (NUM_LABELS)
    if len(class_counts_balanced) == NUM_LABELS and np.all(class_counts_balanced > 0):
        class_weights = torch.tensor(
            [sum(class_counts_balanced) / c for c in class_counts_balanced]
        ).to(device)
    else:
        print(
            "Uyarı: Sınıf ağırlıkları hesaplanırken sorun oluştu. Tüm sınıflar için ağırlıklar 1 olarak ayarlandı."
        )
        class_weights = torch.tensor([1.0] * NUM_LABELS).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    if model is None:
        # Eğer model yüklenemezse (yani dosya yoksa), eğitmeye devam et
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=NUM_LABELS, hidden_dropout_prob=0.3
        )
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * EPOCHS,
        )
        # Modeli eğit
        accuracies, history = train_model(
            model,
            train_loader,
            valid_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            epochs=EPOCHS,
            early_stopping_patience=3,
            model_path=MODEL_PATH,
        )
        print(f"Model başarıyla eğitildi ve {MODEL_PATH} konumuna kaydedildi.")
    else:
        print("Önceden eğitilmiş model kullanılıyor.")
        history = None  # Geçmişi boş olarak ayarlayalım

    # Eğitim veya yükleme sonrası model değerlendirmesi
    print("\n--- Test Seti Değerlendirmesi ---")
    test_loss, test_acc, test_preds, test_labels, test_probs = eval_model(
        model, test_loader, loss_fn, device, desc="Test", return_probs=True
    )
    print(f"Test Kaybı: {test_loss:.4f}, Test Doğruluğu: {test_acc:.2f}%")

    # --- Metrikleri Yazdırma ---
    print("\n--- Model Değerlendirme Metrikleri ---")
    print("Test Doğruluğu:", f"{test_acc:.2f}%")

    # class_names'i doğru şekilde belirleyin
    class_names = [
        str(int(i)) for i in sorted(all_unique_labels)
    ]  # Etiketler sayısal olduğu için int'e dönüştürün

    print("\nSınıflandırma Raporu:")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # --- Karışıklık Matrisi ---
    cm = confusion_matrix(test_labels, test_preds)
    print("\nKarışıklık Matrisi:")
    print(cm)

    # --- Karışıklık Matrisi Grafiği ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Tahmin Edilen Etiket")
    plt.ylabel("Gerçek Etiket")
    plt.title("Karışıklık Matrisi")
    plt.show()

    # --- ROC Eğrisi ve Precision-Recall Eğrisi için etiketleri ikili hale getirme ---
    test_probs_array = np.array(test_probs)
    test_labels_array = np.array(test_labels)

    if NUM_LABELS >= 2:  # ROC ve PR eğrilerini sadece birden fazla sınıf varsa çiz
        # LabelBinarizer kullanarak y_true'yu one-hot encode et
        lb = LabelBinarizer()
        lb.fit(
            all_unique_labels
        )  # Tüm olası etiketleri öğrenmesini sağlayın (0, 1 gibi)
        y_true_binarized = lb.transform(test_labels_array)

        # Eğer ikili sınıflandırma yapılıyorsa (yani sadece 2 sınıf varsa)
        # LabelBinarizer tek bir sütun döndürebilir. Bu durumda, 2. sütunu da eklemeliyiz (1 - ilk sütun)
        if y_true_binarized.shape[1] == 1 and NUM_LABELS == 2:
            y_true_binarized = np.hstack((1 - y_true_binarized, y_true_binarized))
        elif y_true_binarized.shape[1] == 1 and NUM_LABELS > 2:
            print(
                "Hata: LabelBinarizer tek sütun döndürdü ancak birden fazla sınıf var. ROC ve PR eğrileri çizilemedi."
            )
            y_true_binarized = None  # Grafik çizilmesini engelle

        if y_true_binarized is not None:
            # --- ROC Eğrisi ---
            plot_roc_curve(y_true_binarized, test_probs_array, class_names)

            # --- Precision-Recall Eğrisi ---
            plot_precision_recall_curve(y_true_binarized, test_probs_array, class_names)
    else:
        print(
            "ROC ve Precision-Recall eğrileri çizilemedi: En az iki sınıf gereklidir."
        )

    # --- Doğruluk ve Kayıp Grafiği (Eğer eğitim yapıldıysa) ---
    if history is not None:
        plot_training_history(history)

    # GUI oluştur
    root = tk.Tk()
    root.title("Nefret Söylemi Tespiti")
    root.geometry("400x200")

    # Yazı tipi ayarı
    font_style = font.Font(family="Arial", size=12)

    input_label = tk.Label(root, text="Metin Girin:", font=font_style)
    input_label.pack(pady=10)

    input_text = tk.Entry(root, width=50, font=font_style)
    input_text.pack(pady=10)

    def predict_and_show():
        text = input_text.get()
        if not text:
            messagebox.showerror("Hata", "Lütfen bir metin girin.")
            return

        prediction = predict_hate_speech(text, model, tokenizer, device)
        messagebox.showinfo("Sonuç", f"Girdi: '{text}'\nTahmin: {prediction}")

    predict_button = tk.Button(
        root, text="Tahmin Et", command=predict_and_show, font=font_style
    )
    predict_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
