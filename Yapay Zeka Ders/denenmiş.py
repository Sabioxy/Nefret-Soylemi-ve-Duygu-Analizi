import torch
from transformers import BertTokenizer, BertForSequenceClassification
import customtkinter as ctk
from tkinter import messagebox, filedialog
import speech_recognition as sr
import whisper
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from pathlib import Path
import threading
import pickle

# --- MODEL YOLLARI ---
HATE_MODEL_PATH = "hate_speech_model_last.pth"
EMOTION_MODEL_PATH = "best_model1_weights.h5"
SCALER_PATH = "scaler2.pickle"
ENCODER_PATH = "encoder2.pickle"  # Duygu etiketlerini geri dönüştürmek için (şu an label_map kullanılıyor)

# Cihaz yapılandırması (GPU varsa CUDA, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global Model/Araç Yükleyicileri (Uygulama başladığında bir kez yüklenir) ---
# Tokenizer ve Whisper modeli büyük oldukları ve sık kullanıldıkları için globalde yüklenir
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
whisper_model = whisper.load_model("small")


# --- EĞİTİM KODUNUZDAN ALINAN YARDIMCI FONKSİYONLAR ---
# Bu fonksiyonlar, eğitimde özellik çıkarma için kullandığınız mantığı yansıtır.


# ZCR (Zero Crossing Rate) hesaplama
def zcr_feature(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(
        data, frame_length=frame_length, hop_length=hop_length
    )
    return np.squeeze(zcr)


# RMSE (Root Mean Square Energy) hesaplama
def rmse_feature(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


# MFCC (Mel-frequency Cepstral Coefficients) hesaplama
# Eğitim kodunuzdaki mfcc fonksiyonu n_mfcc'yi belirtmediği için varsayılan 20 kullanılır.
# Ayrıca, mfcc.T'yi düzleştirir (ravel).
def mfcc_feature(data, sr, frame_length=2048, hop_length=512):
    # n_mfcc belirtilmediği için librosa'nın varsayılanı olan 20 kullanılır.
    # Eğer eğitimde farklı bir n_mfcc kullandıysanız, burayı güncelleyin.
    mfcc = librosa.feature.mfcc(
        y=data, sr=sr, n_fft=frame_length, hop_length=hop_length
    )
    return np.ravel(mfcc.T)  # Eğitimdeki gibi düzleştir (flatten)


# Tüm özellikleri birleştirme (ZCR, RMSE, MFCC)
def extract_all_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack(
        (
            result,
            zcr_feature(data, frame_length, hop_length),
            rmse_feature(data, frame_length, hop_length),
            mfcc_feature(data, sr, frame_length, hop_length),  # n_mfcc=20 varsayılan
        )
    )
    return result


# Duygu tanıma CNN model mimarisini oluşturan fonksiyon
# BU FONKSİYON KAGGLE'DA EĞİTTİĞİNİZ MİMARİ İLE BİREBİR AYNI OLMALIDIR!
# Conv1D için girdi şekli: (timesteps, features) olmalıdır.
# Sizin kodunuzda X_train.shape[1] timesteps'ı (2376), 1 ise feature sayısını temsil eder.
def create_emotion_model_architecture(input_shape, num_classes):
    model = Sequential(
        [
            Conv1D(
                512,
                kernel_size=5,
                strides=1,
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=5, strides=2, padding="same"),
            Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=5, strides=2, padding="same"),
            Dropout(0.2),
            Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=5, strides=2, padding="same"),
            Conv1D(256, kernel_size=3, strides=1, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=5, strides=2, padding="same"),
            Dropout(0.2),
            Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=3, strides=2, padding="same"),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


# Duygu etiketleri - Burası düzeltildi!
# Bu sıralama, gözlemlediğiniz yanlış tahminlere göre ayarlanmıştır.
label_map = {
    0: "😠 Sinirli",
    1: "😨 Korkmuş",
    2: "😄 Mutlu",
    3: "😐 Nötr",
    4: "😢 Üzgün",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere yapılandırması
        self.title("Duygu ve Metin Analiz Uygulaması")
        self.geometry("800x700")
        self.minsize(750, 650)

        # Tema ayarı
        ctk.set_appearance_mode("System")  # System, Light, Dark
        ctk.set_default_color_theme("dark-blue")

        # Nefret söylemi modelini başlat (PyTorch BERT modeli)
        self.hate_model = load_hate_model(HATE_MODEL_PATH, 2, device)
        self.is_listening = False

        # --- Duygu Modelini (Keras CNN) ve Ön İşleme Araçlarını Yükle ---
        # ÖNEMLİ: Bu değerler Kaggle'da eğitim sırasında kullanılanlarla BİREBİR AYNI OLMALIDIR!
        # Sizin verdiğiniz bilgilere göre güncellendi:
        # Eğitim kodunuzdaki X_train.shape[1] değeri (2376)
        self.emotion_model_timesteps = 2376
        self.emotion_model_num_classes = (
            5  # Duygu sınıf sayısı (label_map ile eşleşmeli)
        )

        # Duygu modeli mimarisini tanımla
        # Conv1D için girdi şekli: (zaman_adımı, özellik_sayısı) -> (self.emotion_model_timesteps, 1)
        emotion_model_input_shape = (self.emotion_model_timesteps, 1)  # Yani (2376, 1)
        self.emotion_model = create_emotion_model_architecture(
            emotion_model_input_shape, self.emotion_model_num_classes
        )

        # Duygu modeli ağırlıklarını yükle
        try:
            self.emotion_model.load_weights(EMOTION_MODEL_PATH)
            print(
                f"Duygu modeli ağırlıkları '{EMOTION_MODEL_PATH}' adresinden yüklendi."
            )
        except Exception as e:
            messagebox.showerror(
                "Model Yükleme Hatası",
                f"Duygu modeli ağırlıkları yüklenemedi: {str(e)}\n"
                f"Lütfen '{EMOTION_MODEL_PATH}' dosyasının doğru yolda olduğundan ve model mimarisiyle eşleştiğinden emin olun.",
            )
            self.emotion_model = (
                None  # Yüklenemezse daha fazla hatayı önlemek için None olarak ayarla
            )

        # Scaler'ı yükle (MFCC normalizasyonu için)
        try:
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"Scaler '{SCALER_PATH}' yüklendi.")
        except Exception as e:
            messagebox.showerror(
                "Scaler Yükleme Hatası",
                f"Scaler yüklenemedi: {str(e)}\n"
                f"Lütfen '{SCALER_PATH}' dosyasının doğru yolda olduğundan emin olun.",
            )
            self.scaler = (
                None  # Yüklenemezse daha fazla hatayı önlemek için None olarak ayarla
            )

        # Encoder'ı yükle (duygu etiketi ters dönüşümü için - isteğe bağlı, şimdilik label_map kullanılıyor)
        try:
            with open(ENCODER_PATH, "rb") as f:
                self.encoder = pickle.load(f)
            print(f"Encoder '{ENCODER_PATH}' yüklendi.")
        except Exception as e:
            messagebox.showwarning(
                "Encoder Yükleme Uyarısı",
                f"Encoder yüklenemedi: {str(e)}\n"
                f"Lütfen '{ENCODER_PATH}' dosyasının doğru yolda olduğundan emin olun.",
            )
            self.encoder = (
                None  # Yüklenemezse daha fazla hatayı önlemek için None olarak ayarla
            )

        # UI oluştur
        self.create_widgets()

    def create_widgets(self):
        # Başlık çerçevesi
        header_frame = ctk.CTkFrame(self, corner_radius=10)
        header_frame.pack(pady=15, padx=15, fill="x")

        # Başlık etiketi
        title_label = ctk.CTkLabel(
            header_frame,
            text="Duygu ve Nefret Söylemi Analiz Aracı",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.pack(pady=(10, 5))

        # Alt başlık
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Metin veya ses kaydını analiz edin",
            font=ctk.CTkFont(size=14),
            text_color="gray70",
        )
        subtitle_label.pack(pady=(0, 10))

        # Girdi çerçevesi
        input_frame = ctk.CTkFrame(self, corner_radius=10)
        input_frame.pack(pady=10, padx=15, fill="x")

        # Metin girdi alanı
        self.input_text = ctk.CTkEntry(
            input_frame,
            placeholder_text="Analiz etmek istediğiniz metni buraya yazın...",
            height=40,
            font=ctk.CTkFont(size=14),
        )
        self.input_text.pack(pady=10, padx=10, fill="x")

        # Düğme ızgarası
        button_grid = ctk.CTkFrame(self, corner_radius=10)
        button_grid.pack(pady=10, padx=200, fill="x")

        # Satır 1
        row1 = ctk.CTkFrame(button_grid, fg_color="transparent")
        row1.pack(fill="x", pady=5)

        self.mic_button = ctk.CTkButton(
            row1,
            text="🎤 Mikrofonla Konuş",
            command=self.start_listening_thread,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
        )
        self.mic_button.pack(side="left", padx=5)

        self.predict_button = ctk.CTkButton(
            row1,
            text="🔍 Metni Analiz Et",
            command=self.analyze_text,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
        )
        self.predict_button.pack(side="left", padx=5)

        # Satır 2
        row2 = ctk.CTkFrame(button_grid, fg_color="transparent")
        row2.pack(fill="x", pady=5)

        self.file_button = ctk.CTkButton(
            row2,
            text="📄 Metin Dosyası Aç",
            command=self.analyze_file,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
        )
        self.file_button.pack(side="left", padx=5)

        self.audio_button = ctk.CTkButton(
            row2,
            text="🔊 Ses Dosyası Analizi",
            command=self.analyze_audio,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
        )
        self.audio_button.pack(side="left", padx=5)

        # Satır 3
        row3 = ctk.CTkFrame(button_grid, fg_color="transparent")
        row3.pack(fill="x", pady=5)

        self.emotion_button = ctk.CTkButton(
            row3,
            text="😊 Duygu Analizi Yap",
            command=self.analyze_emotion,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            fg_color="#FF6B6B",
            hover_color="#FF8E8E",
        )
        self.emotion_button.pack(side="left", padx=5)

        self.clear_button = ctk.CTkButton(
            row3,
            text="🧹 Temizle",
            command=self.clear_output,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            fg_color="#4ECDC4",
            hover_color="#88D8C0",
        )
        self.clear_button.pack(side="left", padx=5)

        # Çıktı çerçevesi
        output_frame = ctk.CTkFrame(self, corner_radius=10)
        output_frame.pack(pady=10, padx=15, fill="both", expand=True)

        # Çıktı etiketi
        output_label = ctk.CTkLabel(
            output_frame,
            text="Analiz Sonuçları",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        output_label.pack(pady=(10, 5))

        # Kaydırma çubuklu çıktı metin kutusu
        self.output_text = ctk.CTkTextbox(
            output_frame,
            width=750,
            height=300,
            font=ctk.CTkFont(size=13),
            wrap="word",
            activate_scrollbars=True,
        )
        self.output_text.pack(pady=5, padx=10, fill="both", expand=True)

        # Durum çubuğu
        self.status_var = ctk.StringVar(value="Hazır")
        status_bar = ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color="gray70",
            anchor="w",
        )
        status_bar.pack(side="bottom", fill="x", padx=15, pady=5)

    def start_listening_thread(self):
        if not self.is_listening:
            threading.Thread(target=self.recognize_speech, daemon=True).start()

    def recognize_speech(self):
        self.is_listening = True
        self.mic_button.configure(state="disabled", text="Dinleniyor...")
        self.status_var.set("Mikrofondan dinleniyor... Konuşun")

        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio, language="tr-TR")
            self.input_text.delete(0, "end")
            self.input_text.insert(0, text)
            self.analyze_text()
        except sr.WaitTimeoutError:
            self.status_var.set("Mikrofon zaman aşımına uğradı")
        except Exception as e:
            self.status_var.set(f"Hata: {str(e)}")
        finally:
            self.is_listening = False
            self.mic_button.configure(state="normal", text="🎤 Mikrofonla Konuş")

    def analyze_text(self):
        text = self.input_text.get()
        if not text.strip():
            messagebox.showwarning("Uyarı", "Lütfen analiz edilecek metni girin")
            return

        self.status_var.set("Metin analiz ediliyor...")
        try:
            prediction = predict_hate_speech(text, self.hate_model, tokenizer, device)
            self.output_text.insert("end", f"📝 Metin: {text}\n")
            self.output_text.insert("end", f"🔍 Sonuç: {prediction}\n\n")
            self.output_text.see("end")
            self.status_var.set("Analiz tamamlandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz sırasında hata: {str(e)}")
            self.status_var.set("Hata oluştu")

    def analyze_file(self):
        file_path = filedialog.askopenfilename(
            title="Metin dosyası seçin", filetypes=(("Metin Dosyaları", "*.txt"),)
        )
        if not file_path:
            return

        self.status_var.set("Dosya okunuyor...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                messagebox.showwarning("Uyarı", "Dosya boş")
                return

            self.input_text.delete(0, "end")
            self.input_text.insert(0, text[:500] + "..." if len(text) > 500 else text)

            prediction = predict_hate_speech(text, self.hate_model, tokenizer, device)
            self.output_text.insert("end", f"📂 Dosya: {Path(file_path).name}\n")
            self.output_text.insert(
                "end", f"📝 İçerik (kısaltılmış): {text[:300]}...\n"
            )
            self.output_text.insert("end", f"🔍 Sonuç: {prediction}\n\n")
            self.output_text.see("end")
            self.status_var.set("Dosya analizi tamamlandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya okunamadı: {str(e)}")
            self.status_var.set("Dosya okuma hatası")

    def analyze_audio(self):
        file_path = filedialog.askopenfilename(
            title="Ses dosyası seçin", filetypes=[("Ses Dosyaları", "*.mp3 *.wav")]
        )
        if not file_path:
            return

        self.status_var.set("Ses dosyası işleniyor...")
        try:
            result = whisper_model.transcribe(file_path, language="Turkish")
            text = result["text"]

            self.input_text.delete(0, "end")
            self.input_text.insert(0, text)

            prediction = predict_hate_speech(text, self.hate_model, tokenizer, device)
            self.output_text.insert("end", f"🔊 Ses Dosyası: {Path(file_path).name}\n")
            self.output_text.insert("end", f"📝 Çıkarılan Metin: {text}\n")
            self.output_text.insert(
                "end", f"🔍 Nefret Söylemi Analizi: {prediction}\n\n"
            )
            self.output_text.see("end")
            self.status_var.set("Ses analizi tamamlandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Ses analizi başarısız: {str(e)}")
            self.status_var.set("Ses analiz hatası")

    def analyze_emotion(self):
        if self.emotion_model is None or self.scaler is None:
            messagebox.showerror("Hata", "Model veya scaler yüklenemedi!")
            return

        file_path = filedialog.askopenfilename(filetypes=[("WAV Dosyaları", "*.wav")])
        if not file_path:
            return

        try:
            self.status_var.set("Duygu analizi için ses özellikleri çıkarılıyor...")

            # Eğitimdeki get_predict_feat fonksiyonuna benzer şekilde ses yükle
            # Eğitimde duration=2.5, offset=0.6 kullanıldığı için burada da kullanıyoruz.
            # sr=22050 varsayılan olarak korunuyor.
            y, sr = librosa.load(
                file_path, sr=22050, mono=True, duration=2.5, offset=0.6
            )

            # Eğitimdeki extract_features fonksiyonuna benzer şekilde tüm özellikleri çıkar
            # Bu, ZCR, RMSE ve düzleştirilmiş (flattened) MFCC'leri içerir.
            # NOT: mfcc_feature fonksiyonu n_mfcc=20 varsayılanını kullanır,
            # çünkü eğitim kodunuzda bu belirtilmemişti.
            extracted_features = extract_all_features(y, sr)

            # Özellik vektörünü (1D) modelin beklediği boyuta getir (2376)
            # Eğitim kodunuzdaki 'result=np.reshape(result,newshape=(1,2376))' adımına karşılık gelir.
            # Eğer çıkarılan özellik vektörünün uzunluğu 2376'dan farklıysa,
            # bu kısım hata verebilir veya yanlış sonuçlar üretebilir.
            # Eğitimde tüm seslerin 2376 uzunluğunda özellik vektörü ürettiği varsayılıyor.
            if extracted_features.shape[0] != self.emotion_model_timesteps:
                # Eğer özellik uzunluğu 2376 değilse, dolgu veya kırpma yap
                if extracted_features.shape[0] < self.emotion_model_timesteps:
                    processed_features = np.pad(
                        extracted_features,
                        (0, self.emotion_model_timesteps - extracted_features.shape[0]),
                        mode="constant",
                    )
                else:
                    processed_features = extracted_features[
                        : self.emotion_model_timesteps
                    ]
                # print(f"Uyarı: Çıkarılan özellik uzunluğu {extracted_features.shape[0]}, 2376'ya ayarlandı.")
            else:
                processed_features = extracted_features

            # Scaler için (1, 2376) şekline getir
            # Eğitimdeki scaler'ın 1D vektörleri beklediği varsayılıyor.
            features_for_scaler = processed_features.reshape(1, -1)  # Shape: (1, 2376)

            # Normalleştirme
            scaled_features = self.scaler.transform(
                features_for_scaler
            )  # Shape: (1, 2376)

            # Model için (1, 2376, 1) şekline getir
            # Eğitimdeki 'final_result=np.expand_dims(i_result, axis=2)' adımına karşılık gelir.
            final_input = np.expand_dims(scaled_features, axis=2)  # Shape: (1, 2376, 1)

            # Tahmin yap
            prediction = self.emotion_model.predict(final_input)

            # --- HATA AYIKLAMA ÇIKTILARI ---
            print(f"Raw prediction probabilities: {prediction}")
            predicted_label = np.argmax(prediction, axis=1)[0]
            print(f"Predicted label index: {predicted_label}")
            # --- HATA AYIKLAMA ÇIKTILARI SONU ---

            emotion = label_map.get(predicted_label, "Bilinmeyen")

            # Sonuçları göster
            self.output_text.insert("end", f"🎵 Ses Dosyası: {Path(file_path).name}\n")
            self.output_text.insert("end", f"😊 Duygu Tahmini: {emotion}\n")
            self.output_text.insert("end", f"🔢 Olasılıklar: {prediction}\n\n")
            self.output_text.see("end")
            self.status_var.set("Duygu analizi tamamlandı")

        except Exception as e:
            messagebox.showerror("Hata", f"Duygu analizi başarısız: {str(e)}")
            self.status_var.set("Duygu analiz hatası")

    def clear_output(self):
        self.output_text.delete("1.0", "end")
        self.status_var.set("Çıktı temizlendi")


# --- Model Fonksiyonları ---
def load_hate_model(model_path, num_labels, device):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, hidden_dropout_prob=0.3
    )
    model.to(device)
    # PyTorch'un FutureWarning'ını göz önünde bulundurarak weights_only=True ekleyebilirsiniz
    # Ancak eski bir model yüklüyorsanız ve bu hata veriyorsa, kaldırabilirsiniz.
    # Güvenlik için weights_only=True önerilir.
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    model.eval()
    return model


def predict_hate_speech(text, model, tokenizer, device, max_len=128):
    cleaned_text = text.strip()
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
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    return "🚫 Nefret Söylemi" if predicted_class == 1 else "✅ Nefret Söylemi Değil"


if __name__ == "__main__":
    app = App()
    app.mainloop()
