# Eflatun - Psikolog Asistanı

Bu proje, "Eflatun" adında bir psikolog asistanını oluşturur. Kullanıcıların sorularını yanıtlamak ve profesyonel psikolojik destek sağlamak için tasarlanmıştır. Google Generative AI ve LangChain gibi araçlar kullanılarak geliştirilmiştir.

## Gereksinimler

- Python 3.8 veya üstü
- Flask
- Flask-CORS
- PyPDF2
- NLTK
- LangChain ve ilgili modüller
- Google Generative AI

## Kurulum

1. Bu projeyi klonlayın:

    ```bash
    git clone https://github.com/kullanici-adiniz/eflatun.git
    cd eflatun
    ```

2. Gerekli bağımlılıkları yükleyin:

    ```bash
    pip install -r requirements.txt
    ```
    
3. Ortam değişkenlerini ayarlamak için `.env` dosyası oluşturun:

    `.env` dosyasında, aşağıdaki değişkeni ekleyin ve kendi Google API anahtarınızla değiştirin:

    ```plaintext
    GOOGLE_API_KEY=your_google_api_key_here
    ```

6. Uygulamayı çalıştırın:

    ```bash
    python app.py
    ```

7. Tarayıcınızda `http://localhost:3000` adresine gidin ve uygulamayı kullanmaya başlayın.

## Kullanım

- Ana sayfada, kullanıcı sorusunu yazabilir ve yanıt almak için gönder tuşuna basabilir.
