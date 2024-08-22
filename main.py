from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import Document
import PyPDF2
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")

# NLTK indirme işlemleri (durdurma kelimeleri ve lemmatizer için)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

# Load environment variables
load_dotenv()

# Google API key için yapılandırma
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Konuşma geçmişini saklamak için veri yapısı
conversation_history = []

# PDF dosyasından metin çıkarma fonksiyonu
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Ön işleme için yardımcı fonksiyon
def preprocess_question(question):
    question = question.lower()
    question = re.sub(r"[^\w\s]", "", question)
    stop_words = set(stopwords.words("turkish"))  # Türkçe durdurma kelimeleri
    tokens = word_tokenize(question)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_question = " ".join(lemmatized_tokens)
    return preprocessed_question

# Dokümanları ön işleme tabi tutma fonksiyonu
def preprocess_documents(docs):
    preprocessed_docs = []
    for doc in docs:
        raw_text = doc.page_content
        preprocessed_text = preprocess_question(raw_text)
        preprocessed_doc = Document(page_content=preprocessed_text, metadata=doc.metadata)
        preprocessed_docs.append(preprocessed_doc)
    return preprocessed_docs

# Metinleri parçalara ayırmak için fonksiyon
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

# Metin parçalarını FAISS ile vektör mağazasına dönüştürmek için fonksiyon
def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("Metin parçası listesi boş!")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Konuşma geçmişini saklamak için yardımcı fonksiyon
def store_conversation_history(question, answer):
    new_entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "question": question,
        "answer": answer,
    }
    conversation_history.append(new_entry)

# Kullanıcı girdisini işlemek için fonksiyon
def user_input(user_question, vector_store):
    preprocessed_question = preprocess_question(user_question)

    # Geçmiş konuşmaları içeren bir metin oluştur
    context_text = "\n".join(
        [f"{item['timestamp']} - Kullanıcı: {item['question']}\nEflatun: {item['answer']}" for item in
         conversation_history]
    )

    # Geçmiş konuşmaları da vektör aramalarına dahil et
    context_chunks = get_text_chunks(context_text)
    context_docs = [Document(page_content=chunk, metadata={}) for chunk in context_chunks]

    # PDF'den elde edilen metinlerle benzerlik araması
    docs_from_pdf = vector_store.similarity_search(preprocessed_question)
    preprocessed_docs = preprocess_documents(docs_from_pdf + context_docs)

    prompt_template = """
    Senin ismin Eflatun. Bir psikolog gibi davranacak ve danışanlara profesyonel psikolojik destek sağlayacaksın. 
    Sana verilen bağlam, John Sommers-Flanagan ve Rita Sommers-Flanagan tarafından yazılmış "Klinik Görüşme" adlı kitabın içeriğidir. 
    Bu içerik doğrultusunda, danışanlarla yapılan klinik görüşmelerde izlenecek adımları, etik kuralları, dinleme ve izleme becerilerini kullanarak yanıtlar ver. 

    Görevin:
    1. Danışanın anlattıklarını dikkatle dinlemek ve anladığını göstermek.
    2. Anlamadığın ya da emin olmadığın konularda açıklayıcı sorular sormak.
    3. Danışanın duygusal durumunu ve mental sağlığını değerlendirmek.
    4. Etik kurallara uygun hareket ederek danışana profesyonel önerilerde bulunmak.
    5. Özellikle intihar eğilimi gibi acil durumlarda gerekli müdahaleyi sağlamak (Numara veya internet adresi önerme).
    6. Zorlu danışanlarla çalışma, çokkültürlülük gibi özel durumları göz önünde bulundurmak.
    7. Danışanın ruh sağlığıyla ilgili yapıcı ve destekleyici geri bildirimler sunmak.
    8. Danışanı daha fazla konuşmaya teşvik etmek için açık davetler sunmak (Konuşmaya Açık Davet).
    9. Danışana uzun uzun öneriler verme, adım adım danışanın sorunu kendisinin bulmasını sağlamak.

    Context (bağlam): {context}

    Danışanın Soru veya Konusu (kullanıcı tarafından): {question}

    

    Psikolog Eflatun'un Yanıtı:
    """

    model = GoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": preprocessed_docs, "question": preprocessed_question, "context": context_text})
    answer = response.get("output_text", "Bir sorun oluştu, lütfen tekrar deneyin.")

    store_conversation_history(user_question, answer)

    return answer


app = Flask(__name__)
CORS(app)  # CORS'u burada etkinleştiriyoruz

# PDF dosyasının yolu ve vector store'ü başlat
file_path = "C:/Users/Eyüp/Desktop/veri.pdf"  # PDF dosyasının yolu
raw_text = extract_text_from_pdf(file_path)
text_chunks = get_text_chunks(raw_text)
vector_store = get_vector_store(text_chunks)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get('question', '')
    if user_question:
        answer = user_input(user_question, vector_store)
        return jsonify({"answer": answer, "author": "Eflatun"})
    return jsonify({"error": "Soru sağlanamadı."}), 400


if __name__ == "__main__":
    app.run(port=3000)
