import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
import re
import logging
from PyPDF2 import PdfReader
import docx
from itertools import islice

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Модель
model_name = "IlyaGusev/rut5_base_sum_gazeta"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
token_cache = {}

# Кэшируем данные 
def tokenize_cached(text):
    if text in token_cache:
        return token_cache[text]
    tokens = tokenizer.tokenize(text)
    token_cache[text] = tokens
    return tokens

# Разбиваем текст на параграфы (абзацы)
def split_into_paragraphs(text):
    paragraphs, buffer = [], []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            buffer.append(line)
        elif buffer:
            paragraphs.append(" ".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer))
    return paragraphs

# Сбор из параграфов окна текста
def split_into_windows(paragraphs, max_tokens=512):
    windows, current_window, current_len = [], [], 0
    for para in paragraphs:
        token_count = len(tokenize_cached(para))
        if token_count > max_tokens:
            words = para.split()
            chunk, chunk_len = [], 0
            for word in words:
                word_len = len(tokenize_cached(word))
                if chunk_len + word_len > max_tokens:
                    windows.append(" ".join(chunk))
                    chunk, chunk_len = [], 0
                chunk.append(word)
                chunk_len += word_len
            if chunk:
                windows.append(" ".join(chunk))
        else:
            if current_len + token_count <= max_tokens:
                current_window.append(para)
                current_len += token_count
            else:
                windows.append(" ".join(current_window))
                current_window, current_len = [para], token_count
    if current_window:
        windows.append(" ".join(current_window))
    return windows

# Генерация саммари
def summarize_text(text, max_length=250, min_length=100, num_beams=4):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    try:
        summary_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Ошибка генерации: {e}")
        return f"[Ошибка: {e}]"

# Удаляем из результата похожие по смыслу предложения 
def postprocess_summary(text: str, semantic_threshold: float = 0.85) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return text.strip()
    try:
        sbert = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
        embeddings = sbert.encode(sentences, convert_to_tensor=True)
        selected, used = [], set()
        for i in range(len(sentences)):
            if i in used:
                continue
            selected.append(sentences[i])
            sims = util.cos_sim(embeddings[i], embeddings[i+1:])
            for j, sim in enumerate(sims[0]):
                if sim.item() > semantic_threshold:
                    used.add(i + 1 + j)
        return " ".join(selected)
    except Exception as e:
        logging.warning(f"SBERT отключена: {e}")
        return " ".join(sentences)

# Разбиение текста на слова (токены)
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Список из последовательности токенов
def get_ngrams(tokens, n):
    return list(zip(*(islice(tokens, i, None) for i in range(n))))

# Фильтр предложений итогового саммари 
def factual_filter(summary: str, original_text: str, n: int = 4) -> str:
    orig_tokens = simple_tokenize(original_text)
    orig_ngrams = set(get_ngrams(orig_tokens, n))

    result = []
    for sent in re.split(r'(?<=[.!?])\s+', summary):
        sent_tokens = simple_tokenize(sent)
        sent_ngrams = get_ngrams(sent_tokens, n)
        if any(gram in orig_ngrams for gram in sent_ngrams):
            result.append(sent.strip())
    return " ".join(result)

def full_summarization_pipeline(raw_text, max_length=250, min_length=100, num_beams=4):
    paragraphs = split_into_paragraphs(raw_text)
    windows = split_into_windows(paragraphs)
    summaries = [summarize_text(chunk, max_length, min_length, num_beams) for chunk in windows]
    merged = " ".join(summaries)
    cleaned = postprocess_summary(merged)
    factual = factual_filter(cleaned, raw_text)
    return factual

# Извлечение текста из ПДФ файла
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# Извлечение текста из докс файла
def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

# Интерфейс Streamlit
st.title("Генератор саммари на русском ")

text_input = st.text_area("Введите текст", height=300, placeholder="Или загрузите файл ниже")
uploaded_file = st.file_uploader("Загрузите файл", type=["txt", "pdf", "docx"])

if uploaded_file and text_input.strip():
    st.error("Невозможно одновременно использовать текст и файл, выберите что-то одно")
    st.stop()

# Получение текста
text = ""
if uploaded_file:
    try:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Данный тип файла не поддерживается")
            text = ""
        st.text_area("Извлечённый текст", value=text, height=300, disabled=True)
    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
else:
    text = text_input.strip()

filename = "summary_results.txt"
# Кнопка для генерации
if st.button("Сделать саммари"):
    if not text:
        st.warning("Введите текст или загрузите файл")
    else:
        with st.spinner("Саммари генерируется..."):
            summary = full_summarization_pipeline(text)
            st.text_area("Результат саммари", value=summary, height=300, disabled=True)

            # Подсчёт количества уже записанных саммари
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.startswith("Саммари"))
            except FileNotFoundError:
                count = 0

            count += 1
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"Саммари {count}:\n{summary}\n\n")

            st.success(f"Саммари №{count} успешно записано в файл {filename}")


