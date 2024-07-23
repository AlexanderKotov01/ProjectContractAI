import time
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import OPENAI_API_KEY
from src.gpt_analysis import analyze_contract
from src.file_reader import read_docx, read_pdf

# Загрузка дообученной модели и токенизатора
model_path = 'models/fine-tuned-risk-model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Определение маппинга меток риска
label_mapping = {
    0: "Низкий",
    1: "Ниже среднего",
    2: "Средний",
    3: "Выше среднего",
    4: "Высокий"
}

# Функция для классификации уровня риска
def classify_risk_level(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    risk_level = label_mapping[predictions.item()]
    return risk_level

# Функция для извлечения ошибок из анализа
def extract_errors(analysis_result):
    errors = []
    current_error = None
    for line in analysis_result.split('\n'):
        if line.startswith("Недочет:") or line.startswith("Ошибка:"):
            if current_error:
                errors.append(current_error)
            current_error = {"description": line.split(":")[1].strip()}
        elif line.startswith("Уровень риска:"):
            if current_error:
                current_error["risk_level"] = line.split(":")[1].strip()
        elif line.startswith("Место ошибки:"):
            if current_error:
                current_error["position"] = line.split(":")[1].strip()
                errors.append(current_error)
                current_error = None
    if current_error:
        errors.append(current_error)
    return errors

# Функция для анализа и классификации ошибок
def analyze_and_classify_errors(text):
    analysis_result = analyze_contract(text)
    print("Мы проанализировали Ваш договор и нашли следующие недостатки:\n", analysis_result)
    errors = extract_errors(analysis_result)
    classified_errors = []
    for error in errors:
        if 'risk_level' not in error:  # Если уровень риска не указан, классифицируем его
            risk_level = classify_risk_level(error['description'])
            error['risk_level'] = risk_level
        classified_errors.append(error)
    return classified_errors

# Пример использования
file_path = 'data/45.docx'
if file_path.endswith('.docx'):
    contract_text = read_docx(file_path)
elif file_path.endswith('.pdf'):
    contract_text = read_pdf(file_path)
else:
    raise ValueError("Unsupported file format. Please provide a .docx or .pdf file.")

# Отображение прогресса анализа
start_time = time.time()
print("Начинаем анализ текста договора...")
progress_intervals = [20, 40, 60, 80, 100]
total_duration = 20  # total duration for progress simulation in seconds

for i, progress in enumerate(progress_intervals):
    elapsed_time = time.time() - start_time
    if elapsed_time >= total_duration:
        break
    remaining_time = total_duration - elapsed_time
    sleep_time = remaining_time / (len(progress_intervals) - i)
    print(f"Выполнено: {progress}%")
    time.sleep(sleep_time)

classified_errors = analyze_and_classify_errors(contract_text)
elapsed_time = time.time() - start_time

# Форматирование вывода

for idx, error in enumerate(classified_errors, 1):
    print(f"Ошибка {idx}: {error['description']}")
    print(f"Место ошибки: {error.get('position', 'Не указано')}")
    print(f"Уровень риска: {error['risk_level']}")
    print("")
print(f"\nАнализ завершен за {elapsed_time:.2f} секунд.")






















    
   