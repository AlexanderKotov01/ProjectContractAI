# gpt_analysis.py
import openai
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def analyze_contract(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Вы полезный юридический ассистент."},
                {"role": "user", "content": f"Проанализируйте следующий текст договора и выявите возможные ошибки или недочеты, присвоив каждому недочету один из следующих уровней риска: Низкий, Ниже среднего, Средний, Выше среднего, Высокий. Также укажи, где конкретно содержится ошибка в тексте:\n\n{text}\n\nОшибки и недочеты:"}
            ],
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return f"Произошла ошибка при обращении к API OpenAI: {str(e)}"
