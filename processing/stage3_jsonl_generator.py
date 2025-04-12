import pandas as pd
import json
import base64
from pathlib import Path

# Загрузка CSV с результатами второго этапа
stage2_results = pd.read_csv('cache/parsed_stage2_results.csv')

# Функция кодирования изображений в base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

# Список для хранения всех запросов
batch_requests = []

# Проходимся по каждой странице
for page_number in stage2_results['page_number'].unique():
    page_text = "\n".join(stage2_results[stage2_results['page_number'] == page_number]['final_text'])

    # Формируем контент запроса
    content_list = [
        {"type": "text", "text": page_text}
    ]

    # Добавляем 6 изображений для страницы
    for img_number in range(1, 7):
        img_path = Path(f'cache/stage3_fragments/stage3_{page_number}_{img_number}.jpg')
        img_base64 = encode_image_to_base64(img_path)
        content_list.append({
            "type": "image_url",
            "image_url": {"url": img_base64}
        })

    # Формируем тело запроса
    request_body = {
        "custom_id": f"stage3_page_{page_number}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.5-preview-2025-02-27",
            "temperature": 0,
            "top_p": 0.1,
            "messages": [
                {"role": "system", "content": "Ты — ассистент-корректор рукописного текста. Тебе предоставлен текст страницы и оригинальные 6 фрагментов изображений этой страницы. Проверь точность текста и исправь ошибки. Выдай итоговый откорректированный текст страницы."},
                {"role": "user", "content": content_list}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "verified_page_text",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "verified_text": {"type": "string"}
                        },
                        "required": ["verified_text"],
                        "additionalProperties": False
                    }
                }
            }
        }
    }

    batch_requests.append(request_body)

# Сохраняем запросы в JSONL-файл
output_jsonl = "cache/batch_stage3.jsonl"
with open(output_jsonl, 'w', encoding='utf-8') as file:
    for request in batch_requests:
        file.write(json.dumps(request, ensure_ascii=False) + '\n')

print(f"Batch запросы сформированы и сохранены в {output_jsonl}")
