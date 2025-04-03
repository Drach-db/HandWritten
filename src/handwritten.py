import base64
import os
import asyncio
import logging
from io import BytesIO
from typing import List

from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from pydantic import BaseModel

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Константы
SEPARATOR = "=" * 40
MODEL = "gpt-4o-mini"
PMODEL = "gpt-4.5-preview-2025-02-27"

# Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Инициализация OpenAI клиента
from openai import AsyncOpenAI  # Импорт после загрузки переменных окружения
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Функции для обработки изображений
def convert_doc_to_images(path):
    """Конвертирует PDF-документ в список изображений."""
    return convert_from_path(path, fmt='jpeg')

def load_and_preprocess_image(image: Image.Image):
    """Разделяет изображение на три перекрывающихся сегмента."""
    width, height = image.size
    segment_height = int(height * 0.4)
    overlap = int(segment_height * 0.15)

    segments = [
        image.crop((0, 0, width, segment_height + overlap)),
        image.crop((0, segment_height - overlap, width, 2 * segment_height)),
        image.crop((0, 2 * segment_height - overlap, width, height))
    ]
    return segments

def get_img_uri(img):
    """Конвертирует изображение в base64-encoded data URI."""
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"

# Классы данных
class PageChunk(BaseModel):
    chunk_number: int
    text: str

class FoundName(BaseModel):
    name: str

class PageResult(BaseModel):
    text: str

# Запуск основного асинхронного кода
async def main():
    pdf_path = "raw/коррект рукопись мамы 5-24-removed-removed.pdf"
    images = convert_doc_to_images(pdf_path)

    img_uris = []
    for page_number, image in enumerate(images, start=1):
        segments = load_and_preprocess_image(image)
        full_page_img_uri = get_img_uri(image)
        for n, segment in enumerate(segments):
            img_uri = get_img_uri(segment)
            img_uris.append((page_number, full_page_img_uri, n, img_uri))

    # Подготовка данных для верификации
    ver_img_uris = []
    for page_number, image in enumerate(images, start=1):
        segments = load_and_preprocess_image_verify_chunks(image)
        for n, segment in enumerate(segments):
            img_uri = get_img_uri(segment)
            ver_img_uris.append((page_number, n, img_uri))

    # Обработка страниц
    total_results = []
    max_page_number = max(page_number for page_number, _, _, _ in img_uris)

    for page_num in range(1, max_page_number + 1):
        page_img_uris = [uri for uri in img_uris if uri[0] == page_num]
        page_result = await process_page(page_img_uris, ver_img_uris)
        total_results.append((page_num, page_result))

    # Вывод результатов
    for page_num, result in total_results:
        print(f"Страница {page_num}: \n{result.text}\n{SEPARATOR}")

# Функции обработки
async def process_page(page_img_uris, ver_img_uris):
    chunk_prompt = '''
Действуй, как точный распознаватель рукописного текста. Тебе будут предоставлены фрагменты печатного текста и рукописных правок на нём.

Пожалуйста, заполни следующую схему извлеченной информацией:

1. text: String - Распознанный текст фрагмента. Извлекай все доступные символы и даты. Сохраняй все знаки препинания и обрывы строк.
2. chunk_number: Integer - Номер фрагмента.

! На фрагменте могут быть рукописные правки или исправления на печатном тексте. Извлекай текст с учётом рукописных исправлений.

Пожалуйста, убедитесь, что все поля заполнены точно на основе предоставленного текста документа.
'''

    # Распознавание текста в фрагментах (получаем три варианта для каждого фрагмента)
    intermediate_results = []
    for _, _, chunk_number, img_uri in page_img_uris:
        tasks = [text_recognition(chunk_prompt, img_uri) for _ in range(3)]
        texts = await asyncio.gather(*tasks)
        intermediate_results.append(texts)

    # Составление страницы
    page_prompt = '''
Ты - помощник по распознаванию рукописного текста. Тебе предоставлены фрагменты текста, каждый из которых содержит часть страницы в трёх вариантах распознавания. Тебе нужно выбрать наиболее точные варианты и составить единый текст страницы. Сохраняй форматирование для лучшей читаемости.
'''

    for idx, result in enumerate(intermediate_results):
        if result:
            page_prompt += f"\nФрагмент {idx + 1}:\n"
            for variant_num, text in enumerate(result, 1):
                if text:
                    page_prompt += f"Вариант {variant_num}: {text.text}\n"
                else:
                    page_prompt += f"Вариант {variant_num}: Ошибка распознавания\n"

    full_page_img_uri = page_img_uris[0][1]
    composed_text = await page_composer(page_prompt)

    # Верификация текста
    current_ver_img_uris = [uri for uri in ver_img_uris if uri[0] == page_img_uris[0][0]]
    verified_text = await verify_text(composed_text.text, current_ver_img_uris)

    return verified_text

async def text_recognition(prompt, img_uri):
    try:
        logger.info(f"Обработка изображения: {img_uri[:100]}")
        response = await async_client.beta.chat.completions.parse(
            model=PMODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_uri}}]},
            ],
            response_format=PageChunk,
            top_p=0.1
        )
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения {img_uri[:100]}: {e}")
        return None

async def page_composer(prompt):
    try:
        logger.info("Составление текста страницы")
        response = await async_client.beta.chat.completions.parse(
            model=PMODEL,
            temperature=0,
            messages=[{"role": "system", "content": prompt}],
            response_format=PageResult,
            top_p=0.3
        )
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Ошибка при составлении страницы: {e}")
        return None

async def verify_text(composed_text, ver_img_uris):
    verification_prompt = f'''
Ты - ассистент-корректор рукописного текста. Тебе предоставлен составленный текст и оригинальные фрагменты изображений. Проверь точность текста и исправь ошибки.

Составленный текст:
{composed_text}

Пожалуйста, предоставь откорректированный текст.
'''

    try:
        messages = [{"role": "system", "content": verification_prompt}]
        for _, _, img_uri in ver_img_uris:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_uri}}]})

        response = await async_client.beta.chat.completions.parse(
            model=PMODEL,
            messages=messages,
            response_format=PageResult,
            temperature=0,
            top_p=0.1
        )
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Ошибка при верификации текста: {e}")
        return None

def load_and_preprocess_image_verify_chunks(image: Image.Image):
    """Разделяет изображение на шесть перекрывающихся сегментов для верификации."""
    width, height = image.size
    segment_height = int(height * 0.175)
    overlap = int(segment_height * 0.10)

    segments = [
        image.crop((0, i * segment_height - overlap if i > 0 else 0, width, (i + 1) * segment_height + overlap))
        for i in range(6)
    ]
    return segments

if __name__ == "__main__":
    asyncio.run(main())
