import os
import base64
import logging
from io import BytesIO
from typing import List
from pathlib import Path

from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# -----------------------------
# Конвертация PDF в изображения (по страницам)
# -----------------------------
def convert_doc_to_images(pdf_path: str) -> List[Image.Image]:
    try:
        images = convert_from_path(pdf_path, fmt='jpeg')
        logger.info(f"Конвертировано {len(images)} страниц из PDF: {pdf_path}")
        return images
    except Exception as e:
        logger.error(f"Ошибка конвертации PDF в изображения: {e}")
        raise

# -----------------------------
# Нарезка на 2 фрагмента (этап 1)
# -----------------------------
def load_and_preprocess_image(image: Image.Image) -> List[Image.Image]:
    width, height = image.size
    segment_height = int(height / 2)
    overlap = int(segment_height * 0.15)

    segments = [
        image.crop((0, 0, width, segment_height + overlap)),
        image.crop((0, segment_height - overlap, width, height))
    ]
    logger.debug(f"Изображение нарезано на 2 фрагмента (размер страницы: {width}x{height})")
    return segments

# -----------------------------
# Нарезка на 6 фрагментов (этап 3, верификация)
# -----------------------------
def load_and_preprocess_image_verify_chunks(image: Image.Image) -> List[Image.Image]:
    width, height = image.size
    segment_height = int(height * 0.175)
    overlap = int(segment_height * 0.10)

    segments = []
    for i in range(6):
        top = max(i * segment_height - overlap, 0)
        bottom = min((i + 1) * segment_height + overlap, height)
        segments.append(image.crop((0, top, width, bottom)))

    logger.debug(f"Изображение нарезано на 6 фрагментов (размер страницы: {width}x{height})")
    return segments

# -----------------------------
# Кодировка изображения в base64 data URI
# -----------------------------
def get_img_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"

# -----------------------------
# Сохранение фрагментов на диск
# -----------------------------
def save_fragments_to_disk(
    fragments: List[Image.Image],
    output_dir: str,
    page_num: int,
    prefix: str = "page"
) -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, fragment in enumerate(fragments):
        filename = f"{prefix}_{page_num}_{i + 1}.jpg"
        filepath = Path(output_dir) / filename
        fragment.save(filepath, format="JPEG")
        saved_paths.append(str(filepath))
        logger.debug(f"Фрагмент сохранён: {filepath}")

    logger.info(f"Сохранено {len(saved_paths)} фрагментов страницы {page_num} в папку {output_dir}")
    return saved_paths

# -----------------------------
# Пример использования (если нужно выполнить отдельно)
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pdf_path = "raw/коррект рукопись мамы 5-24-removed-removed.pdf"
    stage1_dir = "cache/stage1_fragments"
    stage3_dir = "cache/stage3_fragments"

    if not Path(pdf_path).exists():
        logger.error(f"Файл не найден: {pdf_path}")
        exit(1)

    pages = convert_doc_to_images(pdf_path)

    for page_num, page_image in enumerate(pages, start=1):
        fragments_stage1 = load_and_preprocess_image(page_image)
        save_fragments_to_disk(fragments_stage1, stage1_dir, page_num, "stage1")

        fragments_stage3 = load_and_preprocess_image_verify_chunks(page_image)
        save_fragments_to_disk(fragments_stage3, stage3_dir, page_num, "stage3")
