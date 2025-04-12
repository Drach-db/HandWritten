import json
import logging
from pathlib import Path
from typing import Union
from PIL import Image
from pdf_utils import get_img_uri

logger = logging.getLogger(__name__)

# Обновленный промпт для structured-response (убрал упоминание функции)
CHUNK_PROMPT = """
Действуй как точный распознаватель рукописного текста.
Тебе будут предоставлены фрагменты печатного текста и рукописных правок на нём.

Ответ строго в формате JSON:
{
    "text": "полный распознанный текст фрагмента (учитывая рукописные правки)",
    "chunk_number": номер фрагмента (целое число)
}

Извлекай весь текст и все правки. Не добавляй ничего лишнего.

"""

def generate_stage1_jsonl(
    fragments_dir: Union[str, Path] = "cache/stage1_fragments",
    output_jsonl: Union[str, Path] = "cache/batch_stage1.jsonl",
    model_name: str = "gpt-4.5-preview-2025-02-27"
) -> None:
    """
    Генерирует JSONL-файл с запросами для Batch API OpenAI.

    Для каждого JPG-фрагмента генерируются 3 запроса с уникальными custom_id,
    чтобы получить 3 версии распознавания текста. Используется response_format.

    Args:
        fragments_dir (str | Path): Директория с JPG-фрагментами.
        output_jsonl (str | Path): Путь для сохранения JSONL-файла.
        model_name (str): Название модели OpenAI для обработки.
    """
    fragments_dir = Path(fragments_dir)
    output_jsonl = Path(output_jsonl)
    jpg_files = sorted(fragments_dir.glob("*.jpg"))

    if not jpg_files:
        logger.warning(f"Не найдено JPG-фрагментов в папке: {fragments_dir}")
        return

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_requests = 0
    try:
        with output_jsonl.open("w", encoding="utf-8") as out_f:
            for file_path in jpg_files:
                base_name = file_path.stem

                with Image.open(file_path) as img:
                    data_url = get_img_uri(img)

                for variant in range(1, 4):
                    custom_id = f"{base_name}_v{variant}"
                    request_payload = {
    "custom_id": custom_id,
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4.5-preview-2025-02-27",  # <-- чётко указана используемая модель
        "temperature": 0,
        "top_p": 0.1,
        "messages": [
            {"role": "system", "content": CHUNK_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "parse_handwritten",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "chunk_number": {"type": "integer"}
                    },
                    "required": ["text", "chunk_number"],
                    "additionalProperties": False
                }
            }
        }
    }
}

                    out_f.write(json.dumps(request_payload, ensure_ascii=False) + "\n")
                    total_requests += 1

        logger.info(f"JSONL для Stage 1 успешно создан: {output_jsonl}")
        logger.info(f"Всего записано: {total_requests} запросов.")
    except Exception as e:
        logger.error(f"Ошибка при создании JSONL: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_stage1_jsonl()
