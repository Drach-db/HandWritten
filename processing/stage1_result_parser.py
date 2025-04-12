import json
import csv
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def clean_content(content_str: str) -> str:
    """Удаляет маркеры кода JSON из строки."""
    cleaned = re.sub(r"^```json\s*", "", content_str)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

def extract_variant_from_custom_id(custom_id: str) -> int:
    """Извлекает номер варианта из custom_id (например, '_v2')."""
    match = re.search(r"_v(\d+)$", custom_id)
    return int(match.group(1)) if match else 1

def parse_message_content(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Парсит сообщение модели, извлекая JSON-данные.
    Возвращает словарь с результатами или None.
    """
    content = message.get("content") or message.get("function_call", {}).get("arguments")
    if not content:
        return None

    cleaned_content = clean_content(content)
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка JSON-парсинга: {e}")
        return None

def parse_stage1_results(
    input_path: Union[str, Path] = "cache/results/stage1_results.jsonl",
    output_path: Union[str, Path] = "cache/parsed_stage1_results.csv"
) -> None:
    """
    Парсит результаты из JSONL-файла после Batch API Stage 1 и сохраняет в CSV.

    Args:
        input_path (str | Path): путь к входному JSONL-файлу.
        output_path (str | Path): путь для сохранения CSV-файла.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Некорректный JSON: {e}")
                continue

            custom_id = item.get("custom_id", "")
            variant = extract_variant_from_custom_id(custom_id)

            if error := item.get("error"):
                logger.warning(f"[{custom_id}] Ошибка API: {error}")
                continue

            response_body = item.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if not choices:
                logger.warning(f"[{custom_id}] Нет данных в choices.")
                continue

            message_data = parse_message_content(choices[0]["message"])
            if message_data is None:
                logger.warning(f"[{custom_id}] Не удалось распарсить сообщение.")
                continue

            if isinstance(message_data, list):
                for variant_index, chunk_data in enumerate(message_data, start=1):
                    results.append({
                        "custom_id": custom_id,
                        "variant": variant_index,
                        "chunk_number": chunk_data.get("chunk_number"),
                        "recognized_text": chunk_data.get("text", "").strip()
                    })
            elif isinstance(message_data, dict):
                results.append({
                    "custom_id": custom_id,
                    "variant": variant,
                    "chunk_number": message_data.get("chunk_number"),
                    "recognized_text": message_data.get("text", "").strip()
                })
            else:
                logger.warning(f"[{custom_id}] Неизвестный формат данных: {type(message_data)}")

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "custom_id", "variant", "chunk_number", "recognized_text"
        ])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Парсинг завершен. CSV сохранён: {output_path}")

if __name__ == "__main__":
    parse_stage1_results()
