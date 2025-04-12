import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_stage3_results(jsonl_path: str = "cache/results/stage3_results.jsonl", csv_output_path: str = "cache/parsed_stage3_results.csv"):
    """
    Парсит результаты третьего этапа из JSONL в CSV.

    Args:
        jsonl_path (str): Путь к JSONL-файлу с результатами.
        csv_output_path (str): Путь к файлу CSV для сохранения результатов.
    """

    records = []

    logger.info(f"Начинаем парсинг файла: {jsonl_path}")

    # Чтение и обработка JSONL файла
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                custom_id = data.get("custom_id", "")
                page_number = int(custom_id.split("_")[-1])

                verified_text = json.loads(data["response"]["body"]["choices"][0]["message"]["content"]).get("verified_text", "")

                records.append({
                    "page_number": page_number,
                    "verified_text": verified_text
                })

            except Exception as e:
                logger.error(f"Ошибка при обработке строки: {line}. Ошибка: {e}")

    # Сохранение результатов в CSV
    df = pd.DataFrame(records)
    df.sort_values(by="page_number", inplace=True)
    df.to_csv(csv_output_path, index=False, encoding="utf-8")

    logger.info(f"Результаты успешно сохранены в: {csv_output_path}")

if __name__ == "__main__":
    parse_stage3_results()
