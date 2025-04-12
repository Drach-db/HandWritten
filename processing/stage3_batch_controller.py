import os
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден!")

client = OpenAI(api_key=OPENAI_API_KEY)

def run_batch_stage3(
    jsonl_path: str = "cache/batch_stage3.jsonl",
    result_path: str = "cache/results/stage3_results.jsonl",
    completion_window: str = "24h"
):
    """
    Отправляет JSONL файл батчем в OpenAI и сохраняет результаты третьего этапа.

    Args:
        jsonl_path (str): Путь к JSONL-файлу для отправки.
        result_path (str): Путь к файлу, в который сохранить результаты.
        completion_window (str): Окно времени выполнения батча.
    """

    logger.info("Запуск batch Stage 3")
    logger.info(f"JSONL path: {jsonl_path}")
    logger.info(f"Результаты будут сохранены в: {result_path}")

    # 1. Загрузка файла
    with open(jsonl_path, "rb") as f:
        upload_response = client.files.create(file=f, purpose="batch")
    file_id = upload_response.id
    logger.info(f"Файл загружен, file_id={file_id}")

    # 2. Создание батча
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window
    )
    batch_id = batch_response.id
    logger.info(f"Batch создан, batch_id={batch_id}")

    # 3. Проверка статуса батча (polling)
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status

        if status in ["completed", "failed", "cancelled", "expired"]:
            logger.info(f"Batch завершен со статусом: {status}")
            break

        logger.info(f"Статус batch: {status}. Проверка через 10 сек.")
        time.sleep(10)

    if status != "completed":
        logger.error(f"Batch завершился неуспешно: статус {status}")
        return

    # 4. Получение и сохранение результатов
    result_file_id = batch_status.output_file_id
    if not result_file_id:
        logger.error("Отсутствует output_file_id, нечего загружать.")
        return

    file_stream = client.files.content(result_file_id)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, "wb") as result_file:
        for chunk in file_stream.iter_bytes():
            result_file.write(chunk)

    logger.info(f"Результаты успешно сохранены в {result_path}")

if __name__ == "__main__":
    run_batch_stage3()
