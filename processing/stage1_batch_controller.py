import logging
import time
from pathlib import Path
from typing import Union
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Загрузка переменных окружения (.env)
load_dotenv()

# Получение API-ключа
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

client = OpenAI(api_key=OPENAI_API_KEY)

def upload_file(jsonl_path: Path) -> str:
    """Загружает файл в OpenAI и возвращает file_id."""
    try:
        with jsonl_path.open("rb") as f:
            upload_resp = client.files.create(file=f, purpose="batch")
        logger.info(f"Файл загружен, file_id={upload_resp.id}")
        return upload_resp.id
    except OpenAIError as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        raise

def create_batch(file_id: str, completion_window: str = "24h") -> str:
    """Создаёт новый batch и возвращает его ID."""
    try:
        batch_resp = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window
        )
        logger.info(f"Batch создан, batch_id={batch_resp.id}")
        return batch_resp.id
    except OpenAIError as e:
        logger.error(f"Ошибка создания batch: {e}")
        raise

def poll_batch_status(batch_id: str, interval: int = 10, timeout: int = 3600) -> dict:
    """Ожидает завершения batch, периодически проверяя его статус."""
    start_time = time.time()
    while True:
        try:
            batch_status = client.batches.retrieve(batch_id)
        except OpenAIError as e:
            logger.error(f"Ошибка получения статуса batch: {e}")
            raise

        status = batch_status.status
        if status in ["completed", "failed", "cancelled", "expired"]:
            logger.info(f"Batch завершен со статусом: {status}")
            return batch_status

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"Превышен таймаут ожидания завершения batch: {timeout} сек.")

        logger.info(f"Статус batch: {status}. Проверка через {interval} сек.")
        time.sleep(interval)

def download_batch_result(result_file_id: str, result_path: Path) -> None:
    """Скачивает и сохраняет результаты выполнения batch."""
    try:
        file_stream = client.files.content(result_file_id)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("wb") as f:
            for chunk in file_stream.iter_bytes():
                f.write(chunk)
        logger.info(f"Результаты сохранены: {result_path}")
    except OpenAIError as e:
        logger.error(f"Ошибка скачивания результата batch: {e}")
        raise

def run_batch_stage1(
    jsonl_path: Union[str, Path] = "cache/batch_stage1.jsonl",
    result_path: Union[str, Path] = "cache/results/stage1_results.jsonl",
    completion_window: str = "24h"
):
    """
    Запускает полный цикл отправки и получения результатов batch-запросов OpenAI.

    Args:
        jsonl_path (str | Path): Путь к JSONL-файлу с запросами.
        result_path (str | Path): Путь сохранения результатов.
        completion_window (str): Максимальное время выполнения batch (например, "24h").
    """
    jsonl_path = Path(jsonl_path)
    result_path = Path(result_path)

    logger.info("Запуск batch Stage 1")
    logger.info(f"JSONL path: {jsonl_path}")
    logger.info(f"Результаты будут сохранены в: {result_path}")

    try:
        # Шаг 1: Загрузка файла
        file_id = upload_file(jsonl_path)

        # Шаг 2: Создание batch
        batch_id = create_batch(file_id, completion_window)

        # Шаг 3: Ожидание завершения batch
        batch_status = poll_batch_status(batch_id)

        if batch_status.status != "completed":
            logger.error(f"Batch завершен неудачно со статусом: {batch_status.status}")
            return

        # Шаг 4: Загрузка результата
        result_file_id = batch_status.output_file_id
        if not result_file_id:
            logger.error("Отсутствует output_file_id, нечего загружать.")
            return

        download_batch_result(result_file_id, result_path)
        logger.info("Batch успешно выполнен и обработан.")

    except Exception as e:
        logger.exception(f"Ошибка в процессе выполнения batch: {e}")

if __name__ == "__main__":
    run_batch_stage1()
