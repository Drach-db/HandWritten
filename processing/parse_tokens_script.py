import json
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Цены за 1000 токенов по моделям
MODEL_COSTS = {
    "gpt-4o-2024-11-20": {"prompt": 0.005, "completion": 0.015},
    "gpt-4.5-preview-2025-02-27": {"prompt": 0.0005, "completion": 0.0015}
}

def parse_tokens(stage, jsonl_path):
    records = []
    if not os.path.exists(jsonl_path):
        logger.warning(f"Файл не найден: {jsonl_path}")
        return records

    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                custom_id = data.get("custom_id", "")
                parts = custom_id.split("_")

                if stage == 1:
                    page_number = int(parts[1])  # stage1_{page}_{chunk}_v{variant}
                else:
                    page_number = int(parts[-1])  # stage{stage}_page_{page}

                usage = data["response"]["body"]["usage"]
                model = data["response"]["body"].get("model", "unknown")

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                prompt_cost = prompt_tokens / 1000 * MODEL_COSTS.get(model, {}).get("prompt", 0)
                completion_cost = completion_tokens / 1000 * MODEL_COSTS.get(model, {}).get("completion", 0)
                total_cost = prompt_cost + completion_cost

                records.append({
                    "stage": stage,
                    "page_number": page_number,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "prompt_cost_usd": prompt_cost,
                    "completion_cost_usd": completion_cost,
                    "total_cost_usd": total_cost
                })

            except Exception as e:
                logger.error(f"Ошибка обработки строки: {line}. Ошибка: {e}")
    return records

def main():
    stages = [
        (1, "cache/results/stage1_results.jsonl"),
        (2, "cache/results/stage2_results.jsonl"),
        (3, "cache/results/stage3_results.jsonl")
    ]

    all_records = []

    for stage, path in stages:
        logger.info(f"Парсинг токенов для stage {stage} из файла: {path}")
        records = parse_tokens(stage, path)
        all_records.extend(records)

    if all_records:
        df = pd.DataFrame(all_records)
        df.sort_values(by=["stage", "page_number"], inplace=True)
        df.to_csv("cache/token_usage_summary.csv", index=False, encoding="utf-8")
        logger.info("Результаты успешно сохранены в cache/token_usage_summary.csv")
    else:
        logger.warning("Нет данных для сохранения.")

if __name__ == "__main__":
    main()