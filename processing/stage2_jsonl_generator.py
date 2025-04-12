import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGE2_PROMPT = """
Ты — помощник по распознаванию рукописного текста. 
Тебе предоставлены три фрагмента страницы, каждый в трех вариантах распознавания.
Для каждого фрагмента выбери наиболее правильный вариант текста и выдай итоговый текст всей страницы с разделением по номерам фрагментов.
"""

def generate_stage2_jsonl(
    input_csv: Path = Path("cache/parsed_stage1_results.csv"),
    output_jsonl: Path = Path("cache/batch_stage2.jsonl"),
    model_name: str = "gpt-4o-2024-11-20"
):
    df = pd.read_csv(input_csv)

    if df.empty:
        logger.warning("CSV-файл пустой!")
        return

    # Четко извлекаем все нужные столбцы из custom_id
    df[['stage', 'page_number', 'fragment_number', 'variant']] = df['custom_id'].str.extract(
        r'stage(\d+)_(\d+)_(\d+)_v(\d+)'
    ).astype(int)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for page_num in sorted(df['page_number'].unique()):
            page_df = df[df['page_number'] == page_num]
            fragments_text = ""

            for fragment_num in sorted(page_df['fragment_number'].unique()):
                fragment_df = page_df[page_df['fragment_number'] == fragment_num]
                fragments_text += f"Фрагмент {fragment_num}:\n"
                for variant_num in sorted(fragment_df['variant'].unique()):
                    variant_text = fragment_df[fragment_df['variant'] == variant_num]['recognized_text'].values[0]
                    fragments_text += f"Вариант {variant_num}:\n{variant_text}\n\n"

            custom_id = f"stage2_page_{page_num}"
            request_payload = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "temperature": 0,
                    "top_p": 0.1,
                    "messages": [
                        {"role": "system", "content": STAGE2_PROMPT.strip()},
                        {"role": "user", "content": fragments_text.strip()}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "final_page_selection",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "fragments": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "chunk_number": {"type": "integer"},
                                                "final_text": {"type": "string"}
                                            },
                                            "required": ["chunk_number", "final_text"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": ["fragments"],
                                "additionalProperties": False
                            }
                        }
                    }
                }
            }
            out_f.write(json.dumps(request_payload, ensure_ascii=False) + "\n")

    logger.info(f"JSONL-файл для второго этапа успешно создан: {output_jsonl}")

if __name__ == "__main__":
    generate_stage2_jsonl()
