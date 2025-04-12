import json
import csv
import os
import re


def parse_stage2_results(
    input_path: str = "cache/results/stage2_results.jsonl",
    output_path: str = "cache/parsed_stage2_results.csv"
):
    results = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)

            custom_id = data["custom_id"]

            # Извлекаем номер страницы из custom_id
            match_page = re.search(r'stage2_page_(\d+)', custom_id)
            page_number = int(match_page.group(1)) if match_page else None

            response_body = data["response"]["body"]["choices"][0]["message"]["content"]

            fragments_data = json.loads(response_body)["fragments"]

            for fragment in fragments_data:
                results.append({
                    "page_number": page_number,
                    "chunk_number": fragment["chunk_number"],
                    "final_text": fragment["final_text"]
                })

    # Сортировка по номеру страницы и фрагмента
    results.sort(key=lambda x: (x["page_number"], x["chunk_number"]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Запись результата в CSV-файл
    with open(output_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["page_number", "chunk_number", "final_text"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Итоговый CSV сохранён в {output_path}.")


if __name__ == "__main__":
    parse_stage2_results()
