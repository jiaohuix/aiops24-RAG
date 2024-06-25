from typing import Iterable
import json
import jsonlines


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content

def write_jsonl(data, path):
    """
    Writes a list of dictionaries to a JSON Lines file.

    :param data: List of dictionaries to write.
    :param path: Path to the output JSON Lines file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


def save_answers(
    queries: Iterable, results: Iterable, path: str = "data/answers.jsonl"
):
    answers = []
    for query, result in zip(queries, results):
        answers.append(
            {"id": query["id"], "query": query["query"], "answer": result.text}
        )

    # use jsonlines to save the answers
    def write_jsonl(path, content):
        with jsonlines.open(path, "w") as json_file:
            json_file.write_all(content)

    # 保存答案到 data/answers.jsonl
    write_jsonl(path, answers)
