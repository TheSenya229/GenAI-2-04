from transformers import pipeline
import sys


def read_text_file(filepath: str) -> str:
    """
    Читает текст из файла.

    Args:
        filepath (str): Путь к файлу.

    Returns:
        str: Текст из файла.

    Raises:
        IOError: При ошибках чтения файла.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError("Файл пустой")
        return text
    except Exception as e:
        raise IOError(f"Ошибка при чтении файла '{filepath}': {e}")


def write_text_file(filepath: str, text: str) -> None:
    """
    Записывает текст в файл.

    Args:
        filepath (str): Путь к файлу.
        text (str): Текст для записи.

    Raises:
        IOError: При ошибках записи в файл.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise IOError(f"Ошибка при записи в файл '{filepath}': {e}")


def create_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6"):
    """
    Создаёт pipeline для суммаризации.

    Args:
        model_name (str): Имя модели для суммаризации.

    Returns:
        transformers.Pipeline: Объект pipeline для суммаризации.
    """
    return pipeline('summarization', model=model_name)


def summarize_text(summarizer, text: str) -> str:
    """
    Выполняет суммаризацию текста с использованием дефолтных параметров модели.

    Args:
        summarizer: Pipeline для суммаризации.
        text (str): Исходный текст.

    Returns:
        str: Краткое резюме текста.
    """
    result = summarizer(text)
    return result[0]['summary_text']


def split_text_into_paragraphs(text: str) -> list[str]:
    """
    Разбивает текст на абзацы по двойным переносам строк.

    Args:
        text (str): Исходный текст.

    Returns:
        list[str]: Список абзацев.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


def limit_text_length(text: str, max_length: int) -> str:
    """
    Ограничивает длину текста до max_length символов, обрезая аккуратно по словам.

    Args:
        text (str): Текст для обрезки.
        max_length (int): Максимальная длина текста.

    Returns:
        str: Обрезанный текст.
    """
    if len(text) <= max_length:
        return text
    trimmed = text[:max_length]
    last_space = trimmed.rfind(' ')
    if last_space == -1:
        return trimmed
    return trimmed[:last_space]


def main():
    """
    Основная функция, реализующая процесс:
    1. Чтение исходного текста из файла.
    2. Однократная суммаризация всего текста.
    3. Разбиение текста на абзацы и суммаризация каждого.
    4. Повторная суммаризация объединённого резюме абзацев.
    5. Ограничение итогового резюме 30% длины исходного текста.
    6. Запись результатов в выходной файл.
    """
    input_path = "input.txt"
    output_path = "output.txt"

    try:
        text = read_text_file(input_path)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    summarizer = create_summarizer()

    try:
        summary_single = summarize_text(summarizer, text)
    except Exception as e:
        print(f"Ошибка при однократной суммаризации: {e}", file=sys.stderr)
        sys.exit(2)

    paragraphs = split_text_into_paragraphs(text)

    try:
        summaries = [summarize_text(summarizer, p) for p in paragraphs]
    except Exception as e:
        print(f"Ошибка при суммаризации абзацев: {e}", file=sys.stderr)
        sys.exit(3)

    combined_summary_text = " ".join(summaries)

    try:
        summary_recursive = summarize_text(summarizer, combined_summary_text)
    except Exception as e:
        print(f"Ошибка при повторной суммаризации: {e}", file=sys.stderr)
        sys.exit(4)

    max_length = int(len(text) * 0.3)
    summary_recursive_limited = limit_text_length(summary_recursive, max_length)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=== Однократная суммаризация всего текста ===\n")
            f.write(summary_single + "\n\n")
            f.write("=== Рекурсивная суммаризация, ограниченная 30% от исходного ===\n")
            f.write(summary_recursive_limited + "\n")
    except Exception as e:
        print(f"Ошибка при записи файла: {e}", file=sys.stderr)
        sys.exit(5)

    print(f"Результаты суммаризации успешно сохранены в '{output_path}'")


if __name__ == "__main__":
    main()
