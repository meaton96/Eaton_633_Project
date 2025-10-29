import re
from pathlib import Path

def alphabetize_citations():
    file_path = Path("citations.txt")
    if not file_path.exists():
        print("Error: citations.txt not found.")
        return

    text = file_path.read_text(encoding="utf-8").strip()
    entries = [e.strip() for e in re.split(r"\n\s*\n", text) if e.strip()]

    def sort_key(entry: str):
        match = re.match(r"([A-Za-z.\-'\s]+?),", entry)
        if match:
            return match.group(1).strip().lower()
        return entry.lower()

    entries.sort(key=sort_key)

    file_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    print("Sorted and saved back to citations.txt")

if __name__ == "__main__":
    alphabetize_citations()
