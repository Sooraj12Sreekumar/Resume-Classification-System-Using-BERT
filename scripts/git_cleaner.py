import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GITHUB_RESUMES_DIR = (BASE_DIR / "../data/raw/github_resumes").resolve()
JUNK_DIR = (BASE_DIR / "../data/raw/junk_data").resolve()

IMPORTANT_KEYWORDS = [
    'skills', 'experience', 'project', 'education',
    'work', 'summary', 'certification', 'achievement'
]

def is_junk(content):
    if len(content.strip()) < 100:
        return True

    content_lower = content.lower()

    if not any(keyword in content_lower for keyword in IMPORTANT_KEYWORDS):
        return True

    alpha_chars = sum(c.isalpha() for c in content)
    if alpha_chars / max(len(content), 1) < 0.5:
        return True

    return False


def clean_github_resumes():
    for filename in os.listdir(GITHUB_RESUMES_DIR):
        file_path = os.path.join(GITHUB_RESUMES_DIR, filename)

        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if is_junk(content):
                    print(f"JUNK → {filename}")
                    shutil.move(file_path, os.path.join(JUNK_DIR, filename))
                else:
                    print(f"{filename}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                shutil.move(file_path, os.path.join(JUNK_DIR, filename))


if __name__ == "__main__":
    clean_github_resumes()
