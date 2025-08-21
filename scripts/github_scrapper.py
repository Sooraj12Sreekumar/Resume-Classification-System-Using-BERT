import os
import requests
import time
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.text-match+json"
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUME_DIR = os.path.join(PROJECT_ROOT,"data", "raw", "github_resumes")
os.makedirs(RESUME_DIR, exist_ok=True)

VALID_EXTENSIONS = ('.pdf', '.txt', '.doc', '.docx')

existing_files = set(os.listdir(RESUME_DIR))

keywords = [
    "data scientist resume", "data analyst cv", "machine learning resume",
    "deep learning cv", "AI engineer resume", "NLP data scientist resume",
    "bioinformatics data analyst cv", "data science cv", "Genai cv",
    "AI enginner cv", "ml engineer cv"
]

BASE_URL = "https://api.github.com/search/code"
MAX_PAGES = 10
RATE_LIMIT_SLEEP = 60

session = requests.Session()

def fetch_and_save_resumes():
    downloaded_count = 0
    for keyword in keywords:
        print(f"\nSearching: {keyword}")
        for page in range(1, MAX_PAGES + 1):
            query = quote(f'{keyword} in:path extension:pdf OR extension:txt OR extension:doc OR extension:docx')
            url = f"{BASE_URL}?q={query}&page={page}&per_page=30"
            try:
                response = session.get(url, headers=HEADERS, timeout=20)
                if response.status_code == 403:
                    print("Rate limit hit. Sleeping...")
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue
                elif response.status_code != 200:
                    print(f"Failed ({response.status_code}): {url}")
                    continue

                results = response.json().get("items", [])
                if not results:
                    break

                for item in results:
                    file_name = item['name']
                    if not file_name.lower().endswith(VALID_EXTENSIONS):
                        continue

                    repo_path = item['repository']['full_name'].replace('/', '_')
                    file_key = f"{repo_path}_{file_name}"

                    if file_key in existing_files:
                        continue

                    raw_url = item["html_url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

                    try:
                        r = session.get(raw_url, headers=HEADERS, timeout=15)
                        if r.status_code == 200:
                            file_path = os.path.join(RESUME_DIR, file_key)
                            with open(file_path, "wb") as f:
                                f.write(r.content)
                            print(f"Saved: {file_key}")
                            existing_files.add(file_key)
                            downloaded_count += 1
                        else:
                            print(f"Cannot fetch file: {raw_url}")
                    except Exception as e:
                        print(f"Error downloading: {raw_url} | {e}")

            except Exception as e:
                print(f"Network error: {e}")
                continue

    print(f"\nDownloaded {downloaded_count} new resumes.")

if __name__ == "__main__":
    fetch_and_save_resumes()