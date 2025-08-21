import os
import google.generativeai as genai
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent
output_dir = (BASE_DIR / "../data/raw/synthetic_data").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT = """Generate a synthetic resume for a NLP Engineer role.
It should be realistic and written in professional tone.
Include sections like:
- Name
- Summary
- Skills
- Experience (2-3 roles)
- Education
- Certifications
- Projects

Format it in plain text (TXT only). Do NOT include any Markdown formatting."""

resume_count = 100 
resume_label = "NLP Engineer Resume"
safe_label = resume_label.replace(" ","_")

for i in range(resume_count):
    file_path = output_dir / f"{safe_label}_{i+1}.txt"

    if file_path.exists():
        print(f"Skipping {resume_label} {i+1}. Already exists.")
        continue

    try:
        response = model.generate_content(PROMPT)
        content = response.text

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

        print(f"Saved {resume_label} {i+1}.txt")

        time.sleep(1.5)

    except Exception as e:
        print(f"Error generating resume_{i+1}: {str(e)}")
        if "quota" in str(e).lower():
            print("Possibly reached free quota. Stopping.")
            break

    

        