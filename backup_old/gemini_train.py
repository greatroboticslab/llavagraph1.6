# create environment using: pip install -U google-genai
# set API key :export GOOGLE_API_KEY='your_API_KEY'

import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types



BASE_DIR = Path("~/Desktop/train").expanduser()
OUTPUT_FILE = Path("~/Desktop/trainingData_Gemini_V1.json").expanduser()


QUESTIONS = [
    "Describe the repetition and density of the signal across the timeline.",
    "Identify the shape of the most prominent 'edges' or 'transitions'.",
    "State the vertical range (min/max) and the polarity (positive/negative) of the displacement."
]


# ===========================================

class GeminiSignalAnalyzer:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No API Key detected. Please execute in terminal: export GOOGLE_API_KEY='your_key'")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"

    def process_image(self, img_path):
        """Send image to Gemini and get answers to three questions"""
        print(f"Analyzing: {img_path.parent.name}/{img_path.name}")

        prompt = f"""
        Analyze this signal displacement graph. 
        Please provide a detailed technical answer for each of the following questions:
        1. {QUESTIONS[0]}
        2. {QUESTIONS[1]}
        3. {QUESTIONS[2]}

        Format your response as:
        A1: [Your answer]
        A2: [Your answer]
        A3: [Your answer]
        """

        try:
            with open(img_path, "rb") as f:
                img_data = f.read()

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_data, mime_type="image/jpeg")
                ]
            )
            return self._parse_answers(response.text)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            return None

    def _parse_answers(self, text):
        """Split Gemini's response into three separate answers"""
        import re
        # Use regex to find content marked by A1:, A2:, A3:
        a1 = re.search(r"A1:(.*?)(?=A2:|$)", text, re.DOTALL)
        a2 = re.search(r"A2:(.*?)(?=A3:|$)", text, re.DOTALL)
        a3 = re.search(r"A3:(.*?)$", text, re.DOTALL)

        return [
            a1.group(1).strip() if a1 else "Data missing",
            a2.group(1).strip() if a2 else "Data missing",
            a3.group(1).strip() if a3 else "Data missing"
        ]

    def start_work(self):
        final_data = []
        # Scan subfolders: noise, sine, square
        categories = ["noise", "sine", "square"]

        for cat in categories:
            cat_path = BASE_DIR / cat
            if not cat_path.exists():
                print(f"Skipping: Folder not found {cat_path}")
                continue

            image_files = list(cat_path.glob("*.jpg")) + list(cat_path.glob("*.png")) + list(cat_path.glob("*.jpeg"))

            for img_path in image_files:
                answers = self.process_image(img_path)
                if answers:
                    entry = {
                        "id": f"{cat}_{img_path.stem}",
                        "image": f"train/{cat}/{img_path.name}",
                        "conversations": [
                            {"from": "human", "value": QUESTIONS[0]},
                            {"from": "gpt", "value": answers[0]},
                            {"from": "human", "value": QUESTIONS[1]},
                            {"from": "gpt", "value": answers[1]},
                            {"from": "human", "value": QUESTIONS[2]},
                            {"from": "gpt", "value": answers[2]}
                        ]
                    }
                    final_data.append(entry)

                # Free API recommends no more than 15 requests per minute, 4 seconds sleep is safe
                time.sleep(4)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"\nSuccess! Generated dataset contains {len(final_data)} records.")
        print(f"File location: {OUTPUT_FILE}")


if __name__ == "__main__":
    analyzer = GeminiSignalAnalyzer()
    analyzer.start_work()
