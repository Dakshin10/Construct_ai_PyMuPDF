import fitz
import re

doc = fitz.open("20-055 - BUILDING-C Structural Dwg. Set.pdf")
bar_pattern = re.compile(r'^([A-Z]\d+)$')
samples = []

for page_num in range(min(4, doc.page_count)):
    page = doc[page_num]
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 0:
            all_texts = [s["text"].strip() for l in block["lines"] for s in l["spans"] if s["text"].strip()]
            for i, text in enumerate(all_texts):
                if bar_pattern.match(text):
                    context = all_texts[max(0, i-2):i+6]
                    samples.append({"bar_mark": text, "context": context})

seen = set()
for s in samples:
    key = s["bar_mark"]
    if key not in seen:
        seen.add(key)
        print("BAR:", s["bar_mark"], "| CONTEXT:", s["context"])
