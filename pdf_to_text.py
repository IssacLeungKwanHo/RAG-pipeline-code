import fitz

def pdf_to_txt(pdf_name):
    pdf_path = f"/Users/issacleung/Desktop/RAGpipline/books/{pdf_name}.pdf"
    txt_path = f"/Users/issacleung/Desktop/RAGpipline/books/{pdf_name}.txt"

    doc = fitz.open(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        for page in doc:
            text = page.get_text()
            txt_file.write(text + "\n")


for file_name in ["clientcenter", "adhd", "person", "musictherapy"]:
    pdf_to_txt(file_name)

print("done")