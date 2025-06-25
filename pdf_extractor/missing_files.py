import os
import psycopg2 
pdf_folder=r"C:\Users\Contract.DESKTOP-AF4IHN4\Desktop\pdf_extractor\pdf"
all_pdf_names=[os.path.splitext(f)[0] for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
conn= psycopg2.connect(
    dbname="my_project",
    user="postgres",
    password="ittil@123",
    host="localhost"
)
cur=conn.cursor()
cur.execute("SELECT name FROM  extracted_projects")
inserted_names=set(row[0] for row in cur.fetchall())

missing_pdfs=[name for name in all_pdf_names if name not in inserted_names]

print(f"Missing ({len(missing_pdfs)}):")
for name in missing_pdfs:
    print(name)

cur.close()
conn.close()