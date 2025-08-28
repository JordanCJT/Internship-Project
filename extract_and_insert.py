import re
import mysql.connector

# === Load text from CRAFT output ===
with open("top.txt", "r", encoding="utf-8") as f:
    top_text = f.read()
with open("left.txt", "r", encoding="utf-8") as f:
    left_text = f.read()
with open("right.txt", "r", encoding="utf-8") as f:
    right_text = f.read()

full_text = "\n".join([top_text, left_text, right_text])

# === Extract data using regex ===
name_match = re.search(r'(?i)(?:Name[:\-]?\s*)?([A-Z][a-z]+\s+[A-Z][a-z]+)', full_text)
email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_text)
phone_match = re.search(r'(?:\+?6?0?-?)?\d{2,3}[- ]?\d{7,8}', full_text)

name = name_match.group(1) if name_match else None
email = email_match.group(0) if email_match else None
phone = phone_match.group(0) if phone_match else None

print(f"Name: {name}\nEmail: {email}\nPhone: {phone}")

# === Insert into MySQL ===
conn = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
)
cursor = conn.cursor()

cursor.execute("""
    INSERT INTO resumes (name, email, phone)
    VALUES (%s, %s, %s)
""", (name, email, phone))

conn.commit()
cursor.close()
conn.close()

print("[INFO] Data inserted into MySQL successfully.")
