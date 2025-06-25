import psycopg2

try:
    conn = psycopg2.connect(
        dbname="my_project",
        user="postgres",
        password="ittil@123",
        host="localhost"
    )
    print("Connection successful!")

    # You can add queries here later

    conn.close()
    print("Connection closed.")

except Exception as e:
    print("Error connecting to database:", e)
