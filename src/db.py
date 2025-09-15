import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env file
load_dotenv()

PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DB   = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")

def get_engine():
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url)

# ✅ NEW helper function to get a raw connection for pandas
def get_connection():
    engine = get_engine()
    return engine.connect()

if __name__ == "__main__":
    with get_engine().connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM raw_data;"))
        print("Row count in raw_data:", result.scalar())
print(f"Connecting to DB: {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB}")
