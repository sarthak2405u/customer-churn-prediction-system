import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

# Database credentials

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Create database URL

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME
)

# Create SQLAlchemy engine

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# Test database connection

def test_connection():

    try:

        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT version();")
            )

            print("PostgreSQL Connected Successfully!")

            print(result.scalar())

    except Exception as e:

        print("Database Connection Failed!")

        print(e)