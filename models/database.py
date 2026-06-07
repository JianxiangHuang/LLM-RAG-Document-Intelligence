import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()
DATABASE_URL=os.getenv('DATABASE_URL')
if DATABASE_URL is None:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

engine=create_engine(DATABASE_URL)

SessionLocal=sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base=declarative_base()

def create_tables():
    Base.metadata.create_all(bind=engine)