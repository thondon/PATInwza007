from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "/Users/adiluk/Library/Mobile Documents/com~apple~CloudDocs/LOC AI/Dev/myfastapi/users.db"  # หรือใช้ PostgreSQL ก็ได้ในอนาคต

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(LargeBinary)  # เก็บเป็น binary เพราะ bcrypt ให้ hash เป็น bytes

Base.metadata.create_all(bind=engine)

