from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DatasetCatalog(Base):
    __tablename__ = "dataset_catalog"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    location = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, index=True)
    version = Column(Integer, index=True)
    environment = Column(String, index=True)
    artifact_path = Column(String)
    metrics = Column(Text)
    parameters = Column(Text)
    description = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    promotion_timestamp = Column(DateTime(timezone=True), nullable=True)
