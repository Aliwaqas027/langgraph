import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class AzureConfig:
    api_key: str = os.getenv("AZURE_API_KEY")
    api_version: str = os.getenv("AZURE_API_KEY_VERSION")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT")


@dataclass
class PineconeConfig:
    api_key: str = os.getenv("PINECONE_API_KEY")
    environment: str = os.getenv("PINECONE_ENV")
    index_name: str = os.getenv("PINECONE_INDEX")


@dataclass
class GoogleConfig:
    api_key: str = os.getenv("GOOGLE_API_KEY")
    cse_id: str = os.getenv("GOOGLE_CSE_ID")


# Global configs
azure_config = AzureConfig()
pinecone_config = PineconeConfig()
google_config = GoogleConfig()
