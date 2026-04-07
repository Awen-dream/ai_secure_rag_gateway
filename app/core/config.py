from pydantic import BaseModel


class AppSettings(BaseModel):
    app_name: str = "Secure Enterprise RAG Gateway"
    app_version: str = "0.2.0"
    api_prefix: str = "/api/v1"


settings = AppSettings()
