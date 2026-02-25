from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """
    Configuration parameters strictly for the Neo4j Database connection.
    """

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_", env_file=".env", extra="ignore"
    )


db_config = DatabaseConfig()
