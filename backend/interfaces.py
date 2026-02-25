from abc import ABC, abstractmethod
from typing import Any, Dict, List


class GraphDatabaseInterface(ABC):
    """
    Abstract Base Class for Graph Database operations following SOLID principles.
    """

    @abstractmethod
    def insert_relationship_graph(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> None:
        """
        Inserts a structured JSON graph of entities and relationships into the database.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Closes the database connection.
        """
        pass
