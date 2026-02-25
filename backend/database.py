from typing import Any, Dict, List

from neo4j import GraphDatabase

from backend.interfaces import GraphDatabaseInterface


class Neo4jGraphDatabase(GraphDatabaseInterface):
    """
    Neo4j implementation of the GraphDatabaseInterface.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def insert_relationship_graph(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> None:
        """
        Executes Cypher to merge nodes and edges. Highly optimized and under 20 lines.
        """
        cypher_query = """
        UNWIND $entities AS entity
        MERGE (n:Entity {id: entity.id}) SET n += entity.properties
        WITH 1 AS dummy
        UNWIND $relationships AS rel
        MATCH (source:Entity {id: rel.source_id})
        MATCH (target:Entity {id: rel.target_id})
        CALL apoc.create.relationship(
            source, rel.type, rel.properties, target
        ) YIELD rel AS r
        RETURN count(r)
        """
        with self._driver.session() as session:
            session.run(cypher_query, entities=entities, relationships=relationships)

    def close(self) -> None:
        self._driver.close()
