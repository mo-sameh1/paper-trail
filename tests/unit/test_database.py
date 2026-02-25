from unittest.mock import MagicMock, Mock, patch

from backend.database import Neo4jGraphDatabase


@patch("backend.database.GraphDatabase.driver")
def test_neo4j_graph_database_init_close(mock_driver):
    mock_instance_driver = Mock()
    mock_driver.return_value = mock_instance_driver

    db = Neo4jGraphDatabase("bolt://localhost", "user", "pass")
    mock_driver.assert_called_once_with("bolt://localhost", auth=("user", "pass"))

    db.close()
    mock_instance_driver.close.assert_called_once()


@patch("backend.database.GraphDatabase.driver")
def test_neo4j_graph_database_insert(mock_driver):
    mock_instance_driver = MagicMock()
    mock_session = MagicMock()
    mock_instance_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.return_value = mock_instance_driver

    db = Neo4jGraphDatabase("bolt://localhost", "user", "pass")
    entities = [{"id": "doc_1", "properties": {"type": "Document"}}]
    relationships = [
        {
            "source_id": "doc_1",
            "target_id": "org_cia",
            "type": "MENTIONS",
            "properties": {},
        }
    ]

    db.insert_relationship_graph(entities, relationships)

    mock_session.run.assert_called_once()
    args, kwargs = mock_session.run.call_args
    assert "MERGE (n:Entity {id: entity.id})" in args[0]
    assert kwargs["entities"] == entities
    assert kwargs["relationships"] == relationships
