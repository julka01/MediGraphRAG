# Shared constants for KG creation and database operations

# Community settings (formerly from external module)
MAX_COMMUNITY_LEVELS = 3

# Chunk processing modes
START_FROM_BEGINNING = "start_from_beginning"
START_FROM_LAST_PROCESSED_POSITION = "start_from_last_processed_position"
DELETE_ENTITIES_AND_START_FROM_BEGINNING = "delete_entities_and_start_from_beginning"

# Neo4j chunk and entity queries
QUERY_TO_GET_CHUNKS = """
MATCH (d:Document)
WHERE d.fileName = $filename
WITH d
OPTIONAL MATCH (d)<-[:PART_OF|FIRST_CHUNK]-(c:Chunk)
RETURN c.id as id, c.text as text, c.position as position 
"""
QUERY_TO_DELETE_EXISTING_ENTITIES = """
MATCH (d:Document {fileName:$filename})
WITH d
MATCH (d)<-[:PART_OF]-(c:Chunk)
WITH d,c
MATCH (c)-[:HAS_ENTITY]->(e)
WHERE NOT EXISTS { (e)<-[:HAS_ENTITY]-()<-[:PART_OF]-(d2:Document) }
DETACH DELETE e
"""
QUERY_TO_GET_LAST_PROCESSED_CHUNK_POSITION = """
MATCH (d:Document)
WHERE d.fileName = $filename
WITH d
MATCH (c:Chunk) WHERE c.embedding is null 
RETURN c.id as id,c.position as position 
ORDER BY c.position LIMIT 1
"""
QUERY_TO_GET_LAST_PROCESSED_CHUNK_WITHOUT_ENTITY = """
MATCH (d:Document)
WHERE d.fileName = $filename
WITH d
MATCH (d)<-[:PART_OF]-(c:Chunk) WHERE NOT exists {(c)-[:HAS_ENTITY]->()}
RETURN c.id as id,c.position as position 
ORDER BY c.position LIMIT 1
"""
QUERY_TO_GET_NODES_AND_RELATIONS_OF_A_DOCUMENT = """
MATCH (d:Document)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY]->(e) where d.fileName=$filename
OPTIONAL MATCH (d)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY]->(e2:!Chunk)-[rel]-(e)
RETURN count(DISTINCT e) as nodes, count(DISTINCT rel) as rels
"""

# GCS upload bucket for document sources
BUCKET_UPLOAD = 'llm-graph-builder-upload'

# Node/relationship count queries
NODEREL_COUNT_QUERY_WITH_COMMUNITY = """
MATCH (d:Document)
WHERE d.fileName IS NOT NULL
OPTIONAL MATCH (d)<-[po:PART_OF]-(c:Chunk)
OPTIONAL MATCH (c)-[he:HAS_ENTITY]->(e:__Entity__)
CALL {
  WITH d, c, e
  MATCH (comm:__Community__)
  RETURN COUNT(comm) AS communityCount
}
WITH
  d.fileName AS filename,
  count(DISTINCT c) AS chunkNodeCount,
  count(DISTINCT he) AS entityNodeCount,
  COLLECT(DISTINCT e) AS entities,
  communityCount AS communityNodeCount
CALL {
  WITH entities
  UNWIND entities AS e
  RETURN sum(size((e)-->() )) AS entityEntityRelCount
}
RETURN
  filename,
  COALESCE(chunkNodeCount, 0) AS chunkNodeCount,
  COALESCE(entityNodeCount, 0) AS entityNodeCount,
  COALESCE(entityEntityRelCount, 0) AS entityEntityRelCount,
  COALESCE(communityNodeCount, 0) AS communityNodeCount
"""

NODEREL_COUNT_QUERY_WITHOUT_COMMUNITY = """
MATCH (d:Document)
WHERE d.fileName = $document_name
OPTIONAL MATCH (d)<-[po:PART_OF]-(c:Chunk)
OPTIONAL MATCH (c)-[he:HAS_ENTITY]->(e:__Entity__)
WITH
  d.fileName AS filename,
  count(DISTINCT c) AS chunkNodeCount,
  count(DISTINCT he) AS entityNodeCount,
  COLLECT(DISTINCT e) AS entities
CALL {
  WITH entities
  UNWIND entities AS e
  RETURN sum(size((e)-->() )) AS entityEntityRelCount
}
RETURN
  filename,
  COALESCE(chunkNodeCount, 0) AS chunkNodeCount,
  COALESCE(entityNodeCount, 0) AS entityNodeCount,
  COALESCE(entityEntityRelCount, 0) AS entityEntityRelCount
"""
