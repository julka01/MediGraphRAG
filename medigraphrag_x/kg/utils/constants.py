# Constants copied from llm-graph-builder
BUCKET_UPLOAD = 'llm-graph-builder-upload'
BUCKET_FAILED_FILE = 'llm-graph-builder-failed'
PROJECT_ID = 'llm-experiments-387609'

# Query constants
QUERY_TO_GET_CHUNKS = """
MATCH (d:Document {fileName: $filename})<-[:PART_OF]-(c:Chunk)
RETURN c.id AS id, c.text AS text, c.position AS position
ORDER BY c.position
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

# Processing constants
START_FROM_BEGINNING = "start_from_beginning"
DELETE_ENTITIES_AND_START_FROM_BEGINNING = "delete_entities_and_start_from_beginning"
START_FROM_LAST_PROCESSED_POSITION = "start_from_last_processed_position"

# Count queries
NODEREL_COUNT_QUERY_WITH_COMMUNITY = """
MATCH (d:Document)
WHERE d.fileName IS NOT NULL
OPTIONAL MATCH (d)<-[po:PART_OF]-(c:Chunk)
OPTIONAL MATCH (c)-[he:HAS_ENTITY]->(e:__Entity__)
OPTIONAL MATCH (c)-[sim:SIMILAR]->(c2:Chunk)
OPTIONAL MATCH (c)-[nc:NEXT_CHUNK]->(c3:Chunk)
OPTIONAL MATCH (e)-[ic:IN_COMMUNITY]->(comm:__Community__)
OPTIONAL MATCH (comm)-[pc1:PARENT_COMMUNITY]->(first_level:__Community__)
OPTIONAL MATCH (first_level)-[pc2:PARENT_COMMUNITY]->(second_level:__Community__)
OPTIONAL MATCH (second_level)-[pc3:PARENT_COMMUNITY]->(third_level:__Community__)
WITH
  d.fileName AS filename,
  count(DISTINCT c) AS chunkNodeCount,
  count(DISTINCT po) AS partOfRelCount,
  count(DISTINCT he) AS hasEntityRelCount,
  count(DISTINCT sim) AS similarRelCount,
  count(DISTINCT nc) AS nextChunkRelCount,
  count(DISTINCT e) AS entityNodeCount,
  collect(DISTINCT e) AS entities,
  count(DISTINCT comm) AS baseCommunityCount,
  count(DISTINCT first_level) AS firstlevelcommCount,
  count(DISTINCT second_level) AS secondlevelcommCount,
  count(DISTINCT third_level) AS thirdlevelcommCount,
  count(DISTINCT ic) AS inCommunityCount,
  count(DISTINCT pc1) AS parentCommunityRelCount1,
  count(DISTINCT pc2) AS parentCommunityRelCount2,
  count(DISTINCT pc3) AS parentCommunityRelCount3
WITH
  filename,
  chunkNodeCount,
  partOfRelCount + hasEntityRelCount + similarRelCount + nextChunkRelCount AS chunkRelCount,
  entityNodeCount,
  entities,
  baseCommunityCount + firstlevelcommCount + secondlevelcommCount + thirdlevelcommCount AS commCount,
  inCommunityCount + parentCommunityRelCount1 + parentCommunityRelCount2 + parentCommunityRelCount3 AS communityRelCount
CALL {
  WITH entities
  UNWIND entities AS e
  RETURN sum(COUNT { (e)-->(e2:__Entity__) WHERE e2 in entities }) AS entityEntityRelCount
}
RETURN
  filename,
  COALESCE(chunkNodeCount, 0) AS chunkNodeCount,
  COALESCE(chunkRelCount, 0) AS chunkRelCount,
  COALESCE(entityNodeCount, 0) AS entityNodeCount,
  COALESCE(entityEntityRelCount, 0) AS entityEntityRelCount,
  COALESCE(commCount, 0) AS communityNodeCount,
  COALESCE(communityRelCount, 0) AS communityRelCount
"""

NODEREL_COUNT_QUERY_WITHOUT_COMMUNITY = """
MATCH (d:Document)
WHERE d.fileName = $document_name
OPTIONAL MATCH (d)<-[po:PART_OF]-(c:Chunk)
OPTIONAL MATCH (c)-[he:HAS_ENTITY]->(e:__Entity__)
OPTIONAL MATCH (c)-[sim:SIMILAR]->(c2:Chunk)
OPTIONAL MATCH (c)-[nc:NEXT_CHUNK]->(c3:Chunk)
WITH
  d.fileName AS filename,
  count(DISTINCT c) AS chunkNodeCount,
  count(DISTINCT po) AS partOfRelCount,
  count(DISTINCT he) AS hasEntityRelCount,
  count(DISTINCT sim) AS similarRelCount,
  count(DISTINCT nc) AS nextChunkRelCount,
  count(DISTINCT e) AS entityNodeCount,
  collect(DISTINCT e) AS entities
WITH
  filename,
  chunkNodeCount,
  partOfRelCount + hasEntityRelCount + similarRelCount + nextChunkRelCount AS chunkRelCount,
  entityNodeCount,
  entities
CALL {
  WITH entities
  UNWIND entities AS e
  RETURN sum(COUNT { (e)-->(e2:__Entity__) WHERE e2 in entities }) AS entityEntityRelCount
}
RETURN
  filename,
  COALESCE(chunkNodeCount, 0) AS chunkNodeCount,
  COALESCE(chunkRelCount, 0) AS chunkRelCount,
  COALESCE(entityNodeCount, 0) AS entityNodeCount,
  COALESCE(entityEntityRelCount, 0) AS entityEntityRelCount
"""

# Maximum community levels
MAX_COMMUNITY_LEVELS = 3
