import logging
import os
import time
from neo4j.exceptions import TransientError
from langchain_neo4j import Neo4jGraph
import sys
import os
from typing import Dict, List
# Import from local kg_utils
from ontographrag.kg.utils.common_functions import create_gcs_bucket_folder_name_hashed, delete_uploaded_local_file, load_embedding_model
from ontographrag.kg.utils.constants import BUCKET_UPLOAD, NODEREL_COUNT_QUERY_WITH_COMMUNITY, NODEREL_COUNT_QUERY_WITHOUT_COMMUNITY, MAX_COMMUNITY_LEVELS
from ontographrag.kg.utils.source_node import sourceNode

def delete_file_from_gcs(bucket, folder, filename):
    """GCS file deletion — raises explicitly; GCS is not a supported deployment mode."""
    raise NotImplementedError(
        "GCS file deletion is not supported in this deployment. "
        "Set GCS_FILE_CACHE=False or remove the GCS_FILE_CACHE env var."
    )
import json
from dotenv import load_dotenv

load_dotenv()

class graphDBdataAccess:

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def update_exception_db(self, file_name, exp_msg, retry_condition=None):
        try:
            job_status = "Failed"
            result = self.get_current_status_document_node(file_name)
            if len(result) > 0:
                is_cancelled_status = result[0]['is_cancelled']
                if is_cancelled_status:
                    job_status = 'Cancelled'
            if retry_condition is not None: 
                retry_condition = None
                self.graph.query("""MERGE(d:Document {fileName :$fName}) SET d.status = $status, d.errorMessage = $error_msg, d.retry_condition = $retry_condition""",
                            {"fName":file_name, "status":job_status, "error_msg":exp_msg, "retry_condition":retry_condition},session_params={"database":self.graph._database})
            else :    
                self.graph.query("""MERGE(d:Document {fileName :$fName}) SET d.status = $status, d.errorMessage = $error_msg""",
                            {"fName":file_name, "status":job_status, "error_msg":exp_msg},session_params={"database":self.graph._database})
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in updating document node status as failed: {error_message}")
            raise Exception(error_message)
        
    def create_source_node(self, obj_source_node:sourceNode, kg_name: str = None):
        try:
            job_status = "New"
            logging.info(f"creating source node if does not exist in database {self.graph._database}")
            self.graph.query("""MERGE(d:Document {fileName :$fn}) SET d.fileSize = $fs, d.fileType = $ft ,
                            d.status = $st, d.url = $url, d.awsAccessKeyId = $awsacc_key_id,
                            d.fileSource = $f_source, d.createdAt = $c_at, d.updatedAt = $u_at,
                            d.processingTime = $pt, d.errorMessage = $e_message, d.nodeCount= $n_count,
                            d.relationshipCount = $r_count, d.model= $model, d.gcsBucket=$gcs_bucket,
                            d.gcsBucketFolder= $gcs_bucket_folder, d.language= $language,d.gcsProjectId= $gcs_project_id,
                            d.is_cancelled=False, d.total_chunks=0, d.processed_chunk=0,
                            d.access_token=$access_token,
                            d.chunkNodeCount=$chunkNodeCount,d.chunkRelCount=$chunkRelCount,
                            d.entityNodeCount=$entityNodeCount,d.entityEntityRelCount=$entityEntityRelCount,
                            d.communityNodeCount=$communityNodeCount,d.communityRelCount=$communityRelCount,
                            d.kgName = $kg_name""",
                            {"fn":obj_source_node.file_name, "fs":obj_source_node.file_size, "ft":obj_source_node.file_type, "st":job_status,
                            "url":obj_source_node.url,
                            "awsacc_key_id":obj_source_node.awsAccessKeyId, "f_source":obj_source_node.file_source, "c_at":obj_source_node.created_at,
                            "u_at":obj_source_node.created_at, "pt":0, "e_message":'', "n_count":0, "r_count":0, "model":obj_source_node.model,
                            "gcs_bucket": obj_source_node.gcsBucket, "gcs_bucket_folder": obj_source_node.gcsBucketFolder,
                            "language":obj_source_node.language, "gcs_project_id":obj_source_node.gcsProjectId,
                            "access_token":obj_source_node.access_token,
                            "chunkNodeCount":obj_source_node.chunkNodeCount,
                            "chunkRelCount":obj_source_node.chunkRelCount,
                            "entityNodeCount":obj_source_node.entityNodeCount,
                            "entityEntityRelCount":obj_source_node.entityEntityRelCount,
                            "communityNodeCount":obj_source_node.communityNodeCount,
                            "communityRelCount":obj_source_node.communityRelCount,
                            "kg_name": kg_name
                            },session_params={"database":self.graph._database})
        except Exception as e:
            error_message = str(e)
            logging.info(f"error_message = {error_message}")
            self.update_exception_db(obj_source_node.file_name, error_message)
            raise Exception(error_message)
        
    def update_source_node(self, obj_source_node:sourceNode):
        try:

            params = {}
            if obj_source_node.file_name is not None and obj_source_node.file_name != '':
                params['fileName'] = obj_source_node.file_name

            if obj_source_node.status is not None and obj_source_node.status != '':
                params['status'] = obj_source_node.status

            if obj_source_node.created_at is not None:
                params['createdAt'] = obj_source_node.created_at

            if obj_source_node.updated_at is not None:
                params['updatedAt'] = obj_source_node.updated_at

            if obj_source_node.processing_time is not None and obj_source_node.processing_time != 0:
                params['processingTime'] = round(obj_source_node.processing_time.total_seconds(),2)

            if obj_source_node.node_count is not None :
                params['nodeCount'] = obj_source_node.node_count

            if obj_source_node.relationship_count is not None :
                params['relationshipCount'] = obj_source_node.relationship_count

            if obj_source_node.model is not None and obj_source_node.model != '':
                params['model'] = obj_source_node.model

            if obj_source_node.total_chunks is not None and obj_source_node.total_chunks != 0:
                params['total_chunks'] = obj_source_node.total_chunks

            if obj_source_node.is_cancelled is not None:
                params['is_cancelled'] = obj_source_node.is_cancelled

            if obj_source_node.processed_chunk is not None :
                params['processed_chunk'] = obj_source_node.processed_chunk
            
            if obj_source_node.retry_condition is not None :
                params['retry_condition'] = obj_source_node.retry_condition    

            param= {"props":params}
            
            logging.info(f'Base Param value 1 : {param}')
            query = "MERGE(d:Document {fileName :$props.fileName}) SET d += $props"
            logging.info("Update source node properties")
            self.graph.query(query,param,session_params={"database":self.graph._database})
        except Exception as e:
            error_message = str(e)
            self.update_exception_db(obj_source_node.file_name, error_message)
            raise Exception(error_message)
    
    def get_source_list(self):
        """
        Args:
            uri: URI of the graph to extract
            db_name: db_name is database name to connect to graph db
            userName: Username to use for graph creation ( if None will use username from config file )
            password: Password to use for graph creation ( if None will use password from config file )
            file: File object containing the PDF file to be used
            model: Type of model to use ('Diffbot'or'OpenAI GPT')
        Returns:
        Returns a list of sources that are in the database by querying the graph and
        sorting the list by the last updated date. 
        """
        logging.info("Get existing files list from graph")
        query = "MATCH(d:Document) WHERE d.fileName IS NOT NULL RETURN d ORDER BY d.updatedAt DESC"
        result = self.graph.query(query,session_params={"database":self.graph._database})
        list_of_json_objects = [entry['d'] for entry in result]
        return list_of_json_objects
        
    def update_KNN_graph(self):
        """
        Update the graph node with SIMILAR relationship where embedding scrore match
        """
        index = self.graph.query("""show indexes yield * where type = 'VECTOR' and name = 'vector'""",session_params={"database":self.graph._database})
        # logging.info(f'show index vector: {index}')
        knn_min_score = os.environ.get('KNN_MIN_SCORE')
        if len(index) > 0:
            logging.info('update KNN graph')
            self.graph.query("""MATCH (c:Chunk)
                                    WHERE c.embedding IS NOT NULL AND count { (c)-[:SIMILAR]-() } < 5
                                    CALL db.index.vector.queryNodes('vector', 6, c.embedding) yield node, score
                                    WHERE node <> c and score >= $score MERGE (c)-[rel:SIMILAR]-(node) SET rel.score = score
                                """,
                                {"score":float(knn_min_score)}
                                ,session_params={"database":self.graph._database})
        else:
            logging.info("Vector index does not exist, So KNN graph not update")

    def check_account_access(self, database):
        try:
            query_dbms_componenet = "call dbms.components() yield edition"
            result_dbms_componenet = self.graph.query(query_dbms_componenet,session_params={"database":self.graph._database})

            if  result_dbms_componenet[0]["edition"] == "enterprise":
                query = """
                SHOW USER PRIVILEGES 
                YIELD * 
                WHERE graph = $database AND action IN ['read'] 
                RETURN COUNT(*) AS readAccessCount
                """
            
                logging.info(f"Checking access for database: {database}")

                result = self.graph.query(query, params={"database": database},session_params={"database":self.graph._database})
                read_access_count = result[0]["readAccessCount"] if result else 0

                logging.info(f"Read access count: {read_access_count}")

                if read_access_count > 0:
                    logging.info("The account has read access.")
                    return False
                else:
                    logging.info("The account has write access.")
                    return True
            else:
                #Community version have no roles to execute admin command, so assuming write access as TRUE
                logging.info("The account has write access.")
                return True

        except Exception as e:
            logging.error(f"Error checking account access: {e}")
            return False

    def check_gds_version(self):
        try:
            gds_procedure_count = """
            SHOW FUNCTIONS YIELD name WHERE name STARTS WITH 'gds.version' RETURN COUNT(*) AS totalGdsProcedures
            """
            result = self.graph.query(gds_procedure_count,session_params={"database":self.graph._database})
            total_gds_procedures = result[0]['totalGdsProcedures'] if result else 0

            if total_gds_procedures > 0:
                logging.info("GDS is available in the database.")
                return True
            else:
                logging.info("GDS is not available in the database.")
                return False
        except Exception as e:
            logging.error(f"An error occurred while checking GDS version: {e}")
            return False
            
    def connection_check_and_get_vector_dimensions(self,database):
        """
        Get the vector index dimension from database and application configuration and DB connection status
        
        Args:
            uri: URI of the graph to extract
            userName: Username to use for graph creation ( if None will use username from config file )
            password: Password to use for graph creation ( if None will use password from config file )
            db_name: db_name is database name to connect to graph db
        Returns:
        Returns a status of connection from NEO4j is success or failure
        """
        
        db_vector_dimension = self.graph.query("""SHOW INDEXES YIELD *
                                    WHERE type = 'VECTOR' AND name = 'vector'
                                    RETURN options.indexConfig['vector.dimensions'] AS vector_dimensions
                                """,session_params={"database":self.graph._database})
        
        result_chunks = self.graph.query("""match (c:Chunk) return size(c.embedding) as embeddingSize, count(*) as chunks, 
                                                    count(c.embedding) as hasEmbedding
                                """,session_params={"database":self.graph._database})
        
        embedding_model = os.getenv('EMBEDDING_MODEL')
        embeddings, application_dimension = load_embedding_model(embedding_model)
        logging.info(f'embedding model:{embeddings} and dimesion:{application_dimension}')

        gds_status = self.check_gds_version()
        write_access = self.check_account_access(database=database)
        
        if self.graph:
            if len(db_vector_dimension) > 0:
                return {'db_vector_dimension': db_vector_dimension[0]['vector_dimensions'], 'application_dimension':application_dimension, 'message':"Connection Successful","gds_status":gds_status,"write_access":write_access}
            else:
                if len(db_vector_dimension) == 0 and len(result_chunks) == 0:
                    logging.info("Chunks and vector index does not exists in database")
                    return {'db_vector_dimension': 0, 'application_dimension':application_dimension, 'message':"Connection Successful","chunks_exists":False,"gds_status":gds_status,"write_access":write_access}
                elif len(db_vector_dimension) == 0 and result_chunks[0]['hasEmbedding']==0 and result_chunks[0]['chunks'] > 0:
                    return {'db_vector_dimension': 0, 'application_dimension':application_dimension, 'message':"Connection Successful","chunks_exists":True,"gds_status":gds_status,"write_access":write_access}
                else:
                    return {'message':"Connection Successful","gds_status": gds_status,"write_access":write_access}

    def execute_query(self, query, param=None,max_retries=3, delay=2):
        retries = 0
        while retries < max_retries:
            try:
                return self.graph.query(query, param,session_params={"database":self.graph._database})
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    retries += 1
                    logging.info(f"Deadlock detected. Retrying {retries}/{max_retries} in {delay} seconds...")
                    time.sleep(delay)  # Wait before retrying
                else:
                    raise 
        logging.error("Failed to execute query after maximum retries due to persistent deadlocks.")
        raise RuntimeError("Query execution failed after multiple retries due to deadlock.")

    def get_current_status_document_node(self, file_name):
        query = """
                MATCH(d:Document {fileName : $file_name}) RETURN d.status AS Status , d.processingTime AS processingTime, 
                d.nodeCount AS nodeCount, d.model as model, d.relationshipCount as relationshipCount,
                d.total_chunks AS total_chunks , d.fileSize as fileSize, 
                d.is_cancelled as is_cancelled, d.processed_chunk as processed_chunk, d.fileSource as fileSource,
                d.chunkNodeCount AS chunkNodeCount,
                d.chunkRelCount AS chunkRelCount,
                d.entityNodeCount AS entityNodeCount,
                d.entityEntityRelCount AS entityEntityRelCount,
                d.communityNodeCount AS communityNodeCount,
                d.communityRelCount AS communityRelCount,
                d.createdAt AS created_time
                """
        param = {"file_name" : file_name}
        return self.execute_query(query, param)
    
    def delete_file_from_graph(self, filenames, source_types, deleteEntities:str, merged_dir:str, uri):
        
        filename_list= list(map(str.strip, json.loads(filenames)))
        source_types_list= list(map(str.strip, json.loads(source_types)))
        gcs_file_cache = os.environ.get('GCS_FILE_CACHE')
        
        for (file_name,source_type) in zip(filename_list, source_types_list):
            merged_file_path = os.path.join(merged_dir, file_name)
            if source_type == 'local file' and gcs_file_cache == 'True':
                folder_name = create_gcs_bucket_folder_name_hashed(uri, file_name)
                delete_file_from_gcs(BUCKET_UPLOAD,folder_name,file_name)
            else:
                logging.info(f'Deleted File Path: {merged_file_path} and Deleted File Name : {file_name}')
                delete_uploaded_local_file(merged_file_path,file_name)
                
        query_to_delete_document="""
            MATCH (d:Document)
            WHERE d.fileName IN $filename_list AND coalesce(d.fileSource, "None") IN $source_types_list
            WITH COLLECT(d) AS documents
            CALL (documents) {
            UNWIND documents AS d
            optional match (d)<-[:PART_OF]-(c:Chunk) 
            detach delete c, d
            } IN TRANSACTIONS OF 1 ROWS
            """
        query_to_delete_document_and_entities = """
            MATCH (d:Document)
            WHERE d.fileName IN $filename_list AND coalesce(d.fileSource, "None") IN $source_types_list
            WITH COLLECT(d) AS documents
            CALL (documents) {
            UNWIND documents AS d
            OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
            OPTIONAL MATCH (c:Chunk)-[:HAS_ENTITY]->(e)
            WITH d, c, e, documents
            WHERE NOT EXISTS {
                MATCH (e)<-[:HAS_ENTITY]-(c2)-[:PART_OF]->(d2:Document)
                WHERE NOT d2 IN documents
                }
            WITH d, COLLECT(c) AS chunks, COLLECT(e) AS entities
            FOREACH (chunk IN chunks | DETACH DELETE chunk)
            FOREACH (entity IN entities | DETACH DELETE entity)
            DETACH DELETE d
            } IN TRANSACTIONS OF 1 ROWS
            """
        query_to_delete_communities = """
            MATCH (c:`__Community__`)
            WHERE c.level = 0 AND NOT EXISTS { ()-[:IN_COMMUNITY]->(c) }
            DETACH DELETE c
            WITH 1 AS dummy
            UNWIND range(1, $max_level)  AS level
            CALL (level) {
                MATCH (c:`__Community__`)
                WHERE c.level = level AND NOT EXISTS { ()-[:PARENT_COMMUNITY]->(c) }
                DETACH DELETE c
                }
        """   
        param = {"filename_list" : filename_list, "source_types_list": source_types_list}
        community_param = {"max_level":MAX_COMMUNITY_LEVELS}
        if deleteEntities == "true":
            result = self.execute_query(query_to_delete_document_and_entities, param)
            _ = self.execute_query(query_to_delete_communities,community_param)
            logging.info(f"Deleting {len(filename_list)} documents = '{filename_list}' from '{source_types_list}' from database")
        else :
            result = self.execute_query(query_to_delete_document, param)    
            logging.info(f"Deleting {len(filename_list)} documents = '{filename_list}' from '{source_types_list}' with their entities from database")
        return len(filename_list)
    
    def list_unconnected_nodes(self):
        query = """
        MATCH (e:!Chunk&!Document&!`__Community__`) 
        WHERE NOT exists { (e)--(:!Chunk&!Document&!`__Community__`) }
        OPTIONAL MATCH (doc:Document)<-[:PART_OF]-(c:Chunk)-[:HAS_ENTITY]->(e)
        RETURN 
        e {
            .*,
            embedding: null,
            elementId: elementId(e),
            labels: CASE 
            WHEN size(labels(e)) > 1 THEN 
                apoc.coll.removeAll(labels(e), ["__Entity__"])
            ELSE 
                ["Entity"]
            END
        } AS e, 
        collect(distinct doc.fileName) AS documents, 
        count(distinct c) AS chunkConnections
        ORDER BY e.id ASC
        LIMIT 100
        """
        query_total_nodes = """
        MATCH (e:!Chunk&!Document&!`__Community__`) 
        WHERE NOT exists { (e)--(:!Chunk&!Document&!`__Community__`) }
        RETURN count(*) as total
        """
        nodes_list = self.execute_query(query)
        total_nodes = self.execute_query(query_total_nodes)
        return nodes_list, total_nodes[0]
    
    def delete_unconnected_nodes(self,unconnected_entities_list):
        entities_list = list(map(str.strip, json.loads(unconnected_entities_list)))
        query = """
        MATCH (e) WHERE elementId(e) IN $elementIds
        DETACH DELETE e
        """
        param = {"elementIds":entities_list}
        return self.execute_query(query,param)
    
    def get_duplicate_nodes_list(self):
        score_value = float(os.environ.get('DUPLICATE_SCORE_VALUE'))
        text_distance = int(os.environ.get('DUPLICATE_TEXT_DISTANCE'))
        query_duplicate_nodes = """
                MATCH (n:!Chunk&!Session&!Document&!`__Community__`) with n 
                WHERE n.embedding is not null and n.id is not null // and size(toString(n.id)) > 3
                WITH n ORDER BY count {{ (n)--() }} DESC, size(toString(n.id)) DESC // updated
                WITH collect(n) as nodes
                UNWIND nodes as n
                WITH n, [other in nodes 
                // only one pair, same labels e.g. Person with Person
                WHERE elementId(n) < elementId(other) and labels(n) = labels(other)
                // at least embedding similarity of X
                AND 
                (
                // either contains each other as substrings or has a text edit distinct of less than 3
                (size(toString(other.id)) > 2 AND toLower(toString(n.id)) CONTAINS toLower(toString(other.id))) OR 
                (size(toString(n.id)) > 2 AND toLower(toString(other.id)) CONTAINS toLower(toString(n.id)))
                OR (size(toString(n.id))>5 AND apoc.text.distance(toLower(toString(n.id)), toLower(toString(other.id))) < $duplicate_text_distance)
                OR
                vector.similarity.cosine(other.embedding, n.embedding) > $duplicate_score_value
                )] as similar
                WHERE size(similar) > 0 
                // remove duplicate subsets
                with collect([n]+similar) as all
                CALL {{ with all
                    unwind all as nodes
                    with nodes, all
                    // skip current entry if it's smaller and a subset of any other entry
                    where none(other in all where other <> nodes and size(other) > size(nodes) and size(apoc.coll.subtract(nodes, other))=0)
                    return head(nodes) as n, tail(nodes) as similar
                }}
                OPTIONAL MATCH (doc:Document)<-[:PART_OF]-(c:Chunk)-[:HAS_ENTITY]->(n)
                {return_statement}
                """
        return_query_duplicate_nodes = """
                RETURN n {.*, embedding:null, elementId:elementId(n), labels:labels(n)} as e, 
                [s in similar | s {.id, .description, labels:labels(s), elementId: elementId(s)}] as similar,
                collect(distinct doc.fileName) as documents, count(distinct c) as chunkConnections
                ORDER BY e.id ASC
                LIMIT 100
                """
        total_duplicate_nodes = "RETURN COUNT(DISTINCT(n)) as total"
        
        param = {"duplicate_score_value": score_value, "duplicate_text_distance" : text_distance}
        
        nodes_list = self.execute_query(query_duplicate_nodes.format(return_statement=return_query_duplicate_nodes),param=param)
        total_nodes = self.execute_query(query_duplicate_nodes.format(return_statement=total_duplicate_nodes),param=param)
        return nodes_list, total_nodes[0]
    
    def merge_duplicate_nodes(self,duplicate_nodes_list):
        nodes_list = json.loads(duplicate_nodes_list)
        logging.info(f'Nodes list to merge {nodes_list}')
        query = """
        UNWIND $rows AS row
        CALL { with row
        MATCH (first) WHERE elementId(first) = row.firstElementId
        MATCH (rest) WHERE elementId(rest) IN row.similarElementIds
        WITH first, collect (rest) as rest
        WITH [first] + rest as nodes
        CALL apoc.refactor.mergeNodes(nodes, 
        {properties:"discard",mergeRels:true, produceSelfRel:false, preserveExistingSelfRels:false, singleElementAsArray:true}) 
        YIELD node
        RETURN size(nodes) as mergedCount
        }
        RETURN sum(mergedCount) as totalMerged
        """
        param = {"rows":nodes_list}
        return self.execute_query(query,param)
    
    def drop_create_vector_index(self, isVectorIndexExist):
        """
        drop and create the vector index when vector index dimesion are different.
        """
        embedding_model = os.getenv('EMBEDDING_MODEL')
        embeddings, dimension = load_embedding_model(embedding_model)
        
        if isVectorIndexExist == 'true':
            self.graph.query("""drop index vector""",session_params={"database":self.graph._database})
        
        self.graph.query("""CREATE VECTOR INDEX `vector` if not exists for (c:Chunk) on (c.embedding)
                            OPTIONS {indexConfig: {
                            `vector.dimensions`: $dimensions,
                            `vector.similarity_function`: 'cosine'
                            }}
                        """,
                        {
                            "dimensions" : dimension
                        },session_params={"database":self.graph._database}
                        )
        return "Drop and Re-Create vector index succesfully"


    def update_node_relationship_count(self,document_name):
        logging.info("updating node and relationship count")
        label_query = """CALL db.labels"""
        community_flag = {'label': '__Community__'} in self.execute_query(label_query)
        if (not document_name) and (community_flag):
            result = self.execute_query(NODEREL_COUNT_QUERY_WITH_COMMUNITY)
        elif (not document_name) and (not community_flag):
             return []
        else:
            param = {"document_name": document_name}
            result = self.execute_query(NODEREL_COUNT_QUERY_WITHOUT_COMMUNITY, param)
        response = {}
        if result:
            for record in result:
                filename = record.get("filename",None)
                chunkNodeCount = int(record.get("chunkNodeCount",0))
                chunkRelCount = int(record.get("chunkRelCount",0))
                entityNodeCount = int(record.get("entityNodeCount",0))
                entityEntityRelCount = int(record.get("entityEntityRelCount",0))
                if (not document_name) and (community_flag):
                    communityNodeCount = int(record.get("communityNodeCount",0))
                    communityRelCount = int(record.get("communityRelCount",0))
                else:
                    communityNodeCount = 0
                    communityRelCount = 0
                nodeCount = int(chunkNodeCount) + int(entityNodeCount) + int(communityNodeCount)
                relationshipCount = int(chunkRelCount) + int(entityEntityRelCount) + int(communityRelCount)
                update_query = """
                MATCH (d:Document {fileName: $filename})
                SET d.chunkNodeCount = $chunkNodeCount,
                    d.chunkRelCount = $chunkRelCount,
                    d.entityNodeCount = $entityNodeCount,
                    d.entityEntityRelCount = $entityEntityRelCount,
                    d.communityNodeCount = $communityNodeCount,
                    d.communityRelCount = $communityRelCount,
                    d.nodeCount = $nodeCount,
                    d.relationshipCount = $relationshipCount
                """
                self.execute_query(update_query,{
                    "filename": filename,
                    "chunkNodeCount": chunkNodeCount,
                    "chunkRelCount": chunkRelCount,
                    "entityNodeCount": entityNodeCount,
                    "entityEntityRelCount": entityEntityRelCount,
                    "communityNodeCount": communityNodeCount,
                    "communityRelCount": communityRelCount,
                    "nodeCount" : nodeCount,
                    "relationshipCount" : relationshipCount
                    })
                
                response[filename] = {"chunkNodeCount": chunkNodeCount,
                    "chunkRelCount": chunkRelCount,
                    "entityNodeCount": entityNodeCount,
                    "entityEntityRelCount": entityEntityRelCount,
                    "communityNodeCount": communityNodeCount,
                    "communityRelCount": communityRelCount,
                    "nodeCount" : nodeCount,
                    "relationshipCount" : relationshipCount
                    }

        return response
    
    def get_nodelabels_relationships(self):
        node_query = """
                    CALL db.labels() YIELD label
                    WITH label
                    WHERE NOT label IN ['Document', 'Chunk', '_Bloom_Perspective_', '__Community__', '__Entity__']
                    CALL apoc.cypher.run("MATCH (n:`" + label + "`) RETURN count(n) AS count",{}) YIELD value
                    WHERE value.count > 0
                    RETURN label order by label
                    """

        relation_query = """
                CALL db.relationshipTypes() yield relationshipType
                WHERE NOT relationshipType  IN ['PART_OF', 'NEXT_CHUNK', 'HAS_ENTITY', '_Bloom_Perspective_','FIRST_CHUNK','SIMILAR','IN_COMMUNITY','PARENT_COMMUNITY'] 
                return relationshipType order by relationshipType
                """
            
        try:
            node_result = self.execute_query(node_query)
            node_labels = [record["label"] for record in node_result]
            relationship_result = self.execute_query(relation_query)
            relationship_types = [record["relationshipType"] for record in relationship_result]
            return node_labels,relationship_types
        except Exception as e:
            print(f"Error in getting node labels/relationship types from db: {e}")
            return []

    def get_websource_url(self,file_name):
        logging.info("Checking if same title with different URL exist in db ")
        query = """
                MATCH(d:Document {fileName : $file_name}) WHERE d.fileSource = "web-url"
                RETURN d.url AS url
                """
        param = {"file_name" : file_name}
        return self.execute_query(query, param)

    def get_kg_list(self):
        """
        Get list of all knowledge graphs in the database
        """
        logging.info("Get existing KG list from graph")
        query = """
        MATCH(d:Document)
        WHERE d.fileName IS NOT NULL AND d.kgName IS NOT NULL
        RETURN DISTINCT d.kgName AS kgName,
               count(d) AS documentCount,
               max(d.updatedAt) AS lastUpdated
        ORDER BY d.kgName
        """
        result = self.graph.query(query,session_params={"database":self.graph._database})
        return [entry for entry in result]

    def get_source_list_by_kg(self, kg_name: str):
        """
        Get list of sources for a specific knowledge graph
        """
        logging.info(f"Get existing files list for KG: {kg_name}")
        query = """
        MATCH(d:Document)
        WHERE d.fileName IS NOT NULL AND d.kgName = $kg_name
        RETURN d ORDER BY d.updatedAt DESC
        """
        param = {"kg_name": kg_name}
        result = self.graph.query(query, param, session_params={"database":self.graph._database})
        list_of_json_objects = [entry['d'] for entry in result]
        return list_of_json_objects

    def delete_kg(self, kg_name: str, delete_entities: bool = True):
        """
        Delete an entire knowledge graph and all its documents
        """
        logging.info(f"Deleting KG: {kg_name}")

        # Get all documents in this KG
        docs_query = """
        MATCH(d:Document {kgName: $kg_name})
        RETURN d.fileName AS fileName, d.fileSource AS fileSource
        """
        param = {"kg_name": kg_name}
        docs_result = self.execute_query(docs_query, param)

        if not docs_result:
            logging.info(f"No documents found for KG: {kg_name}")
            return 0

        # Extract filenames and source types
        filenames = [doc['fileName'] for doc in docs_result]
        source_types = [doc['fileSource'] or "None" for doc in docs_result]

        # Use existing delete_file_from_graph method
        # We need to mock the parameters it expects
        import tempfile
        import os

        # Create temporary directory for merged_dir parameter
        with tempfile.TemporaryDirectory() as temp_dir:
            deleted_count = self.delete_file_from_graph(
                filenames=json.dumps(filenames),
                source_types=json.dumps(source_types),
                deleteEntities=str(delete_entities).lower(),
                merged_dir=temp_dir,
                uri=""
            )

        logging.info(f"Deleted KG '{kg_name}' with {deleted_count} documents")
        return deleted_count

    def get_kg_stats(self, kg_name: str = None):
        """
        Get statistics for a specific KG or all KGs
        """
        if kg_name:
            query = """
            MATCH(d:Document {kgName: $kg_name})
            RETURN
                $kg_name AS kgName,
                count(d) AS documentCount,
                sum(d.nodeCount) AS totalNodes,
                sum(d.relationshipCount) AS totalRelationships,
                sum(d.chunkNodeCount) AS chunkNodes,
                sum(d.entityNodeCount) AS entityNodes,
                sum(d.communityNodeCount) AS communityNodes
            """
            param = {"kg_name": kg_name}
        else:
            query = """
            MATCH(d:Document)
            WHERE d.kgName IS NOT NULL
            RETURN
                d.kgName AS kgName,
                count(d) AS documentCount,
                sum(d.nodeCount) AS totalNodes,
                sum(d.relationshipCount) AS totalRelationships,
                sum(d.chunkNodeCount) AS chunkNodes,
                sum(d.entityNodeCount) AS entityNodes,
                sum(d.communityNodeCount) AS communityNodes
            ORDER BY kgName
            """

        result = self.execute_query(query, param if kg_name else None)
        return [entry for entry in result]

    # ========== Named KG Management Methods ==========
    
    def create_kg(self, kg_name: str, description: str = None, data_source: str = None) -> Dict:
        """
        Create a new named Knowledge Graph in the database.
        
        Args:
            kg_name: Unique name for the KG
            description: Optional description of the KG
            data_source: Source of the data (e.g., 'pubmedqa', 'mimic', 'guidelines')
            
        Returns:
            Dictionary with kg_name and creation status
        """
        import uuid
        from datetime import datetime
        
        kg_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        query = """
        CREATE (k:KG {
            id: $kg_id,
            name: $kg_name,
            description: $description,
            data_source: $data_source,
            created_at: $created_at,
            updated_at: $created_at
        })
        RETURN k
        """
        
        try:
            result = self.graph.query(query, {
                "kg_id": kg_id,
                "kg_name": kg_name,
                "description": description or f"Knowledge Graph: {kg_name}",
                "data_source": data_source,
                "created_at": created_at
            }, session_params={"database": self.graph._database})
            
            logging.info(f"Created KG: {kg_name}")
            return {
                "kg_name": kg_name,
                "kg_id": kg_id,
                "created_at": created_at,
                "status": "created"
            }
        except Exception as e:
            logging.error(f"Error creating KG {kg_name}: {e}")
            raise

    def get_kg(self, kg_name: str) -> Dict:
        """
        Get a named Knowledge Graph by name.
        
        Args:
            kg_name: Name of the KG to retrieve
            
        Returns:
            Dictionary with KG metadata or None if not found
        """
        query = """
        MATCH (k:KG {name: $kg_name})
        RETURN k
        """
        
        result = self.execute_query(query, {"kg_name": kg_name})
        
        if result and len(result) > 0:
            kg_node = result[0]['k']
            return {
                "kg_name": kg_node.get("name"),
                "kg_id": kg_node.get("id"),
                "description": kg_node.get("description"),
                "data_source": kg_node.get("data_source"),
                "created_at": kg_node.get("created_at"),
                "updated_at": kg_node.get("updated_at")
            }
        return None

    def list_kgs(self) -> List[Dict]:
        """
        List all named Knowledge Graphs in the database.
        
        Returns:
            List of dictionaries with KG metadata
        """
        query = """
        MATCH (k:KG)
        RETURN k
        ORDER BY k.created_at DESC
        """
        
        result = self.execute_query(query)
        
        kgs = []
        for record in result:
            kg_node = record['k']
            kgs.append({
                "kg_name": kg_node.get("name"),
                "kg_id": kg_node.get("id"),
                "description": kg_node.get("description"),
                "data_source": kg_node.get("data_source"),
                "created_at": kg_node.get("created_at"),
                "updated_at": kg_node.get("updated_at")
            })
        
        return kgs

    def update_kg(self, kg_name: str, description: str = None, data_source: str = None) -> Dict:
        """
        Update a Knowledge Graph's metadata.
        
        Args:
            kg_name: Name of the KG to update
            description: New description (optional)
            data_source: New data source (optional)
            
        Returns:
            Updated KG metadata
        """
        from datetime import datetime
        
        updates = []
        params = {"kg_name": kg_name}
        
        if description is not None:
            updates.append("k.description = $description")
            params["description"] = description
            
        if data_source is not None:
            updates.append("k.data_source = $data_source")
            params["data_source"] = data_source
            
        updates.append("k.updated_at = $updated_at")
        params["updated_at"] = datetime.now().isoformat()
        
        if not updates:
            return self.get_kg(kg_name)
        
        query = f"""
        MATCH (k:KG {{name: $kg_name}})
        SET {', '.join(updates)}
        RETURN k
        """
        
        result = self.execute_query(query, params)
        
        if result and len(result) > 0:
            return self.get_kg(kg_name)
        raise ValueError(f"KG '{kg_name}' not found")

    def delete_kg_by_name(self, kg_name: str, delete_entities: bool = True) -> int:
        """
        Delete a named Knowledge Graph and optionally its entities.
        
        Args:
            kg_name: Name of the KG to delete
            delete_entities: If True, delete entities linked to this KG
            
        Returns:
            Number of documents deleted
        """
        from datetime import datetime
        
        # First, get the KG node to verify it exists
        kg = self.get_kg(kg_name)
        if not kg:
            logging.warning(f"KG '{kg_name}' not found")
            return 0
        
        # Get all documents in this KG
        docs_query = """
        MATCH(d:Document {kgName: $kg_name})
        RETURN d.fileName AS fileName, d.fileSource AS fileSource
        """
        docs_result = self.execute_query(docs_query, {"kg_name": kg_name})

        deleted_count = 0
        if docs_result:
            filenames = [doc['fileName'] for doc in docs_result]
            source_types = [doc['fileSource'] or "None" for doc in docs_result]
            
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                deleted_count = self.delete_file_from_graph(
                    filenames=json.dumps(filenames),
                    source_types=json.dumps(source_types),
                    deleteEntities=str(delete_entities).lower(),
                    merged_dir=temp_dir,
                    uri=""
                )

        # Delete the KG node itself
        kg_delete_query = """
        MATCH (k:KG {name: $kg_name})
        DETACH DELETE k
        """
        self.execute_query(kg_delete_query, {"kg_name": kg_name})
        
        logging.info(f"Deleted KG '{kg_name}' and {deleted_count} documents")
        return deleted_count

    def get_kg_entities(self, kg_name: str, limit: int = 100) -> List[Dict]:
        """
        Get all entities belonging to a specific KG.
        
        Args:
            kg_name: Name of the KG
            limit: Maximum number of entities to return
            
        Returns:
            List of entity dictionaries
        """
        # First verify KG exists
        kg = self.get_kg(kg_name)
        if not kg:
            return []
        
        # Get documents in this KG, then their entities
        query = """
        MATCH (d:Document {kgName: $kg_name})-[:PART_OF]->(c:Chunk)-[:HAS_ENTITY|MENTIONS]->(e)
        WHERE NOT e:Chunk AND NOT e:Document
        RETURN DISTINCT e
        LIMIT $limit
        """
        
        result = self.execute_query(query, {"kg_name": kg_name, "limit": limit})
        
        entities = []
        for record in result:
            entity = record['e']
            entities.append({
                "id": entity.get("id"),
                "name": entity.get("name"),
                "type": entity.get("type"),
                "description": entity.get("description"),
                "labels": list(entity.labels) if hasattr(entity, 'labels') else []
            })
        
        return entities

    def get_kg_chunks(self, kg_name: str, limit: int = 100) -> List[Dict]:
        """
        Get all chunks belonging to a specific KG.
        
        Args:
            kg_name: Name of the KG
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunk dictionaries
        """
        # First verify KG exists
        kg = self.get_kg(kg_name)
        if not kg:
            return []
        
        query = """
        MATCH (d:Document {kgName: $kg_name})-[:PART_OF]->(c:Chunk)
        RETURN c
        ORDER BY c.position
        LIMIT $limit
        """
        
        result = self.execute_query(query, {"kg_name": kg_name, "limit": limit})
        
        chunks = []
        for record in result:
            chunk = record['c']
            chunks.append({
                "id": chunk.get("id"),
                "text": chunk.get("text"),
                "position": chunk.get("position"),
                "start_pos": chunk.get("start_pos"),
                "end_pos": chunk.get("end_pos")
            })
        
        return chunks

    def add_document_to_kg(self, kg_name: str, file_name: str) -> bool:
        """
        Add an existing document to a named KG.
        
        Args:
            kg_name: Name of the KG
            file_name: File name of the document to add
            
        Returns:
            True if successful
        """
        # Verify KG exists
        kg = self.get_kg(kg_name)
        if not kg:
            raise ValueError(f"KG '{kg_name}' not found")
        
        # Update the document's kgName
        query = """
        MATCH (d:Document {fileName: $file_name})
        SET d.kgName = $kg_name
        RETURN d
        """
        
        result = self.execute_query(query, {"kg_name": kg_name, "file_name": file_name})
        
        if result and len(result) > 0:
            logging.info(f"Added document '{file_name}' to KG '{kg_name}'")
            return True
        return False

    def kg_exists(self, kg_name: str) -> bool:
        """
        Check if a KG exists by name.
        
        Args:
            kg_name: Name of the KG to check
            
        Returns:
            True if KG exists
        """
        return self.get_kg(kg_name) is not None
