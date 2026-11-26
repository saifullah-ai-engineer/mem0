import logging
from typing import Dict, List, Optional

from pydantic import BaseModel

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("The 'chromadb' library is required. Please install it using 'pip install chromadb'.")

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # distance
    payload: Optional[Dict]  # metadata


class ChromaDB(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        client: Optional[chromadb.Client] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
    ):
        """
        Initialize the Chromadb vector store.

        Args:
            collection_name (str): Name of the collection.
            client (chromadb.Client, optional): Existing chromadb client instance. Defaults to None.
            host (str, optional): Host address for chromadb server. Defaults to None.
            port (int, optional): Port for chromadb server. Defaults to None.
            path (str, optional): Path for local chromadb database. Defaults to None.
            api_key (str, optional): ChromaDB Cloud API key. Defaults to None.
            tenant (str, optional): ChromaDB Cloud tenant ID. Defaults to None.
        """
        if client:
            self.client = client
        elif api_key and tenant:
            # Initialize ChromaDB Cloud client
            logger.info("Initializing ChromaDB Cloud client")
            self.client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database="mem0"  # Use fixed database name for cloud
            )
        else:
            # Initialize local or server client
            self.settings = Settings(anonymized_telemetry=False)

            if host and port:
                self.settings.chroma_server_host = host
                self.settings.chroma_server_http_port = port
                self.settings.chroma_api_impl = "chromadb.api.fastapi.FastAPI"
            else:
                if path is None:
                    path = "db"

            self.settings.persist_directory = path
            self.settings.is_persistent = True

            self.client = chromadb.Client(self.settings)

        self.collection_name = collection_name
        self.collection = self.create_col(collection_name)

    def _parse_output(self, data: Dict) -> List[OutputData]:
        """
        Parse the output data.

        Args:
            data (Dict): Output data.

        Returns:
            List[OutputData]: Parsed output data.
        """
        keys = ["ids", "distances", "metadatas"]
        values = []

        for key in keys:
            value = data.get(key, [])
            if isinstance(value, list) and value and isinstance(value[0], list):
                value = value[0]
            values.append(value)

        ids, distances, metadatas = values
        max_length = max(len(v) for v in values if isinstance(v, list) and v is not None)

        result = []
        for i in range(max_length):
            raw_distance = distances[i] if isinstance(distances, list) and distances and i < len(distances) else None
            score = 1.0 / (1.0 + raw_distance) if raw_distance is not None else None
            entry = OutputData(
                id=ids[i] if isinstance(ids, list) and ids and i < len(ids) else None,
                score=score,
                payload=(metadatas[i] if isinstance(metadatas, list) and metadatas and i < len(metadatas) else None),
            )
            result.append(entry)

        return result

    def create_col(self, name: str, embedding_fn: Optional[callable] = None):
        """
        Create a new collection.

        Args:
            name (str): Name of the collection.
            embedding_fn (Optional[callable]): Embedding function to use. Defaults to None.

        Returns:
            chromadb.Collection: The created or retrieved collection.
        """
        collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_fn,
        )
        return collection

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Insert vectors into a collection.

        Args:
            vectors (List[list]): List of vectors to insert.
            payloads (Optional[List[Dict]], optional): List of payloads corresponding to vectors. Defaults to None.
            ids (Optional[List[str]], optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        self.collection.add(ids=ids, embeddings=vectors, metadatas=payloads)

    def search(
        self, query: str, vectors: List[list], top_k: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            vectors (List[list]): List of vectors to search.
            top_k (int, optional): Number of results to return. Defaults to 5.
            filters (Optional[Dict], optional): Filters to apply to the search. Defaults to None.

        Returns:
            List[OutputData]: Search results.
        """
        where_clause = self._generate_where_clause(filters) if filters else None
        results = self.collection.query(query_embeddings=vectors, where=where_clause, n_results=top_k)
        final_results = self._parse_output(results)
        return final_results

    def delete(self, vector_id: str):
        """
        Delete a vector by ID.

        Args:
            vector_id (str): ID of the vector to delete.
        """
        self.collection.delete(ids=[vector_id])

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ) -> Optional[bool]:
        """
        Update a vector and its payload.

        Args:
            vector_id (str): ID of the vector to update.
            vector (Optional[List[float]], optional): Updated vector. Defaults to None.
            payload (Optional[Dict], optional): Updated payload. Defaults to None.

        Returns:
            Optional[bool]: True if the record exists and was updated, False if no record has this ID,
                None if the update was sent but the read-back failed.
        """
        self.collection.update(
            ids=[vector_id],
            embeddings=[vector] if vector is not None else None,
            metadatas=[payload] if payload is not None else None,
        )
        # Chroma ignores an update to a missing id and returns no count, so read the id back.
        try:
            return bool(self.collection.get(ids=[vector_id])["ids"])
        except Exception as e:
            logger.warning(f"Updated vector {vector_id} but could not confirm it still exists: {e}")
            return None

    def get(self, vector_id: str) -> Optional[OutputData]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve.

        Returns:
            Optional[OutputData]: Retrieved vector, or None if the ID is not found.
        """
        result = self.collection.get(ids=[vector_id])
        parsed = self._parse_output(result)
        return parsed[0] if parsed else None

    def list_cols(self) -> List[chromadb.Collection]:
        """
        List all collections.

        Returns:
            List[chromadb.Collection]: List of collections.
        """
        return self.client.list_collections()

    def delete_col(self):
        """
        Delete a collection.
        """
        self.client.delete_collection(name=self.collection_name)

    def col_info(self) -> Dict:
        """
        Get information about a collection.

        Returns:
            Dict: Collection information.
        """
        return self.client.get_collection(name=self.collection_name)

    def list(self, filters: Optional[Dict] = None, top_k: int = 100) -> List[OutputData]:
        """
        List all vectors in a collection.

        Args:
            filters (Optional[Dict], optional): Filters to apply to the list. Defaults to None.
            top_k (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            List[OutputData]: List of vectors.
        """
        where_clause = self._generate_where_clause(filters) if filters else None
        results = self.collection.get(where=where_clause, limit=top_k)
        return [self._parse_output(results)]

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.collection = self.create_col(self.collection_name)

    @staticmethod
    def _generate_where_clause(where: dict[str, any]) -> dict[str, any]:
        """
        Generate a properly formatted where clause for ChromaDB.

        ChromaDB's where grammar allows exactly one field or one logical
        operator per dict level, so multiple operators on the same field and
        multi-field conditions must be combined with an explicit ``$and``.

        Args:
            where (dict[str, any]): The filter conditions.

        Returns:
            dict[str, any]: Properly formatted where clause for ChromaDB.
        """
        if where is None:
            return None

        op_map = {
            "eq": "$eq",
            "ne": "$ne",
            "gt": "$gt",
            "gte": "$gte",
            "lt": "$lt",
            "lte": "$lte",
            "in": "$in",
            "nin": "$nin",
        }
        # Negation of each operator. contains/icontains fall back to equality
        # on the positive path (ChromaDB has no substring match), so their
        # negation falls back to inequality for consistency.
        negate_map = {
            "eq": "$ne",
            "ne": "$eq",
            "gt": "$lte",
            "gte": "$lt",
            "lt": "$gte",
            "lte": "$gt",
            "in": "$nin",
            "nin": "$in",
        }

        def convert_condition(key: str, value: any) -> list:
            """Convert one field condition to a list of single-field ChromaDB clauses."""
            if value == "*":
                # Wildcard - match any value (ChromaDB doesn't have direct wildcard, so we skip this filter)
                return []
            if isinstance(value, dict):
                # One clause per operator: ChromaDB rejects field expressions
                # with more than one operator, so a range like
                # {"gte": 18, "lte": 65} must become two clauses combined
                # with $and by the caller (previously each operator
                # overwrote the last, silently dropping bounds).
                # contains/icontains and unknown operators fall back to equality.
                return [{key: {op_map.get(op, "$eq"): val}} for op, val in value.items()]
            # Simple equality
            return [{key: {"$eq": value}}]

        def combine(clauses: list, operator: str):
            """Combine clauses under a logical operator, unwrapping singletons."""
            if not clauses:
                return None
            if len(clauses) == 1:
                return clauses[0]
            return {operator: clauses}

        # Logical operators in mem0's enhanced filter syntax. Qdrant accepts
        # both the uppercase form (AND/OR/NOT) and the dollar form
        # ($and/$or/$not); ChromaDB only understands the dollar form, so
        # normalize here. A bare "AND"/"OR"/"NOT" key is therefore a logical
        # operator, never a metadata field name, matching Qdrant.
        logical_map = {
            "AND": "$and",
            "$and": "$and",
            "OR": "$or",
            "$or": "$or",
            "NOT": "$not",
            "$not": "$not",
        }

        def negate_clause(clause):
            """Negate one ChromaDB where clause (De Morgan for $and/$or)."""
            if "$and" in clause:
                return {"$or": [negate_clause(sub) for sub in clause["$and"]]}
            if "$or" in clause:
                return {"$and": [negate_clause(sub) for sub in clause["$or"]]}
            # Single field expression {field: {op: value}}. The operators
            # here are already ChromaDB dollar form ($gt, $in), while
            # negate_map is keyed by the bare mem0 operators (gt, in),
            # so strip the "$" before the lookup. Its values already
            # carry the "$", so nothing is added back.
            for field, expression in clause.items():
                if isinstance(expression, dict):
                    negated = {}
                    for op, val in expression.items():
                        base = op[1:] if isinstance(op, str) and op.startswith("$") else op
                        negated[negate_map.get(base, "$ne")] = val
                    return {field: negated}
            return clause

        def process(filter_dict):
            """Convert one filter dict to a ChromaDB where clause."""
            processed_filters = []
            for key, value in filter_dict.items():
                operator = logical_map.get(key)
                if operator in ("$and", "$or"):
                    # Each element is a sub-filter dict; recurse so nested
                    # logical operators (e.g. OR inside AND) are normalized
                    # too instead of becoming literal field names.
                    conditions = value if isinstance(value, list) else [value]
                    sub_clauses = []
                    for condition in conditions:
                        clause = process(condition)
                        if clause:
                            sub_clauses.append(clause)
                    combined = combine(sub_clauses, operator)
                    if combined:
                        processed_filters.append(combined)
                elif operator == "$not":
                    # NOT(a AND b) == (NOT a) OR (NOT b), and NOT of several
                    # conditions is the AND of their negations.
                    conditions = value if isinstance(value, list) else [value]
                    negated = []
                    for condition in conditions:
                        clause = process(condition)
                        if clause:
                            negated.append(negate_clause(clause))
                    combined = combine(negated, "$and")
                    if combined:
                        processed_filters.append(combined)
                else:
                    # Regular condition
                    combined = combine(convert_condition(key, value), "$and")
                    if combined:
                        processed_filters.append(combined)
            return combine(processed_filters, "$and")

        return process(where)
