"""
RAG Setup Module
Retrieval-Augmented Generation setup for model-specific data.

This module builds a retrieval system that enables efficient querying
of model data using both direct lookup and semantic search.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import configuration
try:
    from ..config import Config
    from ..utils.file_utils import FileUtils
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pdf_extractor.config import Config
    from pdf_extractor.utils.file_utils import FileUtils

logger = Config.setup_logging(__name__)


class RAGRetriever:
    """
    Retrieval system for model-specific data.
    
    Supports both direct lookup by model name and semantic search
    using embeddings and vector similarity.
    """
    
    def __init__(
        self,
        embedding_model: str = None,
        top_k: int = None
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            embedding_model: Name of the sentence-transformer model
            top_k: Number of top results to return for semantic search
        """
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.top_k = top_k or Config.TOP_K_RESULTS
        
        # Data storage
        self.model_data: Dict[str, Dict[str, Any]] = {}
        self.model_names: List[str] = []
        
        # Embedding components
        self.embedder = None
        self.model_embeddings = None
        self.faiss_index = None
        
        # State flags
        self.direct_lookup_ready = False
        self.semantic_search_ready = False
        
        logger.info(f"Initialized RAGRetriever with embedding model: {self.embedding_model_name}")
    
    def _initialize_embedder(self) -> bool:
        """
        Initialize the sentence transformer for embeddings.
        
        Returns:
            True if successful, False otherwise
        """
        if self.embedder is not None:
            return True
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
            return True
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic search disabled.")
            logger.warning("Install with: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def _initialize_faiss(self, embeddings: np.ndarray) -> bool:
        """
        Initialize FAISS index with embeddings.
        
        Args:
            embeddings: Numpy array of embeddings [num_models, embedding_dim]
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import faiss
            
            # Create FAISS index (L2 distance)
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings.astype(np.float32))
            
            logger.info(f"FAISS index created with {embeddings.shape[0]} vectors")
            return True
            
        except ImportError:
            logger.warning("faiss-cpu not installed. Using numpy-based similarity search.")
            logger.warning("Install with: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return False
    
    def _get_model_text(self, model_data: Dict[str, Any]) -> str:
        """
        Extract text content from model data for embedding.
        
        Args:
            model_data: Consolidated model data dictionary
            
        Returns:
            Text string representing the model's content
        """
        texts = []
        
        # Add model name
        model_name = model_data.get("model_name", "")
        if model_name:
            texts.append(f"Model: {model_name}")
        
        # Add text content
        for text_item in model_data.get("all_text", []):
            text = text_item.get("text", "")
            if text:
                texts.append(text)
        
        # Add table content
        for table in model_data.get("all_tables", []):
            for row in table.get("rows", []):
                for cell in row:
                    if cell:
                        texts.append(str(cell))
        
        return " ".join(texts)
    
    def load_data(self, data_dir: str = None) -> bool:
        """
        Load model data from the grouped_by_model directory.
        
        Args:
            data_dir: Directory containing model subdirectories
            
        Returns:
            True if data was loaded successfully
        """
        data_dir = Path(data_dir) if data_dir else Config.GROUPED_OUTPUT_DIR
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return False
        
        logger.info(f"Loading model data from {data_dir}...")
        
        # Clear existing data
        self.model_data.clear()
        self.model_names.clear()
        
        # Find all model subdirectories
        model_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
        
        if not model_dirs:
            logger.warning(f"No model directories found in {data_dir}")
            return False
        
        # Load data from each model directory
        for model_dir in model_dirs:
            # Find the JSON file in the model directory
            json_files = list(model_dir.glob("*.json"))
            
            if not json_files:
                logger.warning(f"No JSON file found in {model_dir}")
                continue
            
            # Load the first JSON file found
            json_file = json_files[0]
            data = FileUtils.read_json(json_file)
            
            if data is None:
                logger.warning(f"Failed to load {json_file}")
                continue
            
            model_name = data.get("model_name", model_dir.name)
            self.model_data[model_name] = data
            self.model_names.append(model_name)
            
            logger.debug(f"Loaded model: {model_name}")
        
        logger.info(f"Loaded data for {len(self.model_data)} models")
        self.direct_lookup_ready = len(self.model_data) > 0
        
        return self.direct_lookup_ready
    
    def build_index(self) -> bool:
        """
        Build the semantic search index.
        
        Must be called after load_data().
        
        Returns:
            True if index was built successfully
        """
        if not self.model_data:
            logger.error("No model data loaded. Call load_data() first.")
            return False
        
        # Initialize embedder
        if not self._initialize_embedder():
            logger.warning("Semantic search will not be available")
            return False
        
        logger.info("Building semantic search index...")
        
        # Generate embeddings for each model
        model_texts = []
        for model_name in self.model_names:
            data = self.model_data[model_name]
            text = self._get_model_text(data)
            model_texts.append(text)
        
        # Create embeddings
        try:
            logger.info(f"Generating embeddings for {len(model_texts)} models...")
            embeddings = self.embedder.encode(
                model_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            self.model_embeddings = embeddings
            
            # Try to use FAISS
            if not self._initialize_faiss(embeddings):
                logger.info("Using numpy-based similarity search as fallback")
            
            self.semantic_search_ready = True
            logger.info("Semantic search index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build search index: {e}")
            return False
    
    def save_index(self, output_dir: str = None) -> bool:
        """
        Save the index to disk for later use.
        
        Args:
            output_dir: Directory to save index files
            
        Returns:
            True if saved successfully
        """
        output_dir = Path(output_dir) if output_dir else Config.RAG_INDEX_DIR
        FileUtils.ensure_directory(output_dir)
        
        try:
            # Save model names list
            names_file = output_dir / "model_names.json"
            FileUtils.write_json(self.model_names, names_file)
            
            # Save embeddings
            if self.model_embeddings is not None:
                embeddings_file = output_dir / "embeddings.npy"
                np.save(embeddings_file, self.model_embeddings)
                logger.info(f"Saved embeddings to {embeddings_file}")
            
            # Save FAISS index if available
            if self.faiss_index is not None:
                try:
                    import faiss
                    faiss_file = output_dir / "faiss_index.bin"
                    faiss.write_index(self.faiss_index, str(faiss_file))
                    logger.info(f"Saved FAISS index to {faiss_file}")
                except Exception as e:
                    logger.warning(f"Could not save FAISS index: {e}")
            
            logger.info(f"Index saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, index_dir: str = None) -> bool:
        """
        Load a previously saved index from disk.
        
        Args:
            index_dir: Directory containing saved index files
            
        Returns:
            True if loaded successfully
        """
        index_dir = Path(index_dir) if index_dir else Config.RAG_INDEX_DIR
        
        try:
            # Load model names
            names_file = index_dir / "model_names.json"
            model_names = FileUtils.read_json(names_file)
            if model_names:
                self.model_names = model_names
            
            # Load embeddings
            embeddings_file = index_dir / "embeddings.npy"
            if embeddings_file.exists():
                self.model_embeddings = np.load(embeddings_file)
                logger.info(f"Loaded embeddings: {self.model_embeddings.shape}")
                
                # Try to load FAISS index
                faiss_file = index_dir / "faiss_index.bin"
                if faiss_file.exists():
                    try:
                        import faiss
                        self.faiss_index = faiss.read_index(str(faiss_file))
                        logger.info("Loaded FAISS index")
                    except Exception:
                        logger.info("FAISS index not loaded, using numpy similarity")
                
                self.semantic_search_ready = True
            
            logger.info(f"Loaded index from {index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _direct_lookup(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Look up model data by exact name match.
        
        Args:
            model_name: Model name to look up
            
        Returns:
            Model data dictionary or None if not found
        """
        # Exact match
        if model_name in self.model_data:
            return self.model_data[model_name]
        
        # Case-insensitive match
        model_name_lower = model_name.lower()
        for name in self.model_names:
            if name.lower() == model_name_lower:
                return self.model_data[name]
        
        # Partial match
        for name in self.model_names:
            if model_name_lower in name.lower() or name.lower() in model_name_lower:
                return self.model_data[name]
        
        return None
    
    def _semantic_search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (model_name, similarity_score, model_data)
        """
        top_k = top_k or self.top_k
        
        if not self.semantic_search_ready:
            logger.warning("Semantic search not available")
            return []
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            
            results = []
            
            if self.faiss_index is not None:
                # Use FAISS for search
                distances, indices = self.faiss_index.search(
                    query_embedding.astype(np.float32),
                    min(top_k, len(self.model_names))
                )
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(self.model_names):
                        model_name = self.model_names[idx]
                        # Convert L2 distance to similarity score
                        similarity = 1.0 / (1.0 + dist)
                        results.append((
                            model_name,
                            float(similarity),
                            self.model_data.get(model_name, {})
                        ))
            else:
                # Fallback: numpy-based similarity
                similarities = np.dot(
                    self.model_embeddings,
                    query_embedding.T
                ).flatten()
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    model_name = self.model_names[idx]
                    results.append((
                        model_name,
                        float(similarities[idx]),
                        self.model_data.get(model_name, {})
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def query(
        self,
        query_string: str,
        use_semantic: bool = True,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Query the retrieval system.
        
        Tries direct lookup first, then falls back to semantic search.
        
        Args:
            query_string: Query string
            use_semantic: Whether to use semantic search if direct lookup fails
            top_k: Number of results for semantic search
            
        Returns:
            Dictionary with query results
        """
        if not self.direct_lookup_ready:
            logger.error("No data loaded. Call load_data() first.")
            return {"success": False, "error": "No data loaded"}
        
        logger.info(f"Processing query: {query_string}")
        
        result = {
            "query": query_string,
            "method": None,
            "matches": [],
            "success": True
        }
        
        # Try direct lookup first
        direct_result = self._direct_lookup(query_string)
        if direct_result:
            logger.info(f"Direct lookup successful: {direct_result.get('model_name')}")
            result["method"] = "direct_lookup"
            result["matches"] = [{
                "model_name": direct_result.get("model_name"),
                "score": 1.0,
                "data": direct_result
            }]
            return result
        
        # Try semantic search
        if use_semantic and self.semantic_search_ready:
            logger.info("Using semantic search...")
            semantic_results = self._semantic_search(query_string, top_k)
            
            if semantic_results:
                result["method"] = "semantic_search"
                result["matches"] = [
                    {
                        "model_name": name,
                        "score": score,
                        "data": data
                    }
                    for name, score, data in semantic_results
                ]
                logger.info(f"Semantic search returned {len(semantic_results)} results")
                return result
        
        # No results found
        logger.warning(f"No results found for query: {query_string}")
        result["success"] = False
        result["error"] = "No matching results found"
        return result
    
    def get_context_for_llm(
        self,
        query_string: str,
        max_text_items: int = 50,
        include_tables: bool = True
    ) -> str:
        """
        Get formatted context for use with an LLM.
        
        Args:
            query_string: Query string
            max_text_items: Maximum number of text items to include
            include_tables: Whether to include table data
            
        Returns:
            Formatted context string
        """
        result = self.query(query_string)
        
        if not result.get("success") or not result.get("matches"):
            return f"No relevant data found for query: {query_string}"
        
        context_parts = []
        
        for match in result["matches"][:3]:  # Top 3 matches
            model_name = match.get("model_name", "Unknown")
            data = match.get("data", {})
            score = match.get("score", 0)
            
            context_parts.append(f"=== Model: {model_name} (relevance: {score:.2f}) ===\n")
            
            # Add text content
            texts = data.get("all_text", [])[:max_text_items]
            if texts:
                text_content = " ".join(t.get("text", "") for t in texts)
                context_parts.append(f"Content:\n{text_content}\n")
            
            # Add table content
            if include_tables:
                tables = data.get("all_tables", [])
                for i, table in enumerate(tables[:3]):  # Top 3 tables
                    rows = table.get("rows", [])
                    if rows:
                        context_parts.append(f"\nTable {i+1}:")
                        for row in rows[:20]:  # Limit rows
                            row_text = " | ".join(str(cell) for cell in row)
                            context_parts.append(f"  {row_text}")
            
            context_parts.append("\n")
        
        return "\n".join(context_parts)
    
    def interactive_query(self):
        """
        Run an interactive query loop.
        """
        print("\n=== RAG Retrieval System ===")
        print(f"Loaded {len(self.model_data)} models")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                result = self.query(query)
                
                print(f"\nMethod: {result.get('method', 'none')}")
                
                if result.get("success"):
                    for match in result["matches"]:
                        print(f"  - {match['model_name']} (score: {match['score']:.3f})")
                        data = match.get("data", {})
                        print(f"    Pages: {data.get('page_count', 0)}, "
                              f"Tables: {data.get('table_count', 0)}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
                
                print()
                
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")


def main():
    """Command-line entry point for RAG setup."""
    parser = argparse.ArgumentParser(
        description="RAG retrieval system for model-specific data"
    )
    parser.add_argument(
        "--data", "-d",
        default=None,
        help="Directory containing grouped model data (default: output/grouped_by_model)"
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Query string (if not provided, starts interactive mode)"
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build and save the search index"
    )
    parser.add_argument(
        "--load-index",
        action="store_true",
        help="Load previously saved index"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Number of results to return (default: {Config.TOP_K_RESULTS})"
    )
    
    args = parser.parse_args()
    
    # Create retriever
    retriever = RAGRetriever(top_k=args.top_k)
    
    # Load data
    if not retriever.load_data(args.data):
        logger.error("Failed to load data")
        return 1
    
    # Handle index operations
    if args.load_index:
        retriever.load_index()
    elif args.build_index or not retriever.load_index():
        if retriever.build_index():
            retriever.save_index()
        else:
            logger.warning("Semantic search not available, using direct lookup only")
    
    # Process query or start interactive mode
    if args.query:
        result = retriever.query(args.query)
        
        if result.get("success"):
            print(f"\nQuery: {result['query']}")
            print(f"Method: {result['method']}")
            print("\nResults:")
            for match in result["matches"]:
                print(f"  - {match['model_name']} (score: {match['score']:.3f})")
            
            # Print context
            print("\nContext for LLM:")
            print("-" * 50)
            print(retriever.get_context_for_llm(args.query))
        else:
            print(f"No results found: {result.get('error')}")
            return 1
    else:
        retriever.interactive_query()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
