import json
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

class ProjectEmbeddingModel:
    """
    Embedding model for ETHGlobal projects with semantic search capabilities.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.projects: List[Dict[str, Any]] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self.project_ids: List[str] = []
        
        # Initialize the embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded embedding model: {model_name}")
        except ImportError:
            logger.error("❌ Sentence Transformers not available. Install with: pip install sentence-transformers")
            self.model = None
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding using the selected model."""
        if self.model:
            return self.model.encode(text)
        else:
            # Fallback to simple hash-based embedding
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding method."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_bytes = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        embedding = np.array(hash_bytes[:384], dtype=np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def load_projects_from_json(self, filepath: str) -> None:
        """Load projects from JSON file and create embeddings."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.projects = json.load(f)
            
            logger.info(f"📁 Loaded {len(self.projects)} projects from {filepath}")
            self._create_embeddings()
            
        except Exception as e:
            logger.error(f"❌ Error loading projects: {e}")
            self.projects = []
    
    def load_projects_from_data(self, projects: List[Dict[str, Any]]) -> None:
        """Load projects from data and create embeddings."""
        # Only create embeddings if we don't already have them
        if self.embeddings:
            logger.info(f"📁 Embeddings already loaded ({len(self.embeddings)}), skipping creation")
            return
            
        self.projects = projects
        logger.info(f"📁 Loading {len(self.projects)} projects from data")
        self._create_embeddings()
    
    def _create_embeddings(self) -> None:
        """Create embeddings for all projects."""
        if not self.projects:
            logger.warning("No projects to create embeddings for")
            return
        
        logger.info("📁 Loading projects from data")
        
        for i, project in enumerate(self.projects):
            # Create searchable text from project data
            searchable_text = self._create_searchable_text(project)
            
            # Create embedding
            embedding = self.create_embedding(searchable_text)
            
            # Store with project ID
            project_id = f"project_{i}"
            self.embeddings[project_id] = embedding
            self.project_ids.append(project_id)
        
        logger.info(f"✅ Created embeddings for {len(self.projects)} projects")
    
    def _create_searchable_text(self, project: Dict[str, Any]) -> str:
        """Create searchable text from project data."""
        text_parts = []
        
        # Add title
        if 'project_title' in project:
            text_parts.append(project['project_title'])
        
        # Add short description
        if 'short_description' in project:
            text_parts.append(project['short_description'])
        
        # Add full description
        if 'project_description' in project:
            text_parts.append(project['project_description'])
        
        # Add how it's made
        if 'how_its_made' in project:
            text_parts.append(project['how_its_made'])
        
        return " ".join(text_parts)
    
    def search_projects(self, query: str, top_k: int = 5, min_similarity: float = 0.2) -> List[Dict[str, Any]]:
        """Search for similar projects using semantic similarity."""
        if not self.embeddings:
            logger.warning("No embeddings available for search")
            return []
        
        # Create query embedding
        query_embedding = self.create_embedding(query)
        
        # Calculate similarities
        similarities = []
        for project_id, project_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, project_embedding)
            similarities.append((project_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum similarity and get top results
        results = []
        for project_id, similarity in similarities[:top_k * 2]:  # Get more to filter
            if similarity >= min_similarity:
                # Get project index from project_id
                project_index = int(project_id.split('_')[1])
                if project_index < len(self.projects):
                    project = self.projects[project_index]
                    result = {
                        "project": project,
                        "similarity": float(similarity),
                        "project_id": project_id
                    }
                    results.append(result)
                    
                    if len(results) >= top_k:
                        break
        
        return results
    
    def analyze_search_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Analyze and log search results."""
        if not results:
            logger.info(f"🔍 No results found for query: '{query}'")
            return
        
        logger.info(f"🔍 Search Results for: '{query}'")
        logger.info("=" * 60)
        
        for i, result in enumerate(results, 1):
            project = result["project"]
            similarity = result["similarity"]
            
            # Determine match quality
            if similarity >= 0.6:
                quality = "✨ ✅ GOOD MATCH"
            elif similarity >= 0.4:
                quality = "✨ ⚠️  DECENT MATCH"
            else:
                quality = "✨ ❌ WEAK MATCH"
            
            logger.info(f"{i}. {quality} (Score: {similarity:.3f})")
            logger.info(f"   📛 Title: {project.get('project_title', 'N/A')}")
            
            description = project.get('short_description', 'No description')
            if len(description) > 80:
                description = description[:80] + "..."
            logger.info(f"   📝 Description: {description}")
            
            if 'url' in project:
                logger.info(f"   🔗 URL: {project['url']}")
            
            # Find matching terms
            matched_terms = self._find_matching_terms(query, description)
            if matched_terms:
                logger.info(f"   🎯 Matched terms: {', '.join(matched_terms)}")
            
            logger.info("-" * 60)
    
    def _find_matching_terms(self, query: str, description: str) -> List[str]:
        """Find terms that match between query and description."""
        query_terms = set(query.lower().split())
        desc_terms = set(description.lower().split())
        return list(query_terms.intersection(desc_terms))
    
    def get_project_details(self, project_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific project."""
        project_index = int(project_id.split('_')[1])
        if project_index < len(self.projects):
            return self.projects[project_index]
        return {}
    
    def save_embeddings(self, filepath: str) -> None:
        """Save embeddings to file."""
        try:
            data = {
                "embeddings": self.embeddings,
                "project_ids": self.project_ids,
                "model_name": self.model_name,
                "projects": self.projects  # Also save the projects data
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"💾 Saved embeddings to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving embeddings: {e}")
    
    def load_embeddings(self, filepath: str) -> None:
        """Load embeddings from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            logger.info(f"📁 Loading embeddings from {filepath}")
            logger.info(f"📁 Available keys: {list(data.keys())}")

            # Handle both formats
            if "embeddings" in data:
                logger.info("📁 Using 'embeddings' format")
                self.embeddings = data["embeddings"]
                self.project_ids = data["project_ids"]
                self.model_name = data.get("model_name", self.model_name)
                self.projects = data.get("projects", [])
            elif "vector_store" in data:
                logger.info("📁 Using 'vector_store' format")
                self.embeddings = data["vector_store"]
                self.project_ids = list(self.embeddings.keys())
                self.model_name = data.get("model_name", self.model_name)
                # Try to load projects from the same file or from JSON
                if "projects_store" in data:
                    # Extract project data from the dictionary
                    projects_dict = data["projects_store"]
                    self.projects = list(projects_dict.values())  # Get the actual project data
                    logger.info(f"📁 Loaded {len(self.projects)} projects from projects_store")
                else:
                    json_file = filepath.replace('.pkl', '.json')
                    if Path(json_file).exists():
                        with open(json_file, 'r', encoding='utf-8') as f:
                            self.projects = json.load(f)
                        logger.info(f"📁 Loaded {len(self.projects)} projects from JSON fallback")
                    else:
                        self.projects = []
                        logger.warning("⚠️  No projects found in embeddings or JSON")
            else:
                raise KeyError(f"Unknown embedding file format. Available keys: {list(data.keys())}")

            logger.info(f"📁 Successfully loaded embeddings from {filepath}")
            logger.info(f"📁 Loaded {len(self.projects)} projects")
            logger.info(f"📁 Loaded {len(self.embeddings)} embeddings")
            
            # Verify that we have matching data
            if len(self.projects) != len(self.embeddings):
                logger.warning(f"⚠️  Mismatch: {len(self.projects)} projects vs {len(self.embeddings)} embeddings")
            else:
                logger.info("✅ Projects and embeddings count match!")
                
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            self.embeddings = {}
            self.project_ids = []
            self.projects = [] 