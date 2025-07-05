# # This is a sample Python script.

# # Press ⌃R to execute it or replace it with your code.
# # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# import json
# import numpy as np
# from typing import List, Dict, Any, Union
# from pathlib import Path

# class JSONEmbeddingModel:
#     """
#     Embedding model optimized for JSON files with different processing strategies.
#     """
    
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         self.vector_store: Dict[str, np.ndarray] = {}
#         self.metadata_store: Dict[str, Dict[str, Any]] = {}
        
#         # Initialize the embedding model
#         try:
#             from sentence_transformers import SentenceTransformer
#             self.model = SentenceTransformer(model_name)
#         except ImportError:
#             print("Sentence Transformers not available, using fallback method")
#             self.model = None
    
#     def process_json_content(self, json_data: Union[str, Dict, List], strategy: str = "flatten") -> str:
#         """
#         Process JSON content into searchable text based on strategy.
        
#         Strategies:
#         - "flatten": Convert JSON to flat key-value pairs
#         - "structure": Preserve structure with indentation
#         - "values_only": Extract only values
#         - "keys_only": Extract only keys
#         - "mixed": Combine keys and values intelligently
#         """
#         if isinstance(json_data, str):
#             json_data = json.loads(json_data)
        
#         if strategy == "flatten":
#             return self._flatten_json(json_data)
#         elif strategy == "structure":
#             return json.dumps(json_data, indent=2)
#         elif strategy == "values_only":
#             return self._extract_values(json_data)
#         elif strategy == "keys_only":
#             return self._extract_keys(json_data)
#         elif strategy == "mixed":
#             return self._mixed_extraction(json_data)
#         else:
#             return json.dumps(json_data)
    
#     def _flatten_json(self, obj: Any, prefix: str = "") -> str:
#         """Flatten JSON into key-value pairs."""
#         items = []
#         if isinstance(obj, dict):
#             for key, value in obj.items():
#                 new_prefix = f"{prefix}.{key}" if prefix else key
#                 items.append(self._flatten_json(value, new_prefix))
#         elif isinstance(obj, list):
#             for i, value in enumerate(obj):
#                 new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
#                 items.append(self._flatten_json(value, new_prefix))
#         else:
#             items.append(f"{prefix}: {obj}")
#         return " ".join(items)
    
#     def _extract_values(self, obj: Any) -> str:
#         """Extract only values from JSON."""
#         values = []
#         if isinstance(obj, dict):
#             for value in obj.values():
#                 values.extend(self._extract_values(value))
#         elif isinstance(obj, list):
#             for item in obj:
#                 values.extend(self._extract_values(item))
#         else:
#             values.append(str(obj))
#         return " ".join(values)
    
#     def _extract_keys(self, obj: Any) -> str:
#         """Extract only keys from JSON."""
#         keys = []
#         if isinstance(obj, dict):
#             for key in obj.keys():
#                 keys.append(key)
#                 keys.extend(self._extract_keys(obj[key]))
#         elif isinstance(obj, list):
#             for item in obj:
#                 keys.extend(self._extract_keys(item))
#         return " ".join(keys)
    
#     def _mixed_extraction(self, obj: Any) -> str:
#         """Intelligent extraction of both keys and values."""
#         content = []
#         if isinstance(obj, dict):
#             for key, value in obj.items():
#                 content.append(key)
#                 if isinstance(value, (str, int, float, bool)):
#                     content.append(str(value))
#                 else:
#                     content.extend(self._mixed_extraction(value))
#         elif isinstance(obj, list):
#             for item in obj:
#                 content.extend(self._mixed_extraction(item))
#         else:
#             content.append(str(obj))
#         return " ".join(content)
    
#     def create_embedding(self, text: str) -> np.ndarray:
#         """Create embedding using the selected model."""
#         if self.model:
#             return self.model.encode(text)
#         else:
#             # Fallback to simple hash-based embedding
#             return self._fallback_embedding(text)
    
#     def _fallback_embedding(self, text: str) -> np.ndarray:
#         """Simple fallback embedding method."""
#         import hashlib
#         text_hash = hashlib.md5(text.encode()).hexdigest()
#         hash_bytes = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
#         embedding = np.array(hash_bytes[:384], dtype=np.float32)
#         return embedding / np.linalg.norm(embedding)
    
#     def add_json_file(self, filepath: str, strategy: str = "flatten", metadata: Dict[str, Any] = None) -> str:
#         """Add a JSON file to the vector store."""
#         with open(filepath, 'r') as f:
#             json_data = json.load(f)
        
#         processed_text = self.process_json_content(json_data, strategy)
#         embedding = self.create_embedding(processed_text)
        
#         doc_id = f"json_{len(self.vector_store)}"
#         self.vector_store[doc_id] = embedding
#         self.metadata_store[doc_id] = {
#             "filepath": filepath,
#             "strategy": strategy,
#             "processed_text": processed_text[:200] + "..." if len(processed_text) > 200 else processed_text,
#             "metadata": metadata or {}
#         }
        
#         return doc_id
    
#     def search_json(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Search for similar JSON files."""
#         query_embedding = self.create_embedding(query)
        
#         similarities = []
#         for doc_id, doc_embedding in self.vector_store.items():
#             similarity = np.dot(query_embedding, doc_embedding)
#             similarities.append((doc_id, similarity))
        
#         similarities.sort(key=lambda x: x[1], reverse=True)
        
#         results = []
#         for doc_id, similarity in similarities[:top_k]:
#             result = {
#                 "doc_id": doc_id,
#                 "similarity": float(similarity),
#                 "filepath": self.metadata_store[doc_id]["filepath"],
#                 "strategy": self.metadata_store[doc_id]["strategy"],
#                 "processed_text": self.metadata_store[doc_id]["processed_text"],
#                 "metadata": self.metadata_store[doc_id]["metadata"]
#             }
#             results.append(result)
        
#         return results


# # Example usage
# def main():
#     # Initialize with a good model for JSON
#     json_embedding_model = JSONEmbeddingModel("all-MiniLM-L6-v2")
    
#     # Example JSON files (you would load real files)
#     sample_jsons = [
#         {"name": "user1", "age": 25, "skills": ["python", "javascript"]},
#         {"product": "laptop", "price": 999, "specs": {"ram": "16GB", "storage": "512GB"}},
#         {"config": {"api_key": "secret", "endpoint": "https://api.example.com"}}
#     ]
    
#     # Save sample JSONs to files
#     for i, data in enumerate(sample_jsons):
#         with open(f"sample_{i}.json", 'w') as f:
#             json.dump(data, f, indent=2)
    
#     # Add JSON files with different strategies
#     strategies = ["flatten", "structure", "mixed"]
#     for i, strategy in enumerate(strategies):
#         doc_id = json_embedding_model.add_json_file(
#             f"sample_{i}.json", 
#             strategy=strategy,
#             metadata={"category": f"strategy_{strategy}"}
#         )
#         print(f"Added JSON file with {strategy} strategy: {doc_id}")
    
#     # Search for similar content
#     print("\nSearching for 'python programming skills'...")
#     results = json_embedding_model.search_json("python programming skills", top_k=3)
    
#     for i, result in enumerate(results):
#         print(f"\n{i+1}. File: {result['filepath']}")
#         print(f"   Strategy: {result['strategy']}")
#         print(f"   Similarity: {result['similarity']:.4f}")
#         print(f"   Content: {result['processed_text'][:100]}...")


# if __name__ == "__main__":
#     main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import json
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path

class ProjectEmbeddingModel:
    """
    Embedding model specifically optimized for ETHGlobal project data.
    Focuses on matching user queries against short_description fields.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.vector_store: Dict[str, np.ndarray] = {}
        self.projects_store: Dict[str, Dict[str, Any]] = {}
        
        # Initialize the embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded embedding model: {model_name}")
        except ImportError:
            print("❌ Sentence Transformers not available, using fallback method")
            self.model = None
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding using the selected model."""
        if self.model:
            return self.model.encode(text)
        else:
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding method."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_bytes = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        embedding = np.array(hash_bytes[:384], dtype=np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def load_projects_from_json(self, filepath: str) -> None:
        """Load projects from JSON file and create embeddings for short_descriptions."""
        with open(filepath, 'r') as f:
            projects = json.load(f)
        
        print(f"📁 Loading {len(projects)} projects from {filepath}")
        
        for i, project in enumerate(projects):
            # Extract key fields
            project_id = f"project_{i}"
            short_description = project.get('short_description', '')
            project_title = project.get('project_title', '')
            url = project.get('url', '')
            
            # Create embedding from short_description
            embedding = self.create_embedding(short_description)
            
            # Store project data
            self.vector_store[project_id] = embedding
            self.projects_store[project_id] = {
                'project_title': project_title,
                'short_description': short_description,
                'url': url,
                'full_project': project  # Store complete project data
            }
        
        print(f"✅ Created embeddings for {len(self.vector_store)} projects")
    
    def load_projects_from_data(self, projects: List[Dict[str, Any]]) -> None:
        """Load projects directly from data and create embeddings for short_descriptions."""
        print(f"📁 Loading {len(projects)} projects from data")
        
        for i, project in enumerate(projects):
            # Extract key fields
            project_id = f"project_{i}"
            short_description = project.get('short_description', '')
            project_title = project.get('project_title', '')
            url = project.get('url', '')
            
            # Create embedding from short_description
            embedding = self.create_embedding(short_description)
            
            # Store project data
            self.vector_store[project_id] = embedding
            self.projects_store[project_id] = {
                'project_title': project_title,
                'short_description': short_description,
                'url': url,
                'full_project': project  # Store complete project data
            }
        
        print(f"✅ Created embeddings for {len(self.vector_store)} projects")
    
    def search_projects(self, query: str, top_k: int = 5, min_similarity: float = 0.2) -> List[Dict[str, Any]]:
        """
        Search for projects based on user query.
        Returns projects where short_description matches the query.
        """
        query_embedding = self.create_embedding(query)
        
        similarities = []
        for project_id, project_embedding in self.vector_store.items():
            similarity = np.dot(query_embedding, project_embedding)
            similarities.append((project_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum similarity and return top_k results
        results = []
        for project_id, similarity in similarities:
            if similarity >= min_similarity and len(results) < top_k:
                project_data = self.projects_store[project_id]
                result = {
                    'project_id': project_id,
                    'similarity': float(similarity),
                    'project_title': project_data['project_title'],
                    'short_description': project_data['short_description'],
                    'url': project_data['url'],
                    'full_project': project_data['full_project']
                }
                results.append(result)
        
        return results
    
    def analyze_search_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Analyze and explain search results in a user-friendly way."""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 60)
        
        if not results:
            print("❌ No projects found matching your query.")
            print("💡 Try using different keywords or being more specific.")
            return
        
        for i, result in enumerate(results):
            similarity = result['similarity']
            title = result['project_title']
            description = result['short_description']
            
            # Interpret similarity
            if similarity > 0.7:
                match_quality = "🎯 EXCELLENT MATCH"
                emoji = "🔥"
            elif similarity > 0.5:
                match_quality = "✅ GOOD MATCH"
                emoji = "✨"
            elif similarity > 0.3:
                match_quality = "⚠️  MODERATE MATCH"
                emoji = "📋"
            else:
                match_quality = "🔍 WEAK MATCH"
                emoji = "🔎"
            
            print(f"\n{i+1}. {emoji} {match_quality} (Score: {similarity:.3f})")
            print(f"   📛 Title: {title}")
            print(f"   📝 Description: {description}")
            print(f"   🔗 URL: {result['url']}")
            
            # Show why it matched (highlight matching terms)
            matching_terms = self._find_matching_terms(query.lower(), description.lower())
            if matching_terms:
                print(f"   🎯 Matched terms: {', '.join(matching_terms)}")
    
    def _find_matching_terms(self, query: str, description: str) -> List[str]:
        """Find terms that appear in both query and description."""
        query_words = set(query.split())
        desc_words = set(description.split())
        return list(query_words.intersection(desc_words))
    
    def get_project_details(self, project_id: str) -> Dict[str, Any]:
        """Get full details of a specific project."""
        if project_id in self.projects_store:
            return self.projects_store[project_id]['full_project']
        return None
    
    def save_embeddings(self, filepath: str) -> None:
        """Save embeddings to disk for faster loading."""
        import pickle
        data = {
            'vector_store': {k: v.tolist() for k, v in self.vector_store.items()},
            'projects_store': self.projects_store,
            'model_name': self.model_name
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> None:
        """Load embeddings from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vector_store = {k: np.array(v) for k, v in data['vector_store'].items()}
        self.projects_store = data['projects_store']
        self.model_name = data['model_name']
        print(f"📂 Loaded {len(self.vector_store)} project embeddings from {filepath}")


# Example usage with your data structure
def main():
    # Initialize the model
    project_model = ProjectEmbeddingModel("all-MiniLM-L6-v2")
    
    # Try to load saved embeddings first (faster)
    embeddings_file = "project_embeddings.pkl"
    try:
        project_model.load_embeddings(embeddings_file)
        print(f"🚀 Loaded saved embeddings from {embeddings_file}")
        print(f"📊 Ready to search {len(project_model.vector_store)} projects")
    except FileNotFoundError:
        # If no saved embeddings, create them from JSON
        print("📁 No saved embeddings found, creating new ones...")
        try:
            with open('ethglobal_showcase_projects.json', 'r') as f:
                your_projects = json.load(f)
            print(f"📁 Loaded {len(your_projects)} projects from ethglobal_showcase_projects.json")
        except FileNotFoundError:
            print("❌ File 'ethglobal_showcase_projects.json' not found!")
            print("💡 Please create this file with your ETHGlobal projects data")
            return
        
        # Load projects and create embeddings
        project_model.load_projects_from_data(your_projects)
        
        # Save for next time
        project_model.save_embeddings(embeddings_file)
        print(f"💾 Saved embeddings to {embeddings_file}")
        print(f"📁 File location: {Path(embeddings_file).absolute()}")
    
    # Example searches
    test_queries = [
        "AI video content protection",
        "Next.js development tools",
        "mobile app with crypto incentives",
        "blockchain smart contracts",
        "Flutter mobile development"
    ]
    
    for query in test_queries:
        results = project_model.search_projects(query, top_k=3, min_similarity=0.2)
        project_model.analyze_search_results(query, results)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()