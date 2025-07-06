#!/usr/bin/env python3
"""
ETHGlobal Showcase Projects Client with Agentverse Integration

A client for interacting with scraped ETHGlobal showcase projects data.
Supports data loading, filtering, searching, and Agentverse API integration.
Now includes semantic search using embedding models.
"""

import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import os
from pydantic import Field

# Import our embedding model
from embedding_model import ProjectEmbeddingModel

# Optional Fetch.ai imports - only if uagents is available
try:
    from uagents import Agent, Context, Model, Protocol
    from uagents_core.contrib.protocols.chat import (
        ChatMessage, ChatAcknowledgement, TextContent, chat_protocol_spec
    )
    FETCH_AI_AVAILABLE = True
except ImportError:
    FETCH_AI_AVAILABLE = False
    print("Fetch.ai uagents not available. Install with: pip install uagents==0.22.5")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agentverse API Configuration
AGENTVERSE_API_KEY = "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3NTQzNTgzMjgsImlhdCI6MTc1MTc2NjMyOCwiaXNzIjoiZmV0Y2guYWkiLCJqdGkiOiI1NzA2MTRmNmE2Y2JmY2YyMmJiMTg3MDQiLCJzY29wZSI6ImF2Iiwic3ViIjoiYmQ3YmNkNDViMDFmMDJjZTUyN2I3NTBkNDllZjE3OGEyNzJiMTI2MDVhMTVhYjJiIn0.mrDZGgizFCI7ikH_4Fegu29usX2PD2URZAafXwCVuzQtj6ZuwFd1B8t2KnYKVWvFy3w-6QUOnFr710JX3TGC1iIaKAdJMNH9ScjdAUC9dfjVe0WGKjN1vziLZMSc_67_7lKEH3XhrsifHR33gWOP9ip-aLR0b0v8D9jZUyRETkCI8aNF1w0mX-qkco6KP-oP7gEhEiRs36ItswiQeOfglVfgdjDiDnXMcK3o4xCGFcpTewjNmHillu41p4q2LRpkAZBOQrVKVv-wv5lP8TCXUI0I5GwNo4mBCXnE_S-2w9g21QOxpjQO9Ih3Lb6_yCJNSfO5oXcgpullf4xrcah-Lw"
AGENTVERSE_BASE_URL = "https://agentverse.fetch.ai"

class ProjectCategory(Enum):
    """Categories for ETHGlobal projects"""
    DEFI = "DeFi"
    NFT = "NFT"
    GAMING = "Gaming"
    INFRASTRUCTURE = "Infrastructure"
    SOCIAL = "Social"
    TOOLS = "Tools"
    OTHER = "Other"

@dataclass
class Project:
    """Data class for ETHGlobal project data"""
    url: str
    project_title: str
    short_description: str
    created_at: str
    project_description: str
    how_its_made: str
    thumbnail: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "project_title": self.project_title,
            "short_description": self.short_description,
            "created_at": self.created_at,
            "project_description": self.project_description,
            "how_its_made": self.how_its_made,
            "thumbnail": self.thumbnail
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create Project from dictionary"""
        # Handle the case where data has a 'full_project' field
        if 'full_project' in data and isinstance(data['full_project'], dict):
            # Use the full_project data
            project_data = data['full_project']
        else:
            # Use the data directly
            project_data = data
            
        # Extract only the fields that the Project class expects
        return cls(
            url=project_data.get('url', ''),
            project_title=project_data.get('project_title', ''),
            short_description=project_data.get('short_description', ''),
            created_at=project_data.get('created_at', ''),
            project_description=project_data.get('project_description', ''),
            how_its_made=project_data.get('how_its_made', ''),
            thumbnail=project_data.get('thumbnail', '')
        )

class AgentverseClient:
    """Client for interacting with Agentverse API"""
    
    def __init__(self, api_key: str = AGENTVERSE_API_KEY):
        self.api_key = api_key
        self.base_url = AGENTVERSE_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def search_agents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for agents in Agentverse"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/agents/search"
            params = {
                "query": query,
                "limit": limit
            }
            
            try:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("agents", [])
                    else:
                        logger.error(f"Agentverse search failed: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Error searching agents: {e}")
                return []
    
    async def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific agent"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/agents/{agent_id}"
            
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to get agent details: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Error getting agent details: {e}")
                return None
    
    async def send_message_to_agent(self, agent_id: str, message: str) -> Optional[Dict[str, Any]]:
        """Send a message to a specific agent"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/agents/{agent_id}/chat"
            payload = {
                "message": message
            }
            
            try:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to send message: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                return None

class ETHGlobalClient:
    """Client for interacting with ETHGlobal showcase projects data with semantic search"""
    
    def __init__(self, data_file: str = "ethglobal_showcase_projects.json"):
        self.data_file = Path(data_file)
        self.projects: List[Project] = []
        self.agentverse_client = AgentverseClient()
        
        # Initialize embedding model
        self.embedding_model = ProjectEmbeddingModel()
        
        # PRIMARY: Try to load saved embeddings first (treat as main database)
        embeddings_file = Path("project_embeddings.pkl")
        if embeddings_file.exists():
            try:
                self.embedding_model.load_embeddings(str(embeddings_file))
                logger.info("📁 Loaded saved embeddings")
                
                # Use projects from embeddings as primary source
                if self.embedding_model.projects:
                    try:
                        # Convert project data to Project objects, handling different formats
                        self.projects = []
                        for project_data in self.embedding_model.projects:
                            if isinstance(project_data, dict):
                                try:
                                    self.projects.append(Project.from_dict(project_data))
                                except Exception as e:
                                    logger.warning(f"⚠️  Skipping invalid project dict: {e}")
                            elif isinstance(project_data, str):
                                # Try to parse JSON string
                                try:
                                    import json
                                    parsed_data = json.loads(project_data)
                                    self.projects.append(Project.from_dict(parsed_data))
                                except Exception as e:
                                    logger.warning(f"⚠️  Skipping invalid project string: {e}")
                            else:
                                logger.warning(f"⚠️  Skipping invalid project data type: {type(project_data)}")
                        
                        logger.info(f"📁 Loaded {len(self.projects)} projects from embeddings database")
                        logger.info(f"📁 Loaded {len(self.embedding_model.embeddings)} embeddings")
                        return  # Success! Exit here, don't load JSON
                    except Exception as e:
                        logger.error(f"❌ Error converting project data: {e}")
                        logger.warning("⚠️  Falling back to JSON file due to conversion error")
                else:
                    logger.warning("⚠️  Embeddings loaded but no projects found in embeddings file")
            except Exception as e:
                logger.error(f"❌ Error loading embeddings: {e}")
        
        # FALLBACK: Only load JSON if embeddings failed or don't exist
        logger.info("🔄 Falling back to JSON file...")
        self._load_projects_data()
        if self.projects:
            logger.info("📁 Creating new embeddings from JSON data...")
            self.embedding_model.load_projects_from_data([p.to_dict() for p in self.projects])
        else:
            logger.error("❌ No projects loaded from either embeddings or JSON file")
    
    def _load_projects_data(self) -> None:
        """Load project data from JSON file"""
        try:
            if not self.data_file.exists():
                logger.warning(f"Data file {self.data_file} not found. Run scrape_projects.py first.")
                return
            
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.projects = [Project.from_dict(project_data) for project_data in data]
            logger.info(f"Loaded {len(self.projects)} projects from {self.data_file}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.projects = []
    
    def _load_data_and_create_embeddings(self) -> None:
        """Load project data and create embeddings (legacy method)"""
        self._load_projects_data()
        if self.projects:
            self.embedding_model.load_projects_from_data([p.to_dict() for p in self.projects])
    
    def _load_data(self) -> None:
        """Load project data from JSON file (legacy method)"""
        self._load_projects_data()
    
    def save_data(self, filename: Optional[str] = None) -> None:
        """Save current projects to JSON file"""
        if filename:
            output_file = Path(filename)
        else:
            output_file = self.data_file
        
        try:
            data = [project.to_dict() for project in self.projects]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.projects)} projects to {output_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def search_projects(self, query: str, top_k: int = 5, min_similarity: float = 0.2) -> List[Project]:
        """Search projects using semantic similarity"""
        # Use embedding model for semantic search
        search_results = self.embedding_model.search_projects(query, top_k, min_similarity)
        
        # Convert to Project objects
        results = []
        for result in search_results:
            project_data = result["project"]
            project = Project.from_dict(project_data)
            results.append(project)
        
        return results
    
    def search_projects_simple(self, query: str, case_sensitive: bool = False) -> List[Project]:
        """Simple text-based search (legacy method)"""
        if not case_sensitive:
            query = query.lower()
        
        results = []
        for project in self.projects:
            searchable_text = f"{project.project_title} {project.short_description} {project.project_description} {project.how_its_made}"
            if not case_sensitive:
                searchable_text = searchable_text.lower()
            
            if query in searchable_text:
                results.append(project)
        
        return results
    
    def filter_by_category(self, category: ProjectCategory) -> List[Project]:
        """Filter projects by category based on keywords"""
        category_keywords = {
            ProjectCategory.DEFI: ["defi", "decentralized finance", "swap", "lending", "yield", "amm", "dex"],
            ProjectCategory.NFT: ["nft", "non-fungible", "token", "collection", "mint"],
            ProjectCategory.GAMING: ["game", "gaming", "play", "player", "quest", "reward"],
            ProjectCategory.INFRASTRUCTURE: ["infrastructure", "protocol", "layer", "bridge", "oracle", "indexer"],
            ProjectCategory.SOCIAL: ["social", "community", "chat", "forum", "profile", "friends"],
            ProjectCategory.TOOLS: ["tool", "utility", "dashboard", "analytics", "monitoring", "development"]
        }
        
        keywords = category_keywords.get(category, [])
        if not keywords:
            return self.projects
        
        results = []
        for project in self.projects:
            searchable_text = f"{project.project_title} {project.short_description} {project.project_description}".lower()
            if any(keyword in searchable_text for keyword in keywords):
                results.append(project)
        
        return results
    
    def get_recent_projects(self, days: int = 30) -> List[Project]:
        """Get projects created within the last N days"""
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            # This is a simplified approach - you might want to parse actual dates
            return self.projects[:100]  # Return recent projects (assuming they're ordered)
        except Exception as e:
            logger.error(f"Error filtering by date: {e}")
            return []
    
    def get_project_by_url(self, url: str) -> Optional[Project]:
        """Get a specific project by URL"""
        for project in self.projects:
            if project.url == url:
                return project
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the projects"""
        if not self.projects:
            return {"total_projects": 0}
        
        stats = {
            "total_projects": len(self.projects),
            "categories": {},
            "avg_title_length": sum(len(p.project_title) for p in self.projects) / len(self.projects),
            "avg_description_length": sum(len(p.project_description) for p in self.projects) / len(self.projects),
        }
        
        # Count by category
        for category in ProjectCategory:
            category_projects = self.filter_by_category(category)
            stats["categories"][category.value] = len(category_projects)
        
        return stats
    
    def export_to_csv(self, filename: str) -> None:
        """Export projects to CSV file"""
        try:
            import csv
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['url', 'project_title', 'short_description', 'created_at', 
                             'project_description', 'how_its_made', 'thumbnail']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for project in self.projects:
                    writer.writerow(project.to_dict())
            
            logger.info(f"Exported {len(self.projects)} projects to {filename}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    async def find_related_agents(self, project: Project, limit: int = 5) -> List[Dict[str, Any]]:
        """Find related agents in Agentverse based on project content"""
        # Create a search query based on project content
        search_terms = [
            project.project_title,
            project.short_description[:100],  # First 100 chars of description
            " ".join([word for word in project.project_description.split()[:20]])  # First 20 words
        ]
        
        search_query = " ".join(search_terms)
        agents = await self.agentverse_client.search_agents(search_query, limit)
        return agents

# Fetch.ai Agent Integration (Optional)
if FETCH_AI_AVAILABLE:
    class ProjectSearchMessage(Model):
        """Message model for project search requests"""
        query: str = Field(description="Search query for projects")
        category: Optional[str] = Field(default=None, description="Optional category filter")
        limit: Optional[int] = Field(default=10, description="Maximum number of results")
    
    class ProjectSearchResponse(Model):
        """Message model for project search responses"""
        projects: List[Dict[str, Any]] = Field(description="List of matching projects")
        total_found: int = Field(description="Total number of projects found")
        query: str = Field(description="Original search query")
    
    class AgentverseSearchMessage(Model):
        """Message model for Agentverse search requests"""
        query: str = Field(description="Search query for agents")
        limit: Optional[int] = Field(default=5, description="Maximum number of results")
    
    class AgentverseSearchResponse(Model):
        """Message model for Agentverse search responses"""
        agents: List[Dict[str, Any]] = Field(description="List of matching agents")
        total_found: int = Field(description="Total number of agents found")
        query: str = Field(description="Original search query")
    
    class ETHGlobalAgent(Agent):
        """Fetch.ai agent for ETHGlobal project search with Agentverse integration"""
        
        def __init__(self, client: ETHGlobalClient):
            super().__init__(
                name="ethglobal_search_agent",
                seed="b7f8c2d9e5a4f1c3d6e7b8a9c0f2e1d4",  # new random seed
                port=8001,
                endpoint=["http://localhost:8001/submit"],
                mailbox=True
            )
            self.client = client
            self.setup_protocols()
        
        def setup_protocols(self):
            """Setup agent protocols"""
            
            @self.on_message(model=ProjectSearchMessage)
            async def handle_project_search(ctx: Context, sender: str, msg: ProjectSearchMessage):
                """Handle project search requests"""
                try:
                    # Perform search
                    if msg.category:
                        try:
                            category = ProjectCategory(msg.category)
                            results = self.client.filter_by_category(category)
                            # Further filter by query if provided
                            if msg.query:
                                results = [p for p in results if msg.query.lower() in 
                                         f"{p.project_title} {p.short_description}".lower()]
                        except ValueError:
                            results = self.client.search_projects(msg.query)
                    else:
                        results = self.client.search_projects(msg.query)
                    
                    # Limit results
                    if msg.limit:
                        results = results[:msg.limit]
                    
                    # Convert to dict format
                    project_dicts = [p.to_dict() for p in results]
                    
                    response = ProjectSearchResponse(
                        projects=project_dicts,
                        total_found=len(results),
                        query=msg.query
                    )
                    
                    await ctx.send(sender, response)
                    ctx.logger.info(f"Search completed: {len(results)} projects found for query '{msg.query}'")
                    
                except Exception as e:
                    ctx.logger.error(f"Error in project search: {e}")
                    error_response = ProjectSearchResponse(
                        projects=[],
                        total_found=0,
                        query=msg.query
                    )
                    await ctx.send(sender, error_response)
            
            @self.on_message(model=AgentverseSearchMessage)
            async def handle_agentverse_search(ctx: Context, sender: str, msg: AgentverseSearchMessage):
                """Handle Agentverse search requests"""
                try:
                    agents = await self.client.agentverse_client.search_agents(msg.query, msg.limit or 5)
                    
                    response = AgentverseSearchResponse(
                        agents=agents,
                        total_found=len(agents),
                        query=msg.query
                    )
                    
                    await ctx.send(sender, response)
                    ctx.logger.info(f"Agentverse search completed: {len(agents)} agents found for query '{msg.query}'")
                    
                except Exception as e:
                    ctx.logger.error(f"Error in Agentverse search: {e}")
                    error_response = AgentverseSearchResponse(
                        agents=[],
                        total_found=0,
                        query=msg.query
                    )
                    await ctx.send(sender, error_response)
            
            @self.on_event("startup")
            async def startup(ctx: Context):
                ctx.logger.info(f"ETHGlobal Search Agent started: {self.address}")
                ctx.logger.info(f"Loaded {len(self.client.projects)} projects")
                ctx.logger.info("Agentverse integration enabled")
            
            @self.on_event("shutdown")
            async def shutdown(ctx: Context):
                ctx.logger.info("ETHGlobal Search Agent shutting down...")

# Example usage and CLI interface
async def main_async():
    """Async main function for Agentverse operations"""
    client = ETHGlobalClient()
    
    # Example: Search for agents related to DeFi
    print("Searching for DeFi agents in Agentverse...")
    agents = await client.agentverse_client.search_agents("DeFi", 5)
    
    print(f"Found {len(agents)} DeFi agents:")
    for i, agent in enumerate(agents, 1):
        print(f"{i}. {agent.get('name', 'Unknown')} - {agent.get('description', 'No description')[:100]}...")
    
    # Example: Find related agents for a project
    if client.projects:
        project = client.projects[0]
        print(f"\nFinding related agents for: {project.project_title}")
        related_agents = await client.find_related_agents(project, 3)
        
        print(f"Found {len(related_agents)} related agents:")
        for i, agent in enumerate(related_agents, 1):
            print(f"{i}. {agent.get('name', 'Unknown')}")

def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETHGlobal Showcase Projects Client with Agentverse Integration")
    parser.add_argument("--data-file", default="ethglobal_showcase_projects.json", 
                       help="Path to JSON data file")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--category", choices=[c.value for c in ProjectCategory], 
                       help="Filter by category")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--export-csv", help="Export to CSV file")
    parser.add_argument("--start-agent", action="store_true", help="Start Fetch.ai agent")
    parser.add_argument("--agentverse-search", help="Search for agents in Agentverse")
    parser.add_argument("--async-demo", action="store_true", help="Run async Agentverse demo")
    
    args = parser.parse_args()
    
    # Initialize client
    client = ETHGlobalClient(args.data_file)
    
    if args.async_demo:
        asyncio.run(main_async())
    
    elif args.agentverse_search:
        async def search_agents():
            agents = await client.agentverse_client.search_agents(args.agentverse_search, 10)
            print(f"\n=== Agentverse Search Results for '{args.agentverse_search}' ===")
            print(f"Found {len(agents)} agents:")
            for i, agent in enumerate(agents, 1):
                print(f"\n{i}. {agent.get('name', 'Unknown')}")
                print(f"   Description: {agent.get('description', 'No description')[:200]}...")
                print(f"   ID: {agent.get('id', 'Unknown')}")
        
        asyncio.run(search_agents())
    
    elif args.stats:
        stats = client.get_statistics()
        print("\n=== ETHGlobal Projects Statistics ===")
        print(f"Total Projects: {stats['total_projects']}")
        print(f"Average Title Length: {stats['avg_title_length']:.1f} characters")
        print(f"Average Description Length: {stats['avg_description_length']:.1f} characters")
        print("\nProjects by Category:")
        for category, count in stats['categories'].items():
            print(f"  {category}: {count}")
    
    elif args.search:
        if args.category:
            try:
                category = ProjectCategory(args.category)
                results = client.filter_by_category(category)
                # Further filter by search query
                results = [p for p in results if args.search.lower() in 
                          f"{p.project_title} {p.short_description}".lower()]
            except ValueError:
                results = client.search_projects(args.search)
        else:
            results = client.search_projects(args.search)
        
        print(f"\n=== Search Results for '{args.search}' ===")
        print(f"Found {len(results)} projects:")
        for i, project in enumerate(results[:10], 1):  # Show first 10
            print(f"\n{i}. {project.project_title}")
            print(f"   URL: {project.url}")
            print(f"   Description: {project.short_description[:100]}...")
    
    elif args.export_csv:
        client.export_to_csv(args.export_csv)
    
    elif args.start_agent:
        if not FETCH_AI_AVAILABLE:
            print("Fetch.ai not available. Install with: pip install uagents==0.22.5")
            return
        
        agent = ETHGlobalAgent(client)
        print(f"Starting ETHGlobal Search Agent on {agent.address}")
        print("Agentverse integration enabled")
        print("Use Ctrl+C to stop the agent")
        
        try:
            agent.run()
        except KeyboardInterrupt:
            print("\nAgent stopped by user")
    
    else:
        # Default: show recent projects
        print("\n=== Recent ETHGlobal Projects ===")
        recent_projects = client.get_recent_projects()
        for i, project in enumerate(recent_projects[:5], 1):
            print(f"\n{i}. {project.project_title}")
            print(f"   URL: {project.url}")
            print(f"   Description: {project.short_description[:100]}...")
        
        print("\n=== Available Commands ===")
        print("python client.py --stats                    # Show project statistics")
        print("python client.py --search 'defi'            # Search projects")
        print("python client.py --category 'DeFi'          # Filter by category")
        print("python client.py --agentverse-search 'defi' # Search Agentverse agents")
        print("python client.py --async-demo               # Run Agentverse demo")
        print("python client.py --start-agent              # Start Fetch.ai agent")

if __name__ == "__main__":
    main() 