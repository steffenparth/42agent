#!/usr/bin/env python3
"""
Script to create embeddings for ETHGlobal projects
"""

import json
from embedding_model import ProjectEmbeddingModel

def main():
    print("🔍 Loading ETHGlobal projects...")
    
    # Load the projects data
    with open('ethglobal_showcase_projects.json', 'r', encoding='utf-8') as f:
        projects = json.load(f)
    
    print(f"📁 Loaded {len(projects)} projects")
    
    # Create embeddings
    print("🤖 Creating embeddings...")
    model = ProjectEmbeddingModel()
    model.load_projects_from_data(projects)
    
    # Save embeddings
    print("💾 Saving embeddings...")
    model.save_embeddings('project_embeddings.pkl')
    
    print(f"✅ Successfully created embeddings for {len(projects)} projects")
    print("📁 Saved to: project_embeddings.pkl")

if __name__ == "__main__":
    main() 