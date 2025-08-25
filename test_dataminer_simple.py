#!/usr/bin/env python3
"""
Simple DataMiner test with a general extraction model.
"""

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Set the config file to use GenAI
os.environ['LLM_CONFIG_PATH'] = 'config/llm_genai.yaml'

from src.dataminer import (
    DataMinerClient,
    ExtractionConfig,
    ProcessingMode,
    ExtractionSchema,
)
from src.dataminer.core.config import DataMinerConfig
from src.llm import get_llm_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create artifacts directory
ARTIFACTS_DIR = Path("dataminer_artifacts_simple")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Simple General Model
# ============================================================================

class GeneralProjectAnalysis(ExtractionSchema):
    """General project analysis model"""
    project_name: str = Field(description="Name of the project or main folder")
    project_type: str = Field(description="Type of project (e.g., Python library, web app, CLI tool)")
    main_purpose: str = Field(description="Main purpose or functionality in 1-2 sentences")
    main_technologies: List[str] = Field(default_factory=list, description="Main technologies/frameworks used")
    folder_structure: Dict[str, str] = Field(default_factory=dict, description="Key folders and their purposes")
    entry_points: List[str] = Field(default_factory=list, description="Main entry points or scripts")
    dependencies_count: int = Field(default=0, description="Approximate number of dependencies")
    has_documentation: bool = Field(default=False, description="Whether documentation exists")
    estimated_complexity: str = Field(default="medium", description="Project complexity: simple/medium/complex")
    interesting_findings: List[str] = Field(default_factory=list, description="Any interesting or unique aspects")
    
    def get_required_fields(self) -> List[str]:
        return ["project_name", "project_type", "main_purpose"]
    
    def get_confidence_fields(self) -> Dict[str, float]:
        return {
            "project_name": 1.0,
            "project_type": 0.9,
            "main_purpose": 0.8,
            "main_technologies": 0.85,
            "folder_structure": 0.7
        }

# ============================================================================
# Helper Functions
# ============================================================================

def save_artifact(filename: str, data: Any, format: str = "json"):
    """Save extraction artifact to file"""
    filepath = ARTIFACTS_DIR / filename
    
    if format == "json":
        with open(filepath, 'w', encoding='utf-8') as f:
            if hasattr(data, 'model_dump'):
                json.dump(data.model_dump(), f, indent=2, default=str)
            elif hasattr(data, '__dict__'):
                json.dump(data.__dict__, f, indent=2, default=str)
            else:
                json.dump(data, f, indent=2, default=str)
    elif format == "md":
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(data))
    
    print(f"  💾 Saved: {filepath.name}")
    return filepath

def generate_analysis_report(result: Dict[str, Any]) -> str:
    """Generate a markdown report from the analysis"""
    report = []
    report.append("# Project Analysis Report\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Status**: {'✅ Success' if result.get('success') else '❌ Failed'}\n")
    
    if result.get('success') and result.get('data'):
        data = result['data']
        report.append(f"**Confidence**: {result.get('confidence', 0):.1%}\n")
        report.append(f"**Processing Time**: {result.get('processing_time', 0):.2f}s\n\n")
        
        report.append("## 📋 Project Overview\n")
        report.append(f"- **Name**: {data.get('project_name', 'Unknown')}\n")
        report.append(f"- **Type**: {data.get('project_type', 'Unknown')}\n")
        report.append(f"- **Purpose**: {data.get('main_purpose', 'Not specified')}\n")
        report.append(f"- **Complexity**: {data.get('estimated_complexity', 'Unknown')}\n")
        report.append(f"- **Has Documentation**: {'Yes' if data.get('has_documentation') else 'No'}\n")
        report.append(f"- **Dependencies**: ~{data.get('dependencies_count', 0)}\n\n")
        
        if data.get('main_technologies'):
            report.append("## 🛠️ Technologies\n")
            for tech in data['main_technologies']:
                report.append(f"- {tech}\n")
            report.append("\n")
        
        if data.get('folder_structure'):
            report.append("## 📁 Project Structure\n")
            for folder, purpose in data['folder_structure'].items():
                report.append(f"- **{folder}/**: {purpose}\n")
            report.append("\n")
        
        if data.get('entry_points'):
            report.append("## 🚀 Entry Points\n")
            for entry in data['entry_points']:
                report.append(f"- {entry}\n")
            report.append("\n")
        
        if data.get('interesting_findings'):
            report.append("## 💡 Interesting Findings\n")
            for finding in data['interesting_findings']:
                report.append(f"- {finding}\n")
            report.append("\n")
    else:
        report.append(f"\n**Error**: {result.get('error', 'Unknown error')}\n")
    
    return ''.join(report)

def gather_project_content(project_path: Path) -> str:
    """Gather relevant content from the project for analysis"""
    content_parts = []
    
    # Project overview
    content_parts.append(f"Analyzing project at: {project_path.name}")
    content_parts.append(f"Full path: {project_path}\n")
    
    # List top-level contents
    content_parts.append("Top-level contents:")
    for item in sorted(project_path.iterdir())[:20]:  # Limit to first 20 items
        if item.is_dir():
            content_parts.append(f"  📁 {item.name}/")
        else:
            content_parts.append(f"  📄 {item.name}")
    
    # Check for README
    readme_files = list(project_path.glob("README*"))
    if readme_files:
        readme = readme_files[0]
        content_parts.append(f"\n{readme.name} content (first 1500 chars):")
        content_parts.append(readme.read_text()[:1500])
    
    # Check for package.json or setup.py or pyproject.toml
    for config_file in ["package.json", "setup.py", "pyproject.toml", "Cargo.toml"]:
        config_path = project_path / config_file
        if config_path.exists():
            content_parts.append(f"\n{config_file} content (first 800 chars):")
            content_parts.append(config_path.read_text()[:800])
            break
    
    # Check src directory
    src_path = project_path / "src"
    if src_path.exists():
        content_parts.append("\n/src directory structure:")
        for item in sorted(src_path.iterdir())[:15]:
            if item.is_dir():
                # Count files in subdirectory
                py_count = len(list(item.glob("*.py")))
                content_parts.append(f"  📁 {item.name}/ ({py_count} .py files)")
                # Check for __init__.py
                init_file = item / "__init__.py"
                if init_file.exists():
                    init_content = init_file.read_text()[:300]
                    if init_content.strip():
                        content_parts.append(f"    __init__.py preview: {init_content[:100]}...")
            else:
                content_parts.append(f"  📄 {item.name}")
    
    # List Python files in root
    py_files = list(project_path.glob("*.py"))
    if py_files:
        content_parts.append(f"\nPython files in root: {', '.join(f.name for f in py_files[:10])}")
    
    # Check for test files
    test_files = list(project_path.glob("test*.py")) + list(project_path.glob("**/test*.py"))
    if test_files:
        content_parts.append(f"\nTest files found: {len(test_files)}")
        content_parts.append(f"Examples: {', '.join(f.name for f in test_files[:5])}")
    
    # Check config directory
    config_path = project_path / "config"
    if config_path.exists():
        config_files = list(config_path.glob("*"))
        content_parts.append(f"\nConfig files: {', '.join(f.name for f in config_files[:10])}")
    
    return "\n".join(content_parts)

# ============================================================================
# Main Test Function
# ============================================================================

async def analyze_project(client: DataMinerClient, project_path: Path) -> Dict:
    """Analyze the project with general model"""
    print(f"\n🔍 Analyzing Project: {project_path.name}")
    print("=" * 50)
    start_time = datetime.now()
    
    # Gather project content
    print("📊 Gathering project information...")
    content = gather_project_content(project_path)
    
    # Save the raw content for reference
    save_artifact("raw_content.txt", content, format="txt")
    
    print("🤖 Running extraction...")
    # Extract with general model
    result = await client.extract(
        content=content,
        schema=GeneralProjectAnalysis,
        mode=ProcessingMode.FAST  # Use fast mode for general analysis
    )
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return_data = {
        "success": result.success,
        "confidence": result.confidence.overall if result.success else 0,
        "data": result.data.model_dump() if result.success else None,
        "errors": result.errors,
        "processing_time": processing_time,
        "mode": "FAST"
    }
    
    if result.success:
        print(f"✅ Extraction successful!")
        print(f"  📊 Confidence: {result.confidence.overall:.1%}")
        print(f"  ⏱️  Time: {processing_time:.2f}s")
        
        # Save JSON artifact
        save_artifact("project_analysis.json", result.data)
        
        # Display key findings
        print(f"\n📋 Key Findings:")
        print(f"  • Project: {result.data.project_name}")
        print(f"  • Type: {result.data.project_type}")
        print(f"  • Purpose: {result.data.main_purpose}")
        print(f"  • Technologies: {', '.join(result.data.main_technologies[:5])}")
        print(f"  • Complexity: {result.data.estimated_complexity}")
    else:
        print(f"❌ Extraction failed: {result.errors}")
    
    return return_data

# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Main test runner"""
    print("=" * 60)
    print("🚀 SIMPLE DATAMINER TEST - GENERAL PROJECT ANALYSIS")
    print("=" * 60)
    print()
    
    # Get current project directory
    project_path = Path.cwd()
    print(f"📁 Analyzing current directory: {project_path}")
    print(f"🔧 LLM Config: {os.environ.get('LLM_CONFIG_PATH', 'default')}")
    print(f"💾 Artifacts directory: {ARTIFACTS_DIR}/")
    print()
    
    try:
        # Configure DataMiner with simpler settings
        config = DataMinerConfig(
            enable_result_cache=True,
            cache_directory=Path("./cache"),
            log_level="INFO"
        )
        
        # Relaxed settings for general analysis
        config.extraction.min_confidence_threshold = 0.3
        config.extraction.preferred_provider = "genai"
        config.extraction.enable_mcp_tools = False
        config.extraction.use_copilot_reasoning = False
        
        # Initialize DataMiner
        print("⚙️  Initializing DataMiner...")
        client = DataMinerClient(config=config)
        
        # Set up LLM manager
        llm_manager = await get_llm_manager()
        client.llm_manager = llm_manager
        
        await client.initialize()
        
        # Set LLM manager for strategies
        for strategy in client.strategies.values():
            strategy.llm_manager = llm_manager
        
        print("✅ DataMiner initialized successfully")
        
        # Run the analysis
        result = await analyze_project(client, project_path)
        
        # Generate and save report
        print("\n📝 Generating report...")
        report = generate_analysis_report(result)
        save_artifact("analysis_report.md", report, format="md")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "project_path": str(project_path),
            "project_name": project_path.name,
            "success": result['success'],
            "confidence": result.get('confidence', 0),
            "processing_time": result.get('processing_time', 0),
            "llm_provider": "genai",
            "extraction_mode": result.get('mode', 'FAST'),
            "dataminer_stats": {
                "total_extractions": client.total_extractions,
                "cache_hits": client.cache_hits,
                "cache_misses": client.cache_misses
            }
        }
        save_artifact("metadata.json", metadata)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 ANALYSIS COMPLETE")
        print("=" * 60)
        
        if result['success']:
            print("✅ Status: SUCCESS")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
        else:
            print("❌ Status: FAILED")
            print(f"Error: {result.get('errors', 'Unknown error')}")
        
        print(f"\n📁 Artifacts saved to: {ARTIFACTS_DIR.absolute()}/")
        print("\nGenerated files:")
        for file in sorted(ARTIFACTS_DIR.glob("*")):
            size = file.stat().st_size
            print(f"  • {file.name} ({size:,} bytes)")
        
        print("\n✨ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error details
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        save_artifact("error_log.json", error_data)

if __name__ == "__main__":
    asyncio.run(main())