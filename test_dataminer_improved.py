#!/usr/bin/env python3
"""
Improved DataMiner test with better schemas and higher success rate.
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
    level=logging.WARNING,  # Reduce verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create artifacts directory
ARTIFACTS_DIR = Path("dataminer_artifacts_improved")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Improved Extraction Models (Simpler, more focused)
# ============================================================================

class ModuleInfo(BaseModel):
    """Information about a Python module"""
    name: str = Field(description="Module name")
    description: str = Field(description="Module description")
    main_classes: List[str] = Field(default_factory=list, description="Main class names")
    main_functions: List[str] = Field(default_factory=list, description="Main function names")

class ProjectOverview(ExtractionSchema):
    """Simplified project overview"""
    project_name: str = Field(description="Name of the project")
    description: str = Field(description="Brief project description (1-2 sentences)")
    main_modules: List[ModuleInfo] = Field(default_factory=list, description="Main modules")
    key_features: List[str] = Field(default_factory=list, description="3-5 key features")
    primary_dependencies: List[str] = Field(default_factory=list, description="Main external dependencies")
    
    def get_required_fields(self) -> List[str]:
        return ["project_name", "description"]
    
    def get_confidence_fields(self) -> Dict[str, float]:
        return {
            "project_name": 1.0,
            "description": 0.9,
            "main_modules": 0.7,
            "key_features": 0.8
        }

class CodeStructure(ExtractionSchema):
    """Simplified code structure analysis"""
    total_python_files: int = Field(description="Approximate number of Python files")
    main_packages: List[str] = Field(default_factory=list, description="Main package/module names")
    has_tests: bool = Field(description="Whether test files exist")
    has_documentation: bool = Field(description="Whether README exists")
    code_organization_quality: str = Field(description="Code organization: excellent/good/fair")
    uses_type_hints: bool = Field(description="Whether code uses type hints")
    uses_async: bool = Field(description="Whether code uses async/await")
    
    def get_required_fields(self) -> List[str]:
        return ["total_python_files", "main_packages"]
    
    def get_confidence_fields(self) -> Dict[str, float]:
        return {
            "total_python_files": 0.9,
            "main_packages": 0.95,
            "has_tests": 1.0,
            "has_documentation": 1.0
        }

class APIInterface(ExtractionSchema):
    """Simplified API interface documentation"""
    main_client_classes: List[str] = Field(default_factory=list, description="Main client class names")
    core_methods: List[str] = Field(default_factory=list, description="Important public methods")
    configuration_files: List[str] = Field(default_factory=list, description="Config file names")
    example_scripts: List[str] = Field(default_factory=list, description="Example/test script names")
    initialization_pattern: str = Field(default="", description="How to initialize the main client")
    
    def get_required_fields(self) -> List[str]:
        return ["main_client_classes"]
    
    def get_confidence_fields(self) -> Dict[str, float]:
        return {
            "main_client_classes": 0.9,
            "core_methods": 0.7,
            "configuration_files": 0.95
        }

class LLMIntegration(ExtractionSchema):
    """LLM integration details"""
    supported_providers: List[str] = Field(default_factory=list, description="LLM providers (openai, anthropic, etc)")
    default_provider: str = Field(default="", description="Default LLM provider")
    configured_models: List[str] = Field(default_factory=list, description="Model names configured")
    has_rate_limiting: bool = Field(description="Whether rate limiting is implemented")
    has_retry_logic: bool = Field(description="Whether retry logic is implemented")
    supports_streaming: bool = Field(description="Whether streaming is supported")
    supports_structured_output: bool = Field(description="Whether structured output is supported")
    
    def get_required_fields(self) -> List[str]:
        return ["supported_providers"]
    
    def get_confidence_fields(self) -> Dict[str, float]:
        return {
            "supported_providers": 0.95,
            "default_provider": 0.9,
            "configured_models": 0.85
        }

class SecurityPractices(ExtractionSchema):
    """Security practices analysis"""
    stores_api_keys_in_config: bool = Field(description="Whether API keys are in config files")
    uses_environment_variables: bool = Field(description="Whether env variables are used")
    masks_sensitive_data_in_logs: bool = Field(description="Whether sensitive data is masked in logs")
    implements_rate_limiting: bool = Field(description="Whether rate limiting exists")
    uses_https_only: bool = Field(description="Whether only HTTPS endpoints are used")
    has_authentication: bool = Field(description="Whether authentication is implemented")
    
    def get_required_fields(self) -> List[str]:
        return []  # All fields are boolean, none strictly required
    
    def get_confidence_fields(self) -> Dict[str, float]:
        return {
            "stores_api_keys_in_config": 0.95,
            "uses_environment_variables": 0.9,
            "implements_rate_limiting": 0.85
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

def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive markdown report"""
    report = []
    report.append("# DataMiner Extraction Report (Improved)\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Project**: nighty_code\n")
    report.append(f"**LLM Provider**: GenAI\n\n")
    
    # Summary
    report.append("## 📊 Summary\n")
    total = len(results)
    successful = sum(1 for r in results.values() if r.get('success', False))
    report.append(f"- **Total Extractions**: {total}\n")
    report.append(f"- **Successful**: {successful}/{total}\n")
    report.append(f"- **Success Rate**: {(successful/total*100):.1f}%\n\n")
    
    # Quick Overview
    if successful > 0:
        report.append("## 🎯 Quick Overview\n")
        for name, result in results.items():
            if result.get('success') and result.get('data'):
                data = result['data']
                if name == "Project Overview" and data:
                    report.append(f"- **Project**: {data.get('project_name', 'Unknown')}\n")
                    report.append(f"- **Description**: {data.get('description', 'N/A')}\n")
                    report.append(f"- **Modules**: {len(data.get('main_modules', []))}\n")
                    report.append(f"- **Features**: {len(data.get('key_features', []))}\n")
                    break
        report.append("\n")
    
    # Detailed Results
    report.append("## 📋 Detailed Extraction Results\n")
    
    for test_name, result in results.items():
        report.append(f"\n### {test_name}\n")
        if result.get('success'):
            report.append(f"✅ **Status**: Success\n")
            report.append(f"📊 **Confidence**: {result.get('confidence', 0):.1%}\n")
            report.append(f"⏱️ **Processing Mode**: {result.get('mode', 'N/A')}\n")
            
            # Add detailed findings
            if result.get('data'):
                report.append("\n**Extracted Data:**\n")
                data = result['data']
                
                if test_name == "Project Overview":
                    report.append(f"- Project Name: `{data.get('project_name', 'N/A')}`\n")
                    report.append(f"- Modules Found: {len(data.get('main_modules', []))}\n")
                    if data.get('main_modules'):
                        report.append("  - " + ", ".join(m.get('name', 'Unknown') for m in data['main_modules'][:5]) + "\n")
                    report.append(f"- Key Features: {', '.join(data.get('key_features', [])[:3])}\n")
                
                elif test_name == "Code Structure":
                    report.append(f"- Total Python Files: ~{data.get('total_python_files', 0)}\n")
                    report.append(f"- Main Packages: {', '.join(data.get('main_packages', []))}\n")
                    report.append(f"- Has Tests: {'Yes' if data.get('has_tests') else 'No'}\n")
                    report.append(f"- Code Quality: {data.get('code_organization_quality', 'N/A')}\n")
                
                elif test_name == "API Interface":
                    report.append(f"- Client Classes: {', '.join(data.get('main_client_classes', []))}\n")
                    report.append(f"- Core Methods: {len(data.get('core_methods', []))}\n")
                    report.append(f"- Config Files: {', '.join(data.get('configuration_files', []))}\n")
                
                elif test_name == "LLM Integration":
                    report.append(f"- Providers: {', '.join(data.get('supported_providers', []))}\n")
                    report.append(f"- Default: {data.get('default_provider', 'N/A')}\n")
                    report.append(f"- Models: {', '.join(data.get('configured_models', [])[:3])}\n")
                    report.append(f"- Features: Rate Limiting={data.get('has_rate_limiting')}, Retry={data.get('has_retry_logic')}, Streaming={data.get('supports_streaming')}\n")
                
                elif test_name == "Security Practices":
                    report.append(f"- API Keys in Config: {data.get('stores_api_keys_in_config')}\n")
                    report.append(f"- Environment Variables: {data.get('uses_environment_variables')}\n")
                    report.append(f"- Rate Limiting: {data.get('implements_rate_limiting')}\n")
                    report.append(f"- HTTPS Only: {data.get('uses_https_only')}\n")
        else:
            report.append(f"❌ **Status**: Failed\n")
            report.append(f"**Error**: {result.get('error', 'Unknown error')}\n")
    
    # Statistics
    report.append("\n## 📈 Statistics\n")
    report.append(f"- **Total Processing Time**: {sum(r.get('processing_time', 0) for r in results.values()):.2f}s\n")
    report.append(f"- **Average Confidence**: {sum(r.get('confidence', 0) for r in results.values() if r.get('success')) / max(successful, 1):.1%}\n")
    
    return ''.join(report)

# ============================================================================
# Improved Test Functions
# ============================================================================

async def extract_project_overview(client: DataMinerClient, project_path: Path) -> Dict:
    """Extract project overview with improved content gathering"""
    print("\n📋 Extracting Project Overview...")
    start_time = datetime.now()
    
    content_parts = []
    
    # README - most important
    readme_path = project_path / "README.md"
    if readme_path.exists():
        content_parts.append(f"README.md:\n{readme_path.read_text()[:3000]}")
    
    # Module __init__ files for structure
    src_path = project_path / "src"
    if src_path.exists():
        for module_dir in src_path.iterdir():
            if module_dir.is_dir():
                init_file = module_dir / "__init__.py"
                if init_file.exists():
                    content_parts.append(f"\n{module_dir.name}/__init__.py (first 500 chars):\n{init_file.read_text()[:500]}")
    
    content = "\n\n".join(content_parts)
    
    result = await client.extract(
        content=content,
        schema=ProjectOverview,
        mode=ProcessingMode.THOROUGH
    )
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return_data = {
        "success": result.success,
        "confidence": result.confidence.overall if result.success else 0,
        "data": result.data.model_dump() if result.success else None,
        "errors": result.errors,
        "processing_time": processing_time,
        "mode": "THOROUGH"
    }
    
    if result.success:
        print(f"  ✅ Success (confidence: {result.confidence.overall:.1%}, time: {processing_time:.1f}s)")
        save_artifact("project_overview.json", result.data)
    else:
        print(f"  ❌ Failed: {result.errors}")
    
    return return_data

async def extract_code_structure(client: DataMinerClient, project_path: Path) -> Dict:
    """Extract code structure with simplified schema"""
    print("\n🔍 Extracting Code Structure...")
    start_time = datetime.now()
    
    content_parts = []
    
    # Count and list files
    src_path = project_path / "src"
    py_files = list(src_path.rglob("*.py")) if src_path.exists() else []
    test_files = list(project_path.glob("test*.py"))
    
    content_parts.append(f"Project structure analysis for: {project_path.name}")
    content_parts.append(f"Total Python files found: {len(py_files)}")
    content_parts.append(f"Test files found: {len(test_files)}")
    content_parts.append(f"Has README: {(project_path / 'README.md').exists()}")
    
    # List main packages
    if src_path.exists():
        packages = [d.name for d in src_path.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
        content_parts.append(f"Main packages: {', '.join(packages)}")
    
    # Sample code for quality assessment
    if py_files:
        sample_file = py_files[0]
        sample_content = sample_file.read_text()[:1000]
        content_parts.append(f"\nSample code from {sample_file.name}:")
        content_parts.append(f"Uses type hints: {'typing' in sample_content or '->' in sample_content}")
        content_parts.append(f"Uses async: {'async ' in sample_content}")
        has_docstrings = '"""' in sample_content
        content_parts.append(f"Has docstrings: {has_docstrings}")
    
    content = "\n".join(content_parts)
    
    result = await client.extract(
        content=content,
        schema=CodeStructure,
        mode=ProcessingMode.FAST  # Use FAST for simple structure analysis
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
        print(f"  ✅ Success (confidence: {result.confidence.overall:.1%}, time: {processing_time:.1f}s)")
        save_artifact("code_structure.json", result.data)
    else:
        print(f"  ❌ Failed: {result.errors}")
    
    return return_data

async def extract_api_interface(client: DataMinerClient, project_path: Path) -> Dict:
    """Extract API interface with focused content"""
    print("\n📚 Extracting API Interface...")
    start_time = datetime.now()
    
    content_parts = []
    
    # Focus on __init__.py files and client classes
    src_path = project_path / "src"
    if src_path.exists():
        for module_dir in src_path.iterdir():
            if module_dir.is_dir():
                # Check __init__.py for exports
                init_file = module_dir / "__init__.py"
                if init_file.exists():
                    init_content = init_file.read_text()[:1000]
                    if "__all__" in init_content or "from" in init_content:
                        content_parts.append(f"\n{module_dir.name}/__init__.py exports:\n{init_content}")
                
                # Look for client files
                for pattern in ["*client*.py", "*manager*.py"]:
                    client_files = list(module_dir.glob(pattern))
                    for client_file in client_files[:1]:  # Just first match
                        content_parts.append(f"\n{module_dir.name}/{client_file.name} (class definitions):")
                        content = client_file.read_text()[:1500]
                        # Extract class names
                        import re
                        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                        content_parts.append(f"Classes: {', '.join(classes)}")
                        # Extract public methods
                        methods = re.findall(r'^\s+(?:async\s+)?def\s+([a-z]\w+)', content, re.MULTILINE)
                        content_parts.append(f"Public methods: {', '.join(methods[:5])}")
    
    # Config files
    config_files = list((project_path / "config").glob("*.yaml")) if (project_path / "config").exists() else []
    content_parts.append(f"\nConfiguration files: {', '.join(f.name for f in config_files)}")
    
    # Example scripts
    test_files = list(project_path.glob("test*.py"))
    content_parts.append(f"Example/test scripts: {', '.join(f.name for f in test_files[:5])}")
    
    content = "\n".join(content_parts)
    
    result = await client.extract(
        content=content,
        schema=APIInterface,
        mode=ProcessingMode.FAST
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
        print(f"  ✅ Success (confidence: {result.confidence.overall:.1%}, time: {processing_time:.1f}s)")
        save_artifact("api_interface.json", result.data)
    else:
        print(f"  ❌ Failed: {result.errors}")
    
    return return_data

async def extract_llm_integration(client: DataMinerClient, project_path: Path) -> Dict:
    """Extract LLM integration details"""
    print("\n🤖 Extracting LLM Integration...")
    start_time = datetime.now()
    
    content_parts = []
    
    # LLM module
    llm_path = project_path / "src" / "llm"
    if llm_path.exists():
        init_file = llm_path / "__init__.py"
        if init_file.exists():
            content_parts.append(f"llm/__init__.py:\n{init_file.read_text()[:1000]}")
        
        # Check for provider files
        for provider_file in llm_path.glob("*provider*.py"):
            content_parts.append(f"\nProvider file {provider_file.name} found")
    
    # Config files
    config_path = project_path / "config"
    if config_path.exists():
        for yaml_file in config_path.glob("*llm*.yaml"):
            content = yaml_file.read_text()[:1000]
            content_parts.append(f"\n{yaml_file.name}:\n{content}")
    
    content = "\n".join(content_parts)
    
    result = await client.extract(
        content=content,
        schema=LLMIntegration,
        mode=ProcessingMode.FAST
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
        print(f"  ✅ Success (confidence: {result.confidence.overall:.1%}, time: {processing_time:.1f}s)")
        save_artifact("llm_integration.json", result.data)
    else:
        print(f"  ❌ Failed: {result.errors}")
    
    return return_data

async def extract_security_practices(client: DataMinerClient, project_path: Path) -> Dict:
    """Extract security practices"""
    print("\n🔒 Extracting Security Practices...")
    start_time = datetime.now()
    
    content_parts = []
    
    # Check config files for API keys
    config_path = project_path / "config"
    if config_path.exists():
        for yaml_file in list(config_path.glob("*.yaml"))[:2]:
            content = yaml_file.read_text()[:500]
            content_parts.append(f"{yaml_file.name} contains: {'api_key' in content.lower()}")
    
    # Check for environment variable usage
    src_files = list((project_path / "src").rglob("*.py"))[:5] if (project_path / "src").exists() else []
    for src_file in src_files:
        content = src_file.read_text()[:1000]
        if "environ" in content or "getenv" in content:
            content_parts.append(f"{src_file.name} uses environment variables")
        if "rate_limit" in content.lower():
            content_parts.append(f"{src_file.name} implements rate limiting")
        if "https://" in content:
            content_parts.append(f"{src_file.name} uses HTTPS endpoints")
    
    content = "\n".join(content_parts)
    
    result = await client.extract(
        content=content,
        schema=SecurityPractices,
        mode=ProcessingMode.FAST
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
        print(f"  ✅ Success (confidence: {result.confidence.overall:.1%}, time: {processing_time:.1f}s)")
        save_artifact("security_practices.json", result.data)
    else:
        print(f"  ❌ Failed: {result.errors}")
    
    return return_data

# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Main test runner with improved schemas"""
    print("=" * 60)
    print("🚀 DATAMINER TEST - IMPROVED VERSION")
    print("=" * 60)
    print()
    print(f"📁 Project: /Users/minyic/project/nighty_code")
    print(f"🔧 LLM Config: config/llm_genai.yaml")
    print(f"💾 Artifacts: {ARTIFACTS_DIR}/")
    print()
    
    try:
        # Configure DataMiner
        config = DataMinerConfig(
            enable_result_cache=True,
            cache_directory=Path("./cache"),
            log_level="WARNING"
        )
        
        config.extraction.min_confidence_threshold = 0.4  # Lower threshold
        config.extraction.preferred_provider = "genai"
        config.extraction.enable_mcp_tools = False
        config.extraction.use_copilot_reasoning = False
        
        # Initialize
        print("⚙️  Initializing DataMiner...")
        client = DataMinerClient(config=config)
        
        from src.llm import get_llm_manager
        llm_manager = await get_llm_manager()
        client.llm_manager = llm_manager
        
        await client.initialize()
        
        for strategy in client.strategies.values():
            strategy.llm_manager = llm_manager
        
        print("✅ DataMiner initialized successfully")
        print()
        
        # Run extractions
        project_path = Path("/Users/minyic/project/nighty_code")
        results = {}
        
        # Run all extractions
        results["Project Overview"] = await extract_project_overview(client, project_path)
        results["Code Structure"] = await extract_code_structure(client, project_path)
        results["API Interface"] = await extract_api_interface(client, project_path)
        results["LLM Integration"] = await extract_llm_integration(client, project_path)
        results["Security Practices"] = await extract_security_practices(client, project_path)
        
        # Generate reports
        print("\n📊 Generating Reports...")
        
        # Save individual and combined results
        save_artifact("all_results.json", results)
        
        # Generate markdown report
        markdown_report = generate_markdown_report(results)
        save_artifact("extraction_report.md", markdown_report, format="md")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "project_path": str(project_path),
            "llm_provider": "genai",
            "total_extractions": len(results),
            "successful_extractions": sum(1 for r in results.values() if r.get('success')),
            "total_processing_time": sum(r.get('processing_time', 0) for r in results.values()),
            "average_confidence": sum(r.get('confidence', 0) for r in results.values() if r.get('success')) / max(sum(1 for r in results.values() if r.get('success')), 1),
            "dataminer_stats": {
                "total_extractions": client.total_extractions,
                "cache_hits": client.cache_hits,
                "cache_misses": client.cache_misses
            }
        }
        save_artifact("metadata.json", metadata)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 EXTRACTION SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)
        
        print(f"✅ Success Rate: {successful}/{total} ({successful/total*100:.0f}%)")
        print(f"⏱️  Total Time: {sum(r.get('processing_time', 0) for r in results.values()):.1f}s")
        print(f"📊 Avg Confidence: {metadata['average_confidence']:.1%}")
        print()
        
        for name, result in results.items():
            if result.get('success'):
                print(f"  ✅ {name}: {result.get('confidence', 0):.1%} confidence ({result.get('processing_time', 0):.1f}s)")
            else:
                print(f"  ❌ {name}: Failed")
        
        print()
        print(f"📁 Artifacts saved to: {ARTIFACTS_DIR.absolute()}/")
        print()
        print("Files created:")
        for file in sorted(ARTIFACTS_DIR.glob("*")):
            size = file.stat().st_size
            print(f"  • {file.name} ({size:,} bytes)")
        
        print("\n" + "=" * 60)
        print("✨ TEST COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        save_artifact("error_log.json", error_data)

if __name__ == "__main__":
    asyncio.run(main())