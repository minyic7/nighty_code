"""
Output formatters for identity cards and artifacts.

This module provides different formatting options for
exporting identity cards and classification results.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod

from ..identity.schemas import IdentityCard


class BaseFormatter(ABC):
    """Base class for artifact formatters."""
    
    @abstractmethod
    def format(self, cards: List[IdentityCard], metadata: Dict[str, Any] = None) -> str:
        """Format identity cards into a string representation."""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the appropriate file extension for this format."""
        pass


class JsonFormatter(BaseFormatter):
    """JSON formatter for identity cards."""
    
    def format(self, cards: List[IdentityCard], metadata: Dict[str, Any] = None) -> str:
        """Format cards as JSON."""
        if metadata is None:
            metadata = {}
        
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_cards': len(cards),
                'version': '1.0.0',
                'format': 'identity_cards_json',
                **metadata
            },
            'identity_cards': [card.to_dict() for card in cards]
        }
        
        return json.dumps(data, indent=2, default=str)
    
    def get_file_extension(self) -> str:
        return ".json"


class YamlFormatter(BaseFormatter):
    """YAML formatter for identity cards."""
    
    def format(self, cards: List[IdentityCard], metadata: Dict[str, Any] = None) -> str:
        """Format cards as YAML."""
        if metadata is None:
            metadata = {}
        
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_cards': len(cards),
                'version': '1.0.0',
                'format': 'identity_cards_yaml',
                **metadata
            },
            'identity_cards': [card.to_dict() for card in cards]
        }
        
        return yaml.dump(data, default_flow_style=False)
    
    def get_file_extension(self) -> str:
        return ".yaml"


class MarkdownFormatter(BaseFormatter):
    """Markdown formatter for identity cards."""
    
    def format(self, cards: List[IdentityCard], metadata: Dict[str, Any] = None) -> str:
        """Format cards as Markdown."""
        if metadata is None:
            metadata = {}
        
        lines = []
        
        # Header
        title = metadata.get('title', 'Identity Cards Report')
        lines.append(f"# {title}")
        lines.append("")
        
        # Metadata
        lines.append(f"**Generated**: {datetime.now().isoformat()}")
        lines.append(f"**Total Cards**: {len(cards)}")
        
        if 'project_name' in metadata:
            lines.append(f"**Project**: {metadata['project_name']}")
        
        if 'description' in metadata:
            lines.append(f"**Description**: {metadata['description']}")
        
        lines.append("")
        
        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for i, card in enumerate(cards, 1):
            lines.append(f"{i}. [{card.file_name}](#{self._make_anchor(card.file_name)})")
        lines.append("")
        
        # Individual cards
        for i, card in enumerate(cards, 1):
            lines.append(f"## {i}. {card.file_name}")
            lines.append("")
            
            # Basic information table
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            lines.append(f"| Card ID | `{card.card_id}` |")
            lines.append(f"| File Type | {card.file_type.value} |")
            lines.append(f"| Card Type | {card.card_type.value} |")
            lines.append(f"| File Path | `{card.file_path}` |")
            lines.append(f"| Size | {card.file_size_bytes:,} bytes |")
            lines.append(f"| Confidence | {card.classification_confidence:.1%} |")
            lines.append(f"| Complexity | {card.complexity_level.value} |")
            
            if card.detected_frameworks:
                frameworks = [f.value for f in card.detected_frameworks]
                lines.append(f"| Frameworks | {', '.join(frameworks)} |")
            
            lines.append("")
            
            # Quick facts
            if card.quick_facts:
                lines.append("### Quick Facts")
                lines.append("")
                for fact in card.quick_facts:
                    lines.append(f"- {fact}")
                lines.append("")
            
            # Key insights
            if card.key_insights:
                lines.append("### Key Insights")
                lines.append("")
                for insight in card.key_insights:
                    lines.append(f"- {insight}")
                lines.append("")
            
            # Potential issues
            if card.potential_issues:
                lines.append("### Potential Issues")
                lines.append("")
                for issue in card.potential_issues:
                    lines.append(f"- ⚠️ {issue}")
                lines.append("")
            
            # Summary
            if card.llm_summary:
                lines.append("### Summary")
                lines.append("")
                lines.append(card.llm_summary)
                lines.append("")
            
            # LLM context
            lines.append("### LLM Context")
            lines.append("")
            lines.append(f"```")
            lines.append(card.to_llm_context())
            lines.append("```")
            lines.append("")
            
            # Separator
            lines.append("---")
            lines.append("")
        
        return '\n'.join(lines)
    
    def get_file_extension(self) -> str:
        return ".md"
    
    def _make_anchor(self, text: str) -> str:
        """Convert text to markdown anchor format."""
        return text.lower().replace(' ', '-').replace('.', '').replace('_', '-')


class HtmlFormatter(BaseFormatter):
    """HTML formatter for identity cards."""
    
    def format(self, cards: List[IdentityCard], metadata: Dict[str, Any] = None) -> str:
        """Format cards as HTML."""
        if metadata is None:
            metadata = {}
        
        title = metadata.get('title', 'Identity Cards Report')
        
        html_parts = []
        
        # HTML header
        html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .card {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
        .property {{ margin: 5px 0; }}
        .label {{ font-weight: bold; }}
        .frameworks {{ color: #0066cc; }}
        .quick-facts {{ background-color: #e7f3ff; padding: 10px; border-radius: 3px; }}
        .insights {{ background-color: #f0f8e7; padding: 10px; border-radius: 3px; }}
        .issues {{ background-color: #ffeee7; padding: 10px; border-radius: 3px; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 2px; }}
    </style>
</head>
<body>""")
        
        # Header
        html_parts.append(f"    <h1>{title}</h1>")
        
        # Metadata
        html_parts.append('    <div class="metadata">')
        html_parts.append(f'        <p><span class="label">Generated:</span> {datetime.now().isoformat()}</p>')
        html_parts.append(f'        <p><span class="label">Total Cards:</span> {len(cards)}</p>')
        
        if 'project_name' in metadata:
            html_parts.append(f'        <p><span class="label">Project:</span> {metadata["project_name"]}</p>')
        
        html_parts.append('    </div>')
        
        # Cards
        for i, card in enumerate(cards, 1):
            html_parts.append(f'    <div class="card">')
            html_parts.append(f'        <h2>{i}. {card.file_name}</h2>')
            
            # Basic properties
            html_parts.append(f'        <div class="property"><span class="label">Card ID:</span> <code>{card.card_id}</code></div>')
            html_parts.append(f'        <div class="property"><span class="label">File Type:</span> {card.file_type.value}</div>')
            html_parts.append(f'        <div class="property"><span class="label">File Path:</span> <code>{card.file_path}</code></div>')
            html_parts.append(f'        <div class="property"><span class="label">Size:</span> {card.file_size_bytes:,} bytes</div>')
            html_parts.append(f'        <div class="property"><span class="label">Confidence:</span> {card.classification_confidence:.1%}</div>')
            html_parts.append(f'        <div class="property"><span class="label">Complexity:</span> {card.complexity_level.value}</div>')
            
            if card.detected_frameworks:
                frameworks = [f.value for f in card.detected_frameworks]
                html_parts.append(f'        <div class="property"><span class="label">Frameworks:</span> <span class="frameworks">{", ".join(frameworks)}</span></div>')
            
            # Quick facts
            if card.quick_facts:
                html_parts.append('        <div class="quick-facts">')
                html_parts.append('            <h3>Quick Facts</h3>')
                html_parts.append('            <ul>')
                for fact in card.quick_facts:
                    html_parts.append(f'                <li>{fact}</li>')
                html_parts.append('            </ul>')
                html_parts.append('        </div>')
            
            # Key insights
            if card.key_insights:
                html_parts.append('        <div class="insights">')
                html_parts.append('            <h3>Key Insights</h3>')
                html_parts.append('            <ul>')
                for insight in card.key_insights:
                    html_parts.append(f'                <li>{insight}</li>')
                html_parts.append('            </ul>')
                html_parts.append('        </div>')
            
            # Potential issues
            if card.potential_issues:
                html_parts.append('        <div class="issues">')
                html_parts.append('            <h3>Potential Issues</h3>')
                html_parts.append('            <ul>')
                for issue in card.potential_issues:
                    html_parts.append(f'                <li>⚠️ {issue}</li>')
                html_parts.append('            </ul>')
                html_parts.append('        </div>')
            
            # Summary
            if card.llm_summary:
                html_parts.append(f'        <div class="property"><span class="label">Summary:</span> {card.llm_summary}</div>')
            
            html_parts.append('    </div>')
        
        # HTML footer
        html_parts.append("""</body>
</html>""")
        
        return '\n'.join(html_parts)
    
    def get_file_extension(self) -> str:
        return ".html"