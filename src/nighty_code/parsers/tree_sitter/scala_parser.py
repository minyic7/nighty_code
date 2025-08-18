"""
Scala parser using tree-sitter to extract AST and convert to JSON.

This module parses Scala files and converts the AST to JSON format
for inspection and further processing into graphs.
"""

import json
from typing import Dict, Any, List
from pathlib import Path

try:
    import tree_sitter_scala as ts_scala
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError as e:
    TREE_SITTER_AVAILABLE = False
    IMPORT_ERROR = e


class ScalaParser:
    """
    Tree-sitter based Scala parser that converts AST to JSON.
    
    The parser extracts the complete AST structure and converts it
    to JSON format, preserving all node types, fields, and relationships.
    """
    
    def __init__(self):
        """Initialize the Scala parser with tree-sitter."""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter-scala is not installed. "
                "Please install it with: pip install tree-sitter-scala"
            ) from IMPORT_ERROR
        
        self.language = Language(ts_scala.language())
        self.parser = Parser(self.language)
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a Scala file and return its AST as JSON.
        
        Args:
            file_path: Path to the Scala file
            
        Returns:
            Dictionary representing the AST in JSON format
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse the source code
            tree = self.parser.parse(bytes(source_code, 'utf-8'))
            
            # Convert AST to JSON
            ast_json = {
                "file_path": str(file_path),
                "language": "scala",
                "ast": self._node_to_dict(tree.root_node, source_code)
            }
            
            return ast_json
            
        except Exception as e:
            return {
                "file_path": str(file_path),
                "language": "scala",
                "error": str(e),
                "ast": None
            }
    
    def parse_string(self, source_code: str, filename: str = "unnamed.scala") -> Dict[str, Any]:
        """
        Parse Scala source code string and return AST as JSON.
        
        Args:
            source_code: Scala source code as string
            filename: Optional filename for reference
            
        Returns:
            Dictionary representing the AST in JSON format
        """
        try:
            # Parse the source code
            tree = self.parser.parse(bytes(source_code, 'utf-8'))
            
            # Convert AST to JSON
            ast_json = {
                "file_path": filename,
                "language": "scala",
                "ast": self._node_to_dict(tree.root_node, source_code)
            }
            
            return ast_json
            
        except Exception as e:
            return {
                "file_path": filename,
                "language": "scala",
                "error": str(e),
                "ast": None
            }
    
    def _node_to_dict(self, node: Node, source_code: str) -> Dict[str, Any]:
        """
        Recursively convert a tree-sitter Node to a dictionary.
        
        Args:
            node: Tree-sitter Node object
            source_code: Original source code for text extraction
            
        Returns:
            Dictionary representation of the node and its children
        """
        # Basic node information
        node_dict = {
            "type": node.type,
            "start_position": {
                "row": node.start_point[0],
                "column": node.start_point[1]
            },
            "end_position": {
                "row": node.end_point[0],
                "column": node.end_point[1]
            },
            "start_byte": node.start_byte,
            "end_byte": node.end_byte
        }
        
        # Add text content for leaf nodes or important nodes
        if not node.children or node.type in self._important_node_types():
            text = source_code[node.start_byte:node.end_byte]
            # Limit text length for readability
            if len(text) <= 200:
                node_dict["text"] = text
            else:
                node_dict["text"] = text[:200] + "..."
                node_dict["text_truncated"] = True
        
        # Process named fields and children together
        fields = {}
        children = []
        
        for i, child in enumerate(node.children):
            child_dict = self._node_to_dict(child, source_code)
            
            # Check if this child has a field name
            field_name = node.field_name_for_child(i)
            if field_name:
                child_dict["is_field"] = True
                child_dict["field_of_parent"] = field_name
                fields[field_name] = child_dict
            
            children.append(child_dict)
        
        if fields:
            node_dict["fields"] = fields
        
        if children:
            node_dict["children"] = children
        
        # Add some semantic information for specific node types
        node_dict.update(self._extract_semantic_info(node, source_code))
        
        return node_dict
    
    def _important_node_types(self) -> List[str]:
        """
        List of node types whose text should always be preserved.
        
        Returns:
            List of important node type names
        """
        return [
            "identifier",
            "string_literal",
            "integer_literal",
            "float_literal",
            "boolean_literal",
            "comment",
            "import_declaration",
            "package_clause"
        ]
    
    def _extract_semantic_info(self, node: Node, source_code: str) -> Dict[str, Any]:
        """
        Extract additional semantic information based on node type.
        
        Args:
            node: Tree-sitter Node
            source_code: Source code for text extraction
            
        Returns:
            Dictionary with additional semantic information
        """
        semantic_info = {}
        
        # Add semantic tags based on node type
        if node.type == "class_definition":
            semantic_info["semantic_type"] = "class"
            semantic_info["is_entity"] = True
            
        elif node.type == "object_definition":
            semantic_info["semantic_type"] = "object"
            semantic_info["is_entity"] = True
            
        elif node.type == "trait_definition":
            semantic_info["semantic_type"] = "trait"
            semantic_info["is_entity"] = True
            
        elif node.type == "function_definition":
            semantic_info["semantic_type"] = "method"
            semantic_info["is_entity"] = True
            
        elif node.type in ["val_definition", "var_definition"]:
            semantic_info["semantic_type"] = "field"
            semantic_info["is_mutable"] = node.type == "var_definition"
            
        elif node.type == "import_declaration":
            semantic_info["semantic_type"] = "import"
            
        elif node.type == "extends_clause":
            semantic_info["semantic_type"] = "inheritance"
            
        elif node.type == "call_expression":
            semantic_info["semantic_type"] = "method_call"
            
        elif node.type == "instance_expression":
            semantic_info["semantic_type"] = "instantiation"
        
        return semantic_info
    
    def save_ast_json(self, ast_dict: Dict[str, Any], output_path: Path) -> None:
        """
        Save AST dictionary to a JSON file.
        
        Args:
            ast_dict: AST dictionary to save
            output_path: Path where to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ast_dict, f, indent=2, ensure_ascii=False)


def parse_scala_file(file_path: Path) -> Dict[str, Any]:
    """
    Convenience function to parse a Scala file to JSON.
    
    Args:
        file_path: Path to Scala file
        
    Returns:
        AST as JSON dictionary
    """
    parser = ScalaParser()
    return parser.parse_file(file_path)


def parse_scala_string(source_code: str, filename: str = "unnamed.scala") -> Dict[str, Any]:
    """
    Convenience function to parse Scala source string to JSON.
    
    Args:
        source_code: Scala source code
        filename: Optional filename for reference
        
    Returns:
        AST as JSON dictionary
    """
    parser = ScalaParser()
    return parser.parse_string(source_code, filename)