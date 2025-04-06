"""Mermaid renderer for flowcharts."""
from typing import Dict, Optional

from .base import Flowchart, Node, Edge

class MermaidRenderer:
    """Renders flowcharts using Mermaid syntax."""
    
    def __init__(
        self,
        direction: str = "TD",
        theme: str = "default",
        node_style: Optional[Dict[str, str]] = None,
        edge_style: Optional[Dict[str, str]] = None
    ):
        """Initialize the renderer.
        
        Args:
            direction: Graph direction (TD, LR, RL, BT)
            theme: Mermaid theme (default, forest, dark, neutral)
            node_style: Default node style attributes
            edge_style: Default edge style attributes
        """
        self.direction = direction
        self.theme = theme
        self.node_style = node_style or {}
        self.edge_style = edge_style or {}
        
    def render(self, flowchart: Flowchart) -> str:
        """Render a flowchart using Mermaid syntax.
        
        Args:
            flowchart: The flowchart to render
            
        Returns:
            A string containing the Mermaid diagram definition
        """
        # Start with graph definition
        lines = [
            f"flowchart {self.direction}",
            f"%% {flowchart.title}",
        ]
        
        if flowchart.description:
            lines.append(f"%% {flowchart.description}")
            
        # Add nodes
        for node in flowchart.nodes:
            # Create node style based on confidence
            style = self._get_node_style(node.confidence)
            
            # Add node definition
            lines.append(
                f'    {node.id}["{node.label}"]{style}'
            )
            
            # Add tooltip if description exists
            if node.description:
                lines.append(
                    f'    click {node.id} tooltip "{node.description}"'
                )
                
        # Add edges
        for edge in flowchart.edges:
            # Create edge style
            style = self._get_edge_style()
            
            # Add edge definition with label
            lines.append(
                f'    {edge.source} -->|"{edge.label}"| {edge.target}{style}'
            )
            
            # Add tooltip if description exists
            if edge.description:
                lines.append(
                    f'    click {edge.source}_{edge.target} tooltip "{edge.description}"'
                )
                
        # Add theme and style definitions
        lines.extend([
            "",
            "%% Theme and styles",
            f"%%{{init: {{'theme': '{self.theme}'}}}}%%",
            "classDef default " + self._style_dict_to_str(self.node_style),
            "classDef highConfidence fill:#90EE90,stroke:#333",  # Light green
            "classDef mediumConfidence fill:#FFE4B5,stroke:#333",  # Light orange
            "classDef lowConfidence fill:#FFB6C1,stroke:#333",  # Light red
        ])
        
        return "\n".join(lines)
        
    def _get_node_style(self, confidence: float) -> str:
        """Get node style based on confidence level.
        
        Args:
            confidence: Confidence value between 0 and 1
            
        Returns:
            A Mermaid style class definition
        """
        if confidence >= 0.8:
            return ":::highConfidence"
        elif confidence >= 0.6:
            return ":::mediumConfidence"
        else:
            return ":::lowConfidence"
            
    def _get_edge_style(self) -> str:
        """Get edge style.
        
        Returns:
            A Mermaid style class definition
        """
        return ":::default"
        
    def _style_dict_to_str(self, style: Dict[str, str]) -> str:
        """Convert a style dictionary to a Mermaid style string.
        
        Args:
            style: Dictionary of style attributes
            
        Returns:
            A Mermaid style string
        """
        if not style:
            return "fill:#f9f9f9,stroke:#333,stroke-width:2px"
            
        return ",".join(f"{k}:{v}" for k, v in style.items()) 