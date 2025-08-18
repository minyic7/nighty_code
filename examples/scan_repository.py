"""
Example script demonstrating folder scanning functionality (placeholder).

This is a placeholder example showing how the FolderScanner will work
when implemented. Currently all scanner implementations raise NotImplementedError.
"""

from pathlib import Path
import sys
from rich.console import Console

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nighty_code.core import (
    FolderScanner, 
    ScannerConfig,
    scan_folder
)


def main():
    """Main function to demonstrate scanner usage."""
    console = Console()
    
    console.print("[bold red]⚠️  This is a placeholder example![/bold red]")
    console.print("All scanner implementations currently raise NotImplementedError.")
    console.print("This shows the intended API when scanners are implemented.\n")
    
    # Get repository path from command line or use current directory
    if len(sys.argv) > 1:
        repo_path = Path(sys.argv[1])
    else:
        repo_path = Path.cwd()
    
    console.print(f"[bold blue]Would scan folder:[/bold blue] {repo_path}")
    console.print()
    
    # Show how configuration would work
    config_path = repo_path / 'config' / 'default.yaml'
    if config_path.exists():
        console.print(f"[green]✓[/green] Would load configuration from {config_path}")
        try:
            config = ScannerConfig.from_yaml(config_path)
            console.print(f"  Max file size: {config.max_file_size_mb} MB")
            console.print(f"  Follow symlinks: {config.follow_symlinks}")
            console.print(f"  Ignore patterns: {len(config.ignore_patterns)} patterns")
        except Exception as e:
            console.print(f"[red]✗[/red] Error loading config: {e}")
    else:
        console.print("[yellow]![/yellow] Would use default configuration")
    
    console.print()
    
    # Show API usage (but don't actually call it)
    console.print("[bold]Example API usage when implemented:[/bold]")
    console.print("""
[cyan]# Create scanner[/cyan]
scanner = FolderScanner(config)

[cyan]# Scan directory[/cyan]
files = scanner.scan(repo_path)

[cyan]# Or use convenience function[/cyan]
files = scan_folder(repo_path)

[cyan]# Get statistics[/cyan]
stats = scanner.get_statistics(files)
    """)
    
    console.print("[yellow]To implement:[/yellow]")
    console.print("1. Add dependencies: gitignore-parser, etc.")
    console.print("2. Implement FolderScanner.scan_iterator()")
    console.print("3. Add directory traversal with os.walk()")
    console.print("4. Add .gitignore support")
    console.print("5. Add file filtering and size checks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())