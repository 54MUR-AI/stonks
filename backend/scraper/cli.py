import click
import asyncio
import json
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from src.scraper_manager import ScraperManager
from src.config import settings
from pathlib import Path


console = Console()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('url')
@click.option('--output', '-o', type=click.Path(), help='Output file path (JSON)')
@click.option('--no-summary', is_flag=True, help='Skip summarization')
def scrape(url: str, output: str, no_summary: bool):
    asyncio.run(_scrape(url, output, no_summary))


async def _scrape(url: str, output: Optional[str], no_summary: bool):
    console.print(f"\n[bold blue]Scraping:[/bold blue] {url}\n")
    
    manager = ScraperManager()
    
    try:
        if no_summary:
            content = await manager.scrape(url)
            _display_content(content)
            
            if output:
                _save_content(content, output)
        else:
            content, summary = await manager.scrape_and_summarize(url)
            _display_summary(summary, content)
            
            if output:
                _save_summary(summary, content, output)
        
        console.print("\n[bold green]✓ Success![/bold green]\n")
    
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}\n")
        raise click.Abort()


def _display_content(content):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    
    table.add_row("Title", content.title or "N/A")
    table.add_row("Type", content.content_type)
    table.add_row("Author", content.author or "N/A")
    table.add_row("URL", content.url)
    table.add_row("Content Length", f"{len(content.content)} characters")
    
    console.print(table)
    console.print("\n[bold]Content Preview:[/bold]")
    console.print(Panel(content.content[:500] + "..." if len(content.content) > 500 else content.content))


def _display_summary(summary, content):
    console.print(Panel(f"[bold]{summary.title or 'Untitled'}[/bold]", style="bold blue"))
    
    console.print("\n[bold cyan]Summary:[/bold cyan]")
    console.print(Panel(summary.summary, border_style="green"))
    
    if summary.key_points:
        console.print("\n[bold cyan]Key Points:[/bold cyan]")
        for i, point in enumerate(summary.key_points, 1):
            console.print(f"  {i}. {point}")
    
    console.print(f"\n[dim]Content Type: {summary.content_type}[/dim]")
    console.print(f"[dim]Word Count: {summary.word_count}[/dim]")
    console.print(f"[dim]Model: {summary.model_used}[/dim]")


def _save_content(content, output_path: str):
    data = content.model_dump(mode='json')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    console.print(f"\n[green]Content saved to:[/green] {output_path}")


def _save_summary(summary, content, output_path: str):
    data = {
        "summary": summary.model_dump(mode='json'),
        "original_content": content.model_dump(mode='json')
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    console.print(f"\n[green]Summary saved to:[/green] {output_path}")


@cli.command()
@click.argument('urls_file', type=click.File('r'))
@click.option('--output-dir', '-o', type=click.Path(), default='output', help='Output directory')
def batch(urls_file, output_dir: str):
    asyncio.run(_batch(urls_file, output_dir))


async def _batch(urls_file, output_dir: str):
    urls = [line.strip() for line in urls_file if line.strip()]
    
    console.print(f"\n[bold blue]Processing {len(urls)} URLs...[/bold blue]\n")
    
    manager = ScraperManager()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, url in enumerate(urls, 1):
        console.print(f"[{i}/{len(urls)}] {url}")
        
        try:
            content, summary = await manager.scrape_and_summarize(url)
            
            filename = f"summary_{i}.json"
            output_path = Path(output_dir) / filename
            _save_summary(summary, content, str(output_path))
            
            console.print(f"  [green]✓ Saved to {filename}[/green]")
        
        except Exception as e:
            console.print(f"  [red]✗ Error: {str(e)}[/red]")
        
        console.print()
    
    console.print("[bold green]Batch processing complete![/bold green]\n")


if __name__ == '__main__':
    cli()
