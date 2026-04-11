"""
CLI for contractllm. Usage:
    contractllm diff summarise_article v1 v2
    contractllm list
    contractllm runs summarise_article --version v1 --limit 10
"""

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    typer = None


def _missing_cli_dependencies() -> SystemExit:
    return SystemExit(
        "The contractllm CLI requires optional dependencies.\n"
        "Install them with: pip install contractllm[cli]\n"
        "For local development: pip install -e \".[cli]\""
    )


if typer is None:

    def app() -> None:
        raise _missing_cli_dependencies()

else:
    app = typer.Typer(
        name="contractllm",
        help="Manage and diff LLM prompt contracts",
        no_args_is_help=True,
    )
    console = Console()

    @app.command()
    def diff(
        contract_name: str = typer.Argument(
            help="Contract name, e.g. 'summarise_article'"
        ),
        version_a: str = typer.Argument(help="First version, e.g. 'v1'"),
        version_b: str = typer.Argument(help="Second version, e.g. 'v2'"),
    ):
        """
        Compare two versions of a contract.
        Shows schema changes and output divergence across stored runs.
        """
        from contractllm.store.version_store import VersionStore

        store = VersionStore()

        def_a = store.get_definition(contract_name, version_a)
        def_b = store.get_definition(contract_name, version_b)

        if not def_a:
            console.print(f"[red]No contract found: {contract_name} {version_a}[/red]")
            raise typer.Exit(1)
        if not def_b:
            console.print(f"[red]No contract found: {contract_name} {version_b}[/red]")
            raise typer.Exit(1)

        schema_changed = def_a.schema_hash != def_b.schema_hash

        console.print(
            Panel(
                f"[bold]{contract_name}[/bold]: {version_a} -> {version_b}",
                title="Contract Diff",
            )
        )

        table = Table(show_header=True)
        table.add_column("Property", style="dim")
        table.add_column(version_a)
        table.add_column(version_b)

        table.add_row(
            "Schema hash",
            def_a.schema_hash,
            def_b.schema_hash,
            style="red" if schema_changed else "green",
        )
        table.add_row("Model", def_a.model, def_b.model)
        table.add_row("Provider", def_a.provider, def_b.provider)

        console.print(table)

        if schema_changed:
            console.print("[yellow]Schema changed between versions.[/yellow]")
        else:
            console.print("[green]Schema unchanged.[/green]")

    @app.command()
    def list_contracts():
        """List all registered contracts and their versions."""
        from contractllm.store.version_store import VersionStore

        store = VersionStore()
        contracts = store.list_all()

        table = Table(title="Registered Contracts")
        table.add_column("Name")
        table.add_column("Version")
        table.add_column("Schema Hash")
        table.add_column("Provider")
        table.add_column("Model")

        for contract in contracts:
            table.add_row(
                contract.name,
                contract.version,
                contract.schema_hash,
                contract.provider,
                contract.model,
            )

        console.print(table)


if __name__ == "__main__":
    app()
