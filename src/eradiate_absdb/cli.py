from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated

from pathlib import Path
import typer
from rich.logging import RichHandler

logger = logging.getLogger("eradiate_absdb")
app = typer.Typer()


class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


@app.callback()
def cli(
    log_level: Annotated[
        LogLevel, typer.Option(help="Set log level.")
    ] = LogLevel.WARNING,
    debug: Annotated[
        bool,
        typer.Option(
            help="Enable debug mode. This will notably print exceptions with locals."
        ),
    ] = False,
):
    if debug:
        app.pretty_exceptions_enable = True

    logging.basicConfig(
        level=log_level.name,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def check(
    path: Annotated[Path, typer.Argument(help="Path to the checked database.")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Database spectral mode.")],
    fix: Annotated[
        bool, typer.Option("--fix", "-f", help="Fix issues that can be.")
    ] = False,
):
    """
    Check data for availability and integrity, optionally fix them.
    """
    from eradiate_absdb.core import get_absdb_type

    try:
        cls = get_absdb_type(mode)
    except ValueError:
        logger.critical(f"Unsupported mode '{mode}'")
        exit(1)

    logger.info(f"Opening '{path}'")
    try:
        cls.from_directory(path, fix=fix)
        logger.info("Success!")
    except FileNotFoundError:
        pass


def main():
    app()
