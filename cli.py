# CLI for NLP main file
import click

from nlp.cli.cli_train import cli_train
from nlp.cli.cli_eval import cli_eval


@click.group()
def main():
    """CLI for NLP training and evaluating."""
    return


if __name__ == "__main__":
    main.add_command(cli_train)
    main.add_command(cli_eval)
    main()
