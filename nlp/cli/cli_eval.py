import click

from main import NLP
import time


# Command Group
@click.group(name="eval")
def cli_eval():
    """Model evaluation related commands"""
    pass


@cli_eval.command(name="model", help="evaluate model")
@click.argument("model_name_to_load", type=str, default="meta-llama/Llama-2-7b-chat-hf")
@click.argument("query", type=str)
def eval_model(query: str, model_name_to_load: str) -> str:
    click.echo(
        "Evaluating model: " + model_name_to_load + " with query: " + query + "."
    )
    nlp = NLP(
        default_model_name=model_name_to_load,
        #   mistralai/Mistral-7B-v0.1
        models_dir="./nlp/models",
        use_lora=False,
    )
    start = time.time()
    print("Query: ", query)
    response = nlp.generate_response(query)
    end = time.time()
    print("Time it took to generate response: ", end - start)
    click.echo("Response: " + response + ".")
    return response


@cli_eval.command(name="ppo", help="evaluate ppo")
@click.argument("ppo_model_name", type=str, default="my-fine-tuned-model-ppo")
@click.argument("query", type=str)
def eval_ppo_model(query: str, ppo_model_name: str) -> str:
    click.echo("Evaluating model: " + ppo_model_name + " with query: " + query + ".")
    nlp = NLP(
        models_dir="./nlp/models",
        load_model=False,  # Don't instantiate any models when we call the class
        use_lora=True,
    )
    nlp.ppo.load_ppo_model(ppo_model_name, local_only=True, use_lora=True)
    response = nlp.generate_response_from_ppo(query)
    click.echo("Response: " + response + ".")
    return response


if __name__ == "__main__":
    cli_eval()
