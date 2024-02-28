import click

from nlp.main import NLP


# Command Group
@click.group(name="train")
def cli_train():
    """Train related commands"""
    pass


@cli_train.command(name="ppo", help="ppo train")
@click.argument("query", type=str, default="This movie was really")
@click.argument("model_name_to_load", type=str, default="meta-llama/Llama-2-7b-chat-hf")
@click.argument("ppo_model_name_to_save", type=str, default=None, required=False)
# @click.option('--test1', default='1', help='test option')
def train_ppo(query, model_name_to_load: str, ppo_model_name_to_save: str) -> str:
    nlp = NLP(
        default_model_name=model_name_to_load,
        #   mistralai/Mistral-7B-v0.1
        models_dir="./nlp/models",
        load_model=False,  # Don't instantiate any models when we call the class
        use_lora=True,
    )
    ppo_name = nlp.ppo.train_ppo(
        query,
        local_only=False,
        model_name=nlp.model_name,
        ppo_model_name=ppo_model_name_to_save,
    )
    click.echo("Trained PPO model at: " + ppo_name + ".")
    return ppo_name


if __name__ == "__main__":
    cli_train()
