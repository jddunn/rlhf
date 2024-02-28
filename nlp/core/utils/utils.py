from .files import Files
from .nlp import NLP
from .model import Model
from .dataset import Dataset


class Utils:
    """
    Utils class for our NLP library.
    """

    def __init__(
        self,
        max_tokens_to_return: int = 200,
        adjust_max_tokens_dynamic: bool = True,
        max_tokens_dynamic_ratio: float = 0.75,
        use_lora: bool = False,
    ):
        self.files = Files()  # Files utils
        # Model handling utils
        self.model = Model(use_lora=use_lora)  # Model utils
        # General NLP utils
        self.nlp = NLP(
            max_tokens_to_return, adjust_max_tokens_dynamic, max_tokens_dynamic_ratio
        )
        # Dataset utils
        self.dataset = Dataset()
        return
