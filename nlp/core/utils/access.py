import os

# Load from .env file
from dotenv import load_dotenv

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(__file__))

# Connect the path with your '.env' file name
load_dotenv(os.path.join(BASEDIR, "..", "..", "..", ".env"))
hf_token = os.getenv["HUGGING_FACE_HUB_TOKEN"]

from huggingface_hub import login


class Access:
    @staticmethod
    def login_hf(access_token: str = None) -> None:
        """
        Login to Hugging Face Hub with the given access token.
        """
        if access_token is None:
            access_token = hf_token
        login(token=access_token)
        return
