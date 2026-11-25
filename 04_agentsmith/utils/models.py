"""The chat model used by the agent. Set MODEL_NAME in .env as `provider:model`."""
import os
from dotenv import load_dotenv
from utils import PROJECT_ROOT

load_dotenv(PROJECT_ROOT.parent / ".env")

model = os.environ.get("MODEL_NAME", "openai:gpt-6-luna")
