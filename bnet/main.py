"""FastAPI app creation, logger configuration and main API routes."""

from bnet.di import global_injector
from bnet.launcher import create_app

app = create_app(global_injector)
