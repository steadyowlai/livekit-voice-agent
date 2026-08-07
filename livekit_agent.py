import logging
from livekit import agents
from livekit_agent.main import server

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)