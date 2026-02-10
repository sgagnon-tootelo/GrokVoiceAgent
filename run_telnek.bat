@echo off
set DOTENV_OVERRIDE=.env.telnek
uv run python src\agent.py run --port 8081
pause