# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for interacting with the Dhan API (https://api.dhan.co), which provides algorithmic trading and market data services. The project uses Python 3.13 and uv as the package manager.

## Environment Setup

The project requires environment variables for authentication:
- `CLIENT_ID`: Dhan API client ID
- `ACCESS_TOKEN`: Dhan API access token

These should be stored in a `.env` file (already gitignored).

## Development Commands

### Package Management
- Install dependencies: `uv sync`
- Add a new dependency: `uv add <package-name>`
- Run Python scripts: `uv run python <script-path>`

### Running Code
- Execute the main script: `uv run python core/dhan.py`
- The script currently fetches historical chart data from the Dhan API

## Code Architecture

### Project Structure
- `core/`: Contains the main application code
  - `dhan.py`: Main script that demonstrates API interaction with Dhan's historical charts endpoint

### API Integration
The project uses Python's built-in `http.client` for making HTTPS requests to the Dhan API. The current implementation:
- Loads credentials from environment variables using `python-dotenv`
- Makes POST requests to `/v2/charts/historical` endpoint
- Uses headers for authentication: `client-id` and `access-token`

### Authentication Pattern
Authentication is handled via headers passed with each request:
```python
headers = {
    "client-id": CLIENT_ID,
    "access-token": ACCESS_TOKEN,
    "Content-Type": "application/json",
    "Accept": "application/json",
}
```

## Notes
- No test suite is currently present in the project
- No linting or type checking is configured yet
- The project is in early development stages with a single example script
