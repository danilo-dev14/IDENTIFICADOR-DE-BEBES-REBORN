# Dbot2

A simple Discord bot using `discord.py`.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with:

```env
DISCORD_TOKEN=your_discord_bot_token_here
```
```

3. Run the bot:

```bash
python main.py
```

## Notes

- `main.py` also reads `DISCORD_TOKEN` from the environment if `.env` is not used.
- If `DISCORD_TOKEN` is missing, the bot exits with a clear error.
