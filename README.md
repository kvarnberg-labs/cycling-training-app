# hermes-training-tool

CLI tool for the Hermes agent to fetch cycling/running training data from Intervals.icu and generate LLM-powered training recommendations for Discord users.

## Usage

```bash
# List registered users
cd /opt/data/workspaces/github/cycling-training-app && python3 -m app.query_for_user --list-users

# Get structured JSON context (for feeding into any LLM)
python3 -m app.query_for_user joey0624 --context

# Weekly plan (LLM-powered)
python3 -m app.query_for_user joey0624 --weekly

# Form assessment (LLM-powered, falls back to rule-based)
python3 -m app.query_for_user joey0624 --assessment

# Rule-based fallback (no LLM)
python3 -m app.query_for_user joey0624 --rule-based
```

## How it works

1. Fetches training data from the [Intervals.icu](https://intervals.icu) API for a registered Discord user
2. Builds a structured context pack with PMC metrics (CTL/ATL/TSB), recent activities, and athlete profile
3. Default mode calls an LLM to generate personalized coaching recommendations
4. Falls back gracefully to rule-based recommendations if the LLM call fails

The `--context` flag outputs clean JSON for the Hermes agent's own LLM.

## Register a user

```bash
python3 -m app.user_manager register <discord_id> \
  --intervals-key <api_key> \
  --athlete-id <athlete_id>
```

