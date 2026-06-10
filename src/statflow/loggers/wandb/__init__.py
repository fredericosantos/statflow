"""
Weights & Biases run provider.

Talks to the W&B website's public GraphQL API directly, authenticating with the
API key in ``~/.netrc`` (machine ``api.wandb.ai``). W&B projects map to
experiments, run ``config`` to ``params.*``, and numeric run ``summaryMetrics``
to ``metrics.*`` — the same canonical wide DataFrame every provider produces.

wandb/
├── __init__.py   # Package initialization
└── provider.py   # WandbProvider (RunProvider) over the W&B GraphQL API
"""
