# Smart Wallet Rewards Optimizer

A Python demo personal assistant that compares cards in a mock wallet and
recommends a card for a transaction. It considers merchant category, reward
multipliers, point valuations, selected card benefits, and foreign transaction
fees. The project is designed for local development and Google Colab.

> **Disclaimer:** This project uses synthetic wallet data and illustrative
> benefit estimates for demonstration and educational purposes. It is not
> financial advice and should not be used to make real payment decisions.

## What It Does

- Resolves a merchant to a canonical name and spending category using the local
  merchant dataset.
- Loads a synthetic wallet and compares the reward yield of its eligible cards.
- Retrieves mock card-benefit text and applies configurable illustrative perk
  estimates.
- Accounts for foreign transaction fees and prunes cards where the fee is at
  least the reward yield.
- Flags transactions for human approval when the merchant context is uncertain,
  the amount exceeds the configured threshold, or no eligible card remains.
- Includes an audit trail and structured evaluation tree in the result.

## Getting Started

### Google Colab

Open [`colab_setup.ipynb`](./colab_setup.ipynb) in Google Colab and run its cells
from top to bottom. The setup cell clones a fresh copy of the GitHub project on
every run, installs the dependencies, runs the test suite, and starts the
interactive demo. Re-running the clone cell removes the existing
`smart-wallet-optimizer` directory at the notebook's current location before
cloning again.

### Local Setup

Requires Python 3.9 or later.

```bash
git clone https://github.com/cloudbyte/smart-wallet-optimizer.git
cd smart-wallet-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the interactive demo from the project root:

```bash
PYTHONPATH=. python -m src.main
```

Enter a merchant, transaction amount, and whether the transaction is
international. For example, merchants in the sample data include Trader Joe's,
Walmart, Delta Air Lines, Shell, and The Capital Grille.

## Recommendation Flow

The agent resolves the merchant and wallet, then evaluates eligible wallet
cards. For regular local purchases under $50, it uses reward yield only to avoid
unnecessary perk retrieval. Other transactions use the full evaluation:

1. **Yield:** Calculate base rewards from the merchant category, card
   multiplier, and mock point valuation.
2. **Benefits:** Retrieve matching mock benefit text and apply the configured
   purchase-protection or warranty estimate when the card profile supports it.
3. **Fees:** Subtract any international transaction fee and prune a branch when
   the fee equals or exceeds its reward yield.

The response includes the recommendation, alternatives, a `reasoning_tree`, and
an audit trail. The implemented evaluator is deterministic and local; CrewAI,
LangGraph orchestration, and MCP state management from the project checkpoint
are future architectural directions, not runtime requirements.

## Mock Data and Configuration

- [`data/merchants.json`](./data/merchants.json) maps sample merchant names to
  categories and supermarket eligibility.
- [`data/wallets.json`](./data/wallets.json) defines the sample user's cards,
  reward multipliers, point valuations, and preference for tie-breaking.
- [`src/rag/retriever.py`](./src/rag/retriever.py) contains the small in-memory
  mock card-benefit corpus.
- `purchase_protection_value_rate` and `extended_warranty_value_rate` in the
  wallet data are illustrative fractions of the transaction amount, not
  guarantees or actuarial valuations.

Merchant and wallet data are synthetic. Location resolution has offline
shortcuts for sample venues and may query OpenStreetMap Nominatim for other
inputs when network access is available.

## Tests

From the project root, run:

```bash
PYTHONPATH=. pytest
```

The tests cover merchant and wallet tools, recommendations, breadth-first
evaluation, fee pruning, low-value early exit, and approval handling.

## Project Layout

```text
src/
  agent.py          Transaction evaluation flow
  main.py           Interactive command-line demo
  models.py         Request, response, wallet, and reasoning models
  tot.py            Deterministic Tree-of-Thought evaluator
  rag/              Mock card-benefit retrieval
  tools/            Merchant, location, and wallet data tools
data/               Synthetic merchant and wallet profiles
tests/              Unit and integration tests
docs/               Project checkpoint designs
colab_setup.ipynb   Google Colab setup and demo notebook
```
