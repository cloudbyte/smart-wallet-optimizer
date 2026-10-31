# Smart Wallet Rewards Optimizer

A Python demo agent built with LangGraph and an open-weight Qwen3 language
model. Local runs default to Qwen3-0.6B on CPU; the Colab notebook selects
Qwen3-4B for GPU-backed runs. Qwen selects data tools in a constrained ReAct
loop, critiques the Tree-of-Thought (ToT) card branches against retrieved
evidence, and explains the result. Merchant facts, reward calculations, and
hard eligibility rules come from tools and deterministic code. The project is
designed for local development and Google Colab.

> **Disclaimer:** This project uses synthetic wallet data and illustrative
> benefit estimates for demonstration and educational purposes. It is not
> financial advice and should not be used to make real payment decisions.

## What It Does

- Uses the LLM to choose among merchant, location, wallet, and benefit-retrieval
  tools; tool observations are added to LangGraph state for subsequent turns.
- Resolves a merchant to a canonical name and spending category using the local
  merchant dataset.
- Loads a synthetic wallet and compares the reward yield of its eligible cards.
- Downloads official card-benefit PDFs and issuer pages on demand, then retrieves
  card-filtered passages with a purpose-trained MiniLM embedder and MMR.
- Evaluates card branches with deterministic financial calculations and an LLM
  critic that must cite retrieved text verbatim.
- Accounts for foreign transaction fees and prunes cards where the fee is at
  least the reward yield.
- Flags transactions for human approval when the merchant context is uncertain,
  the amount exceeds the configured threshold, or no eligible card remains.
- Uses the selected Qwen3 model to choose tools, critique the reasoning tree,
  and explain the structured recommendation.
- Includes an audit trail and structured evaluation tree in the result.

LLM-selected actions are restricted to allowlisted tools and required context
lookups; an invalid action is logged and replaced with the next required tool.
The LLM critic can add a grounded qualitative assessment, but cannot change
calculated values or bypass hard constraints. Benefit guides are downloaded
from issuer-hosted public sources into the ignored `.cache/card-benefits`
directory; they are not bundled in the repository. RAG citations include each
source URL and PDF page number. Issuer documents can change, so users should
confirm current eligibility and terms with the issuer.

## Getting Started

### Google Colab

Open [`colab_setup.ipynb`](./colab_setup.ipynb) in Google Colab and run its cells
from top to bottom. The setup cell clones a fresh copy of the GitHub project on
every run, installs the dependencies, runs the test suite, and starts the
interactive demo. Re-running the clone cell removes the existing
`smart-wallet-optimizer` directory at the notebook's current location before
cloning again.

The notebook selects Qwen3-4B from Hugging Face (about 8 GB) and loads it into
memory. Use a Colab GPU runtime; free-tier GPU availability depends on the
current quota, and CPU-only generation may be impractically slow or run out of
memory. Local runs default to the smaller Qwen3-0.6B model, which is practical
for CPU testing but still downloads the model weights on first use. Set
`WALLET_AGENT_MODEL` and `WALLET_AGENT_DEVICE` to override the defaults. RAG
additionally downloads the compact
`sentence-transformers/all-MiniLM-L6-v2` embedding model (about 90 MB) on first
use. Benefit-eligible transactions download the issuer documents they need;
internet access is required, and those files are cached locally rather than
added to the repository.

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

The official source URLs are maintained in
[`data/benefit_sources.json`](./data/benefit_sources.json) for Chase Sapphire
Reserve, Capital One Venture X, and American Express Gold. The LangGraph agent
maintains short-term state through a ReAct tool loop. At
each turn Qwen chooses an available action; tool observations are added to state
and provided on the next turn. Merchant, location, and wallet lookups are
required grounding steps. For regular local purchases under $50, the graph
skips benefit retrieval. Other transactions retrieve benefits for each active
card. The agent then performs the full evaluation:

1. **Yield:** Calculate base rewards from the merchant category, card
   multiplier, and mock point valuation.
2. **Benefits:** Download and extract the issuer's PDF/HTML terms into local
   ignored cache, filter passages by card ID, and retrieve up to three relevant
   passages with MiniLM embeddings and maximal marginal relevance (MMR). Returned
   evidence is labeled with source URL, fetch time, and PDF page where available.
   Configured perk estimates remain illustrative and are applied only when
   retrieved text supports the benefit.
3. **Fees:** Subtract any international transaction fee and prune a branch when
   the fee equals or exceeds its reward yield.

The evaluator scores every wallet branch breadth-first using the deterministic
rubric `net value = yield + illustrative perk estimate - foreign transaction
fee`. A Qwen critic assesses the completed tree but its comments are surfaced
only when they name a candidate and quote a phrase found in retrieved evidence;
it never changes the scores or ranking. The response includes the
recommendation, alternatives, optional `tree_assessment`, LLM-generated
`assistant_message`, `reasoning_tree`, and an audit trail. Numerical decisions,
hard pruning, and approval triggers remain deterministic.

## Data and Configuration

- [`data/merchants.json`](./data/merchants.json) maps sample merchant names to
  categories and supermarket eligibility.
- [`data/wallets.json`](./data/wallets.json) defines the sample user's cards,
  reward multipliers, point valuations, and preference for tie-breaking.
- [`data/benefit_sources.json`](./data/benefit_sources.json) lists official
  issuer document sources for cards in the sample wallet.
- [`src/rag/ingestion.py`](./src/rag/ingestion.py) downloads and parses issuer
  PDF/HTML terms; [`src/rag/retriever.py`](./src/rag/retriever.py) handles
  embeddings, ranking, and citations. Downloaded documents remain local and
  are excluded from version control.
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
  agent/            LangGraph orchestration, state, and graph nodes
  main.py           Interactive command-line demo
  llm.py            Runtime-selected Qwen3 planner, critic, and explainer
  models.py         Request, response, wallet, and reasoning models
  reasoning/        Deterministic Tree-of-Thought evaluator and scoring
  safety/           Approval policy and model-output grounding checks
  rag/              Issuer ingestion, parsing, and RAG retrieval
  tools/            Merchant, location, and wallet data tools
data/               Synthetic merchant and wallet profiles
tests/              Unit and integration tests
docs/               Project checkpoint designs
colab_setup.ipynb   Google Colab setup and demo notebook
```
