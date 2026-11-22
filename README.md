# Smart Wallet Rewards Optimizer

A Python demo agent built with LangGraph and deterministic financial logic.
Local runs use the open-weight Qwen3 model by default; the Streamlit UI also
supports Anthropic Claude. The Colab notebook selects Qwen3-4B for GPU-backed
runs. LangGraph coordinates three specialized roles: context and rewards,
benefits and risk, and final decision orchestration. The selected model can
resolve merchant candidates, critique Tree-of-Thought (ToT) card branches
against retrieved evidence, and explain the result. Merchant facts, reward
calculations, and hard eligibility rules come from tools and deterministic
code.

> **Disclaimer:** This project uses synthetic wallet data and illustrative
> benefit estimates for demonstration and educational purposes. It is not
> financial advice and should not be used to make real payment decisions.

## What It Does

- Resolves merchant, location, and wallet context in parallel, then passes
  structured observations through LangGraph's shared workflow state.
- Runs separate Context & Rewards, Benefits & Risk, and Decision Orchestrator
  graph stages with typed handoffs and auditable results.
- Resolves a merchant to a canonical name and spending category using the local
  merchant dataset. When exact lookup fails, the LLM may select a candidate
  from that catalog; MCC and category always come from the catalog, and
  AI-matched names are flagged for human confirmation.
- Loads a synthetic wallet and compares the reward yield of its eligible cards.
- Downloads official card-benefit PDFs and issuer pages on demand, then retrieves
  card-filtered passages from a persistent local ChromaDB vector store using a
  purpose-trained MiniLM embedder and MMR.
- Evaluates card branches with deterministic financial calculations and an LLM
  critic that must cite retrieved text verbatim.
- Accounts for foreign transaction fees and prunes cards where the fee is at
  least the reward yield.
- Flags transactions for human approval when the merchant context is uncertain,
  confidence is below 0.70, the amount exceeds $2,500, required benefit
  retrieval fails, or no eligible card remains.
- The web app offers a review form to confirm or correct merchant/category
  context and rerun the recommendation. Review state and corrections are
  recorded in that evaluation's audit trail; they are not persisted or used to
  automatically change catalogs or routing rules.
- Accepts positive finite amounts in the UI, then blocks recommendations through
  an agent-level guardrail when the amount exceeds `$50,000`.
- Uses the selected Qwen3 or Claude model to resolve ambiguous merchant names,
  critique the reasoning tree, and explain the structured recommendation.
- Displays an evaluation-results section with the recommendation, alternatives,
  visible ToT branch table, and audit trail; safety-blocked requests also show
  their audit details.
- Records end-to-end evaluation latency and uses safe fallbacks when optional
  model critique or explanation calls fail.

The LLM merchant resolver can only select from catalog entries. The LLM critic
can add a grounded qualitative assessment, but cannot change calculated values
or bypass hard constraints. Benefit guides are downloaded
from issuer-hosted public sources into the ignored `.cache/card-benefits`
directory; they are not bundled in the repository. RAG citations include each
source URL and PDF page number. Issuer documents can change, so users should
confirm current eligibility and terms with the issuer.
Expired cached issuer documents are refreshed before use; if refresh fails,
the stale copy is withheld and the affected recommendation requires review.

## Getting Started

### Google Colab

Open [`Smart_Wallet_Rewards_Optimizer_Colab.ipynb`](./Smart_Wallet_Rewards_Optimizer_Colab.ipynb) in Google Colab and run its cells
from top to bottom. The setup cell clones a fresh copy of the GitHub project on
every run, installs the dependencies, and runs the test suite. The final cell
runs the interactive command-line demo directly in the notebook, avoiding
Colab port-proxy issues. For the browser-based app, use the local setup below.
Re-running the clone cell removes the existing `smart-wallet-optimizer`
directory at the notebook's current location before cloning again.

With a GPU runtime, the notebook selects Qwen3-4B from Hugging Face using
4-bit quantization to reduce VRAM use. It also caps each prompt at 2,048 tokens
to limit inference memory. Without a GPU, it selects Qwen3-0.6B on CPU instead
of offloading the larger model to CPU/disk, which can make inference extremely
slow. Free-tier GPU
availability depends on the current quota. Local runs also default to the
smaller Qwen3-0.6B model, which is practical for CPU testing but still
downloads the model weights on first use. Set
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

To run the browser-based app locally instead:

```bash
PYTHONPATH=. streamlit run src/web_app.py
```

Streamlit opens the app at `http://localhost:8501`. It shares the same agent
and evaluation logic as the command-line demo. Choose a model using the
**AI model** selector. The local Qwen option needs no API key; its first
recommendation may take a few minutes while model weights download. Claude
options require an Anthropic API key, entered in the password field for each
evaluation or corrected-context re-evaluation. Claude Haiku 4.5 is the
lowest-cost Claude option currently listed; check Anthropic's
[current API pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)
for rates and model availability.

The app does not write the Anthropic API key to disk, an environment variable,
or a persistent client cache; it is held in memory for the submitted operation
and the password field clears on submission. Choosing Claude sends the key and
the relevant evaluation prompt to Anthropic over HTTPS. Prompts can include
the merchant, transaction amount, catalog candidates, recommendation facts,
and retrieved benefit excerpts. Anthropic API usage may incur charges under
your account. Do not enter real financial account data, credentials, or
personally identifying information.

After an evaluation, the page displays the same key diagnostic sections as the
Colab/CLI output: **EVALUATION RESULTS**, **ToT Evaluation**, and **Audit
Trail**. The ToT table shows each evaluated or pruned card branch; the audit
trail lists workflow events, evidence outcomes, guardrail decisions, and
latency where recorded. These details remain visible for blocked transactions
as well as successful recommendations.

## Recommendation Flow

The official source URLs are maintained in
[`data/benefit_sources.json`](./data/benefit_sources.json) for Chase Sapphire
Reserve, Capital One Venture X, and American Express Gold. The LangGraph
workflow runs independent merchant, location, and wallet lookups in parallel.
The Context & Rewards Specialist calculates yields and identifies active card
candidates; the Benefits & Risk Specialist retrieves relevant evidence when
needed; and the Decision Orchestrator applies ToT scoring and approval rules.
For regular local purchases under $50, the graph skips benefit retrieval.
Other transactions retrieve benefits for each active candidate card. The
workflow then performs the full evaluation:

1. **Yield:** Calculate base rewards from the merchant category, card
   multiplier, and mock point valuation.
2. **Benefits:** Download and extract the issuer's PDF/HTML terms into local
   ignored cache, filter passages by card ID, and retrieve up to three relevant
   passages with MiniLM embeddings and maximal marginal relevance (MMR). Chunks
   are 500 tokens with 50-token overlap and carry card, section, source, date,
   and page metadata. ChromaDB persists the local index under
   `.cache/card-benefits/chroma`. Returned evidence is labeled with source URL,
   fetch time, and PDF page where available. Configured perk estimates remain
   illustrative and are applied only when retrieved text supports the benefit.
3. **Fees:** Subtract any international transaction fee and prune a branch when
   the fee equals or exceeds its reward yield.

The evaluator scores every wallet branch breadth-first using the deterministic
rubric `net value = yield + illustrative perk estimate - foreign transaction
fee`. The selected model assesses the completed tree, but its comments are surfaced
only when they name a candidate and quote a phrase found in retrieved evidence;
it never changes the scores or ranking. The response includes the
recommendation, alternatives, optional `tree_assessment`, LLM-generated
`assistant_message`, `reasoning_tree`, and an audit trail. Numerical decisions,
hard pruning, and approval triggers remain deterministic.

## Data and Configuration

- [`data/merchants.json`](./data/merchants.json) contains a synthetic catalog
  of common grocery, wholesale, retail, pharmacy, dining, travel, gas, and
  subscription merchants, including aliases and MCC/category metadata.
- [`data/wallets.json`](./data/wallets.json) defines the sample user's cards,
  reward multipliers, point valuations, and preference for tie-breaking.
- [`data/benefit_sources.json`](./data/benefit_sources.json) lists official
  issuer document sources for cards in the sample wallet.
- [`src/rag/ingestion.py`](./src/rag/ingestion.py) downloads and parses issuer
  PDF/HTML terms into overlapping token chunks; [`src/rag/retriever.py`](./src/rag/retriever.py)
  indexes embeddings and metadata in a local persistent ChromaDB collection,
  performs card-filtered vector search plus MMR reranking, and formats
  citations. Downloaded documents and the index remain local and are excluded
  from version control.
- `purchase_protection_value_rate` and `extended_warranty_value_rate` in the
  wallet data are illustrative fractions of the transaction amount, not
  guarantees or actuarial valuations.

Merchant records and category mappings are illustrative, not authoritative:
MCCs can vary by location, department, payment channel, or transaction type.
Several entries are flagged for confirmation for that reason. AI merchant
resolution cannot add catalog entries or invent MCC/category values. Location
resolution has offline
shortcuts for sample venues and may query OpenStreetMap Nominatim for other
inputs when network access is available.

## Tests

From the project root, run:

```bash
PYTHONPATH=. pytest
```

The tests cover merchant and wallet tools, recommendations, breadth-first
evaluation, fee pruning, low-value early exit, and approval handling.

## Final Capstone Materials

- [Final capstone report (DOCX)](docs/final_capstone_report.docx)
- [Final capstone report (PDF)](docs/final_capstone_report.pdf)
- [Presentation slide deck](docs/final_capstone_presentation.html) — use the
  arrow keys to navigate and the full-screen control for presenting.
- [Presentation speaker script](docs/final_capstone_speaker_script.md)
- [Video submission summary template](docs/final_capstone_video_submission.html)

The report's evaluation results reflect the automated test suite, not a
production accuracy or latency benchmark. A hosted presentation video must
still be recorded and submitted separately.

## Project Layout

```text
src/
  agent/            LangGraph multi-agent orchestration, state, and role nodes
  main.py           Interactive command-line demo
  llm.py            Qwen3 and Anthropic Claude merchant resolver, critic, and explainer
  models.py         Request, response, wallet, and reasoning models
  reasoning/        Deterministic Tree-of-Thought evaluator and scoring
  safety/           Approval policy and model-output grounding checks
  rag/              Issuer ingestion, parsing, and RAG retrieval
  tools/            Merchant, location, and wallet data tools
data/               Synthetic merchant and wallet profiles
tests/              Unit and integration tests
docs/               Checkpoint designs and final capstone materials
Smart_Wallet_Rewards_Optimizer_Colab.ipynb  Google Colab setup and demo notebook
```
