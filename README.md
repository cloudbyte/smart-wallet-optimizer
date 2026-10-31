# Smart Wallet Rewards Optimizer

An autonomous personal assistant engineered to maximize financial yield and consumer purchase protection during point-of-sale transactions.

## Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/cloudbyte/smart-wallet-optimizer.git
   cd smart-wallet-optimizer
   ```
2. Initialize virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Colab Setup
To run in Google Colab, use the provided `colab_setup.ipynb` notebook or clone the repository directly:
```python
!git clone https://github.com/cloudbyte/smart-wallet-optimizer.git
%cd smart-wallet-optimizer
!pip install -r requirements.txt
```

## Tree-of-Thought Evaluation
Transaction recommendations use a deterministic breadth-first search over the
eligible cards in the mock wallet:

1. **Depth 1:** Calculate rewards using the merchant category and point values.
2. **Depth 2:** Retrieve card benefit text and add configured mock estimates for
   purchase protection and extended warranty.
3. **Depth 3:** Apply foreign transaction fees and prune branches where the fee
   is at least the reward yield.

Local transactions under $50 bypass perk retrieval and fee evaluation unless
they are in a high-risk category. International transactions always receive the
full evaluation. The response includes a `reasoning_tree` and audit entries so
each evaluation and pruning decision can be inspected. If no eligible branch
survives, the response asks for human approval instead of inventing a card
recommendation.

The benefit value rates in `data/wallets.json` are illustrative synthetic
estimates used by this demo; for example, `0.02` estimates a benefit value of
2% of the transaction amount. They are not actuarial valuations or financial
advice. Update those values to change how mock perks affect ranking. The
checkpoint's CrewAI/LangChain/MCP mapping remains an architectural direction;
this first implementation keeps ToT evaluation local and requires no agent
services or API keys, so it can also run in Colab. Merchant resolution may
optionally use OpenStreetMap when an offline match is unavailable.

## Structure
- `src/`: Core recommendation logic, including the ToT evaluator and mock RAG retrieval.
- `data/`: Synthetic merchant, wallet, and card-benefit data.
- `tests/`: Unit and integration tests for evaluation and tool behavior.
- `docs/`: Project checkpoints and architecture designs.
