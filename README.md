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

## Structure
- `src/`: Core logic, including LangGraph orchestrator and ReAct framework.
- `data/`: Mock JSON datasets for merchant categories, user profiles, and policy guides.
- `tests/`: Unit and integration tests for the reasoning loop and tool integration.
- `docs/`: Project documentation and architecture diagrams.
