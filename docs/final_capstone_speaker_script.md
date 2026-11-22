# Smart Wallet Rewards Optimizer — Final Presentation Script

Target delivery: approximately 8–10 minutes at a measured pace. The eight slides follow the seven sections in the supplied presentation-planning outline; the system architecture is split across two slides.

## Slide 1 — Opening and project overview

Hi, everyone. I’m presenting the Smart Wallet Rewards Optimizer, my capstone project in the Personal Assistant track. It helps rewards-aware consumers compare payment cards before making a purchase, bringing together merchant and wallet tools, issuer benefit documents, deterministic reward calculations, and human review when the risk is higher.

The language model has a few bounded jobs, but it doesn’t make the financial decision on its own. This is an educational prototype that recommends a card; it never authorizes or makes a payment. I’ll walk through why this problem matters, how the system works, what testing tells us, and what I’d improve next.

## Slide 2 — Why this problem matters

At checkout, “which card should I use?” sounds simple. In practice, the answer can depend on several things. Reward rates vary by merchant category and merchant category code. A department store, wholesale club, restaurant, or grocery merchant may not qualify the way we expect. International transaction fees can eat into the rewards, and purchase protection or extended warranty terms come with conditions that can change over time.

The intended user is someone who wants a quick comparison without having to memorize all those rules. An agent-based approach makes sense because the system has to bring together merchant information, wallet data, policy documents, and calculations. I don’t want one model response trying to recall all of that from memory.

The goal is an auditable recommendation for a proposed purchase. The user enters transaction details, and the system checks a synthetic wallet and public benefit sources. It doesn’t connect to real accounts or place transactions. And the benefit values in the demo are just illustrative assumptions—not cash guarantees or estimates of insurance value.

## Slide 3 — System architecture: coordinated components

Here’s how the pieces fit together. A LangGraph state graph coordinates the workflow, and a Pydantic model validates the request. Separate tools look up location context, resolve the merchant, and load the synthetic wallet. Their results move through the graph as structured data.

The Context and Rewards stage identifies candidate cards and calculates rewards from the merchant category, card multiplier, and mock point valuation. The Benefits and Risk stage retrieves evidence when the transaction warrants it. The Decision Orchestrator evaluates the candidates, applies hard constraints, and returns a ranked result with alternatives and an audit trail.

The selected language model has a few bounded tasks. In the web app, users can choose local Qwen or an Anthropic Claude model; Qwen is used in the CLI and Colab workflow. The model can suggest a catalog match for an uncertain merchant, critique the completed reasoning tree, or help explain a result. It can’t create an MCC, change a score, or bypass a constraint. In short, the model helps interpret and communicate; deterministic code handles the numbers.

## Slide 4 — System architecture: reasoning and memory

The reasoning tree gives each eligible card its own branch. First, the system calculates the rewards. Then it checks whether retrieved card-specific text supports a benefit estimate. Finally, it applies any international fee and drops a branch if that fee is as large as, or larger than, the reward.

The retrieval subsystem downloads public issuer guides, extracts PDF or HTML passages, attaches card and source metadata, and searches a local ChromaDB collection. It uses MiniLM embeddings and maximal marginal relevance to return relevant, less-redundant passages. Citations retain their source URL and PDF page where available.

Here’s one concrete example from a test: a one-hundred-dollar Trader Joe’s purchase earns six hundred eighty cents in base rewards on the synthetic American Express Gold profile. That’s a four-times multiplier and a configured one-point-seven cents per point. It’s a deterministic calculation from mock inputs. Any additional benefit estimate is illustrative and has to be supported by retrieved text.

## Slide 5 — Key design decisions

Several design choices shaped the system as the project evolved. Early on, I identified merchant-category ambiguity, changing benefit terms, and financial arithmetic as reasons not to rely on prompting alone. I added a local merchant catalog, synthetic wallet access, and retrieval over public issuer guides. Tree-of-Thought reasoning made the trade-offs explicit by comparing rewards, supported benefits, and fees for each card. Then LangGraph brought the context, benefits, and decision roles together with typed handoffs.

Safety shaped the final boundaries, too. The system accepts only a finite, positive amount up to fifty thousand dollars. It asks for confirmation above twenty-five hundred dollars, and when the merchant or category is uncertain, AI-matched, unresolved, or conflicting. If benefit evidence is missing or stale, the system withholds it instead of presenting it as current. The user can confirm or correct the details and run the evaluation again. Those review actions are audited, but they don’t silently change the catalog or system policy.

That lets routine, well-grounded recommendations move quickly while higher-impact or uncertain cases go back to the user. The system still only makes recommendations. If a model call fails, it uses a factual fallback rather than inventing a financial result.

## Slide 6 — Evaluation and results

For evaluation, I ran the automated test suite. On November twenty-second, twenty twenty-six, all eighty-six collected tests passed. They cover merchant aliases and unresolved names, wallet loading, transaction bounds, reward arithmetic, tree ordering, fee handling and pruning, evidence citations, approval routing, model fallbacks, Claude API handling, and web application behavior.

The grocery example is one check of the arithmetic. Other tests confirm that a document title alone doesn’t count as benefit evidence, international fees are converted correctly, uncertain merchants trigger confirmation, and a model failure still gets a factual fallback.

I want to be careful not to overstate that result. Passing tests show that the cases we wrote behave as expected; they don’t give us a real-world accuracy percentage. I haven’t yet run a representative merchant benchmark, a user study, a calibration analysis, or latency tests across different hardware. Those are next steps, not completed results.

## Slide 7 — Repository and implementation

You can find the project in the public GitHub repository, cloudbyte slash smart-wallet-optimizer. It includes the README, source code, synthetic datasets, tests, and a Colab notebook. The README explains setup and how to run the command-line or web demo. Issuer documents and the local vector index are cached rather than committed.

The implementation uses Python, Pydantic version two, and LangGraph for typed validation and workflow orchestration. Streamlit and the command-line interface share the same workflow. ChromaDB, MiniLM embeddings, and maximal marginal relevance support local retrieval over public issuer guides. The web app offers local Qwen and Claude choices for bounded language tasks. When Claude is selected, the user enters an API key for that operation; the app does not persist it, and the relevant prompt is sent to Anthropic. The web page now presents the evaluation heading, ToT branch table, and audit trail directly beneath the recommendation. Together, these components keep the financial ranking deterministic.

The repository is where you can review the code, examples, tests, and setup instructions. The project is reproducible as an educational prototype, but the test results don’t mean it’s ready for production.

## Slide 8 — Closing reflection

My biggest takeaway is that autonomy needs clear boundaries. Combining deterministic tools, retrieved evidence, structured reasoning, bounded model support, and selective review gives us an auditable recommendation without giving the model payment authority. The system is most useful when it makes assumptions and uncertainty visible, rather than hiding them behind a confident-sounding answer.

The prototype still has important limitations. Wallet profiles and point values are synthetic. Benefit estimates are illustrative, not actuarial. Merchant-level categories may not describe the item or payment processor used in a real transaction. There is no user study, representative accuracy benchmark, or persistent reviewer queue.

Next, I’d build a labeled merchant benchmark, measure calibration and reviewer workload, add item-level context, and track latency and source freshness. Any move toward real accounts or payment execution would need a separate security, privacy, authorization, and compliance review.

Thank you. I’m happy to take questions.
