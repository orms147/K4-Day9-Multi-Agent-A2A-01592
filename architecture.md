# System Architecture: Multi-Agent E-commerce Dispute Resolution

## 1. System Overview
The system is designed to automate the investigation and resolution of e-commerce customer support requests based on the Brazilian E-Commerce Public Dataset by Olist. It uses a Multi-Agent Architecture where distinct AI agents specialize in analyzing specific domains of data (orders, payments, delivery, etc.). The agents collaborate, share evidence, and apply business policies to generate a structured JSON resolution for each case.

## 2. Agent Roles and Responsibilities

### 2.1. Coordinator Agent
- **Role**: The central orchestrator.
- **Responsibilities**: 
  - Receives the input JSON case (`EC_XXX.json`).
  - Extracts the `claimed_order_id` and delegates tasks to domain-specific agents.
  - Aggregates the findings from all domain agents.
  - Hands over the aggregated context to the Policy Agent.
  - Structures the final output JSON and sends it to the Verifier Agent for validation before saving.
- **Data Access**: None directly; manages handoffs.

### 2.2. Customer Agent
- **Role**: Customer history analyzer.
- **Responsibilities**: 
  - Finds the `customer_unique_id` based on the current order's `customer_id`.
  - Retrieves all historical orders for this `customer_unique_id` to populate `related_order_ids`.
- **Data Access**: `customers`, `orders`.

### 2.3. Order & Product Agent
- **Role**: Order structure and catalog analyzer.
- **Responsibilities**: 
  - Validates the existence of the `claimed_order_id`.
  - Extracts all items (`item_ids`), sellers (`seller_ids`), products (`product_ids`), and categories (`category_names`).
  - Identifies if the order involves multiple items or multiple sellers.
- **Data Access**: `orders`, `order_items`, `products`, `sellers`, `product_category_name_translation`.

### 2.4. Payment Agent
- **Role**: Financial reconciliation specialist.
- **Responsibilities**: 
  - Retrieves all payment rows (`payment_ids`) for the order.
  - Calculates the total expected value (`sum(price) + sum(freight_value)`).
  - Calculates the total paid value and the difference (`difference_brl`).
  - Determines if the payment is fully reconciled (difference <= 0.10 BRL) and records payment types.
- **Data Access**: `order_payments`, `order_items`.

### 2.5. Delivery Agent
- **Role**: Logistics and SLA analyzer.
- **Responsibilities**: 
  - Analyzes actual delivery times vs estimated delivery times (`delivery_variance_hours`).
  - Analyzes seller handoff times vs shipping limit dates for each seller (`handoff_variance_hours`).
  - Identifies late handoff sellers and carrier delays.
- **Data Access**: `orders`, `order_items`.

### 2.6. Policy Agent
- **Role**: Business logic and decision maker.
- **Responsibilities**: 
  - Takes the aggregated, verifiable facts from the Customer, Order, Payment, and Delivery agents.
  - Applies `EC_POLICY_V2` sequentially.
  - Determines the `primary_issue`, `secondary_issues`, `responsible_parties`, and root causes.
  - Recommends the financial resolution (refund amounts) and the required resolution actions.
  - Compiles the `evidence_ids`.
- **Data Access**: Reads only the factual summaries provided by domain agents.

### 2.7. Verifier Agent
- **Role**: Schema and rule validator.
- **Responsibilities**: 
  - Enforces array limits (e.g., max 5 order IDs, 5 item IDs, 3 seller IDs).
  - Checks if `confidence` is within [0, 1].
  - Validates the structure and existence of `evidence_ids` against standard formats.
  - Ensures the final output strictly adheres to the requested JSON schema.
- **Data Access**: Reads the final drafted JSON.

## 3. Data Access Layer
To prevent LLM hallucinations, agents do not directly run complex logic on raw text. Instead, they interact with a **Data Loader System** (implemented in `data_loader.py`). The datasets are loaded into memory using Pandas DataFrames. Domain agents use deterministic Python queries to extract factual numbers, timestamps, and IDs, ensuring the evidence is 100% verifiable.

## 4. Handoff & Execution Flow

1. **Initialization**: Data Loader loads all 9 CSV files into memory.
2. **Ingestion**: Coordinator Agent reads `input/EC_XXX.json`.
3. **Parallel Domain Analysis**:
   - Coordinator passes `claimed_order_id` to Customer, Order & Product, Payment, and Delivery agents.
   - These agents query the Data Loader and return structured factual JSONs containing their respective analyses.
4. **Policy Application**:
   - Coordinator merges the domain facts and passes them to the Policy Agent.
   - Policy Agent evaluates `EC_POLICY_V2` and generates the final resolution plan, actions, and evidence list.
5. **Validation**:
   - Coordinator drafts the final JSON response.
   - Verifier Agent checks the draft against schema constraints and limits.
6. **Output**: The finalized, verified JSON is written to `output/EC_XXX.json`.
