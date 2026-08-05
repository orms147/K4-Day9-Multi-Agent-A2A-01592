import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from data_loader import DataLoader

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Sử dụng Qwen 2.5 7B (dưới 10B) qua OpenRouter 
MODEL = "qwen/qwen-2.5-7b-instruct" 

class BaseAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

    def invoke(self, user_prompt: str) -> str:
        print(f"[{self.name}] Analyzing...")
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"} if "JSON" in self.system_prompt else None
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{self.name}] API Error: {e}")
            return "{}"

class CustomerAgent(BaseAgent):
    def __init__(self, data_loader: DataLoader):
        super().__init__(
            "CustomerAgent",
            "You are a Customer Agent. Your task is to extract customer details."
        )
        self.db = data_loader

    def process(self, order_id: str):
        # 1. Deterministic Python logic
        details = self.db.get_order_details(order_id)
        if not details: return {}
        
        customer_id = details['order']['customer_id']
        customers = self.db.datasets['customers']
        cust_info = customers[customers['customer_id'] == customer_id]
        if cust_info.empty: return {}
        
        unique_id = cust_info.iloc[0]['customer_unique_id']
        
        orders = self.db.datasets['orders']
        related = orders[orders['customer_id'].isin(
            customers[customers['customer_unique_id'] == unique_id]['customer_id']
        )]
        related_ids = related['order_id'].tolist()
        if order_id in related_ids:
            related_ids.remove(order_id)
            
        # Limit to 5 as per rules
        related_ids = related_ids[:5]
        
        return {
            "customer_unique_id": unique_id,
            "related_order_ids": related_ids
        }

class OrderProductAgent(BaseAgent):
    def __init__(self, data_loader: DataLoader):
        super().__init__("OrderProductAgent", "Extract order and product context.")
        self.db = data_loader
        
    def process(self, order_id: str):
        details = self.db.get_order_details(order_id)
        if not details: return {}
        
        items = details['items']
        item_ids = [f"{order_id}:{item['order_item_id']}" for item in items][:5]
        seller_ids = list(set([item['seller_id'] for item in items]))[:3]
        product_ids = list(set([item['product_id'] for item in items]))[:5]
        
        products = self.db.datasets['products']
        prod_info = products[products['product_id'].isin(product_ids)]
        categories = prod_info['product_category_name'].dropna().unique().tolist()[:5]
        
        return {
            "item_ids": item_ids,
            "seller_ids": seller_ids,
            "product_ids": product_ids,
            "category_names": categories,
            "multi_item_order": len(items) >= 2,
            "multi_seller_order": len(seller_ids) >= 2,
            "multiple_categories": len(categories) >= 2
        }

class PaymentAgent(BaseAgent):
    def __init__(self, data_loader: DataLoader):
        super().__init__("PaymentAgent", "Calculate payment reconciliation.")
        self.db = data_loader
        
    def process(self, order_id: str):
        details = self.db.get_order_details(order_id)
        if not details or not details['items']: 
            return None # Return None as requested by rules if no items
            
        items = details['items']
        payments = details['payments']
        
        item_total = sum(item['price'] for item in items)
        freight_total = sum(item['freight_value'] for item in items)
        expected_total = item_total + freight_total
        
        payment_total = sum(p['payment_value'] for p in payments)
        diff = payment_total - expected_total
        reconciled = abs(diff) <= 0.10
        
        payment_types = list(set([p['payment_type'] for p in payments]))
        payment_ids = [f"{order_id}:{p['payment_sequential']}" for p in payments][:5]
        
        return {
            "payment_reconciliation": {
                "currency": "BRL",
                "item_total_brl": round(item_total, 2),
                "freight_total_brl": round(freight_total, 2),
                "expected_total_brl": round(expected_total, 2),
                "payment_total_brl": round(payment_total, 2),
                "difference_brl": round(diff, 2),
                "reconciled": reconciled,
                "payment_types": payment_types
            },
            "split_payment": len(payments) >= 2,
            "payment_ids": payment_ids
        }

class DeliveryAgent(BaseAgent):
    def __init__(self, data_loader: DataLoader):
        super().__init__("DeliveryAgent", "Analyze delivery variance.")
        self.db = data_loader
        
    def process(self, order_id: str):
        details = self.db.get_order_details(order_id)
        if not details or not details['items']: return None
        
        order = details['order']
        items = details['items']
        
        delivered_at = order['order_delivered_customer_date']
        estimated_at = order['order_estimated_delivery_date']
        carrier_handoff = order['order_delivered_carrier_date']
        
        delivery_variance = None
        if pd.notna(delivered_at) and pd.notna(estimated_at):
            delivery_variance = (delivered_at - estimated_at).total_seconds() / 3600.0
            
        seller_analysis = []
        late_handoff_sellers = []
        
        # Group items by seller to find earliest shipping limit
        sellers_limits = {}
        for item in items:
            s_id = item['seller_id']
            limit = item['shipping_limit_date']
            if pd.notna(limit):
                if s_id not in sellers_limits or limit < sellers_limits[s_id]:
                    sellers_limits[s_id] = limit
                    
        for s_id, limit in sellers_limits.items():
            variance = None
            late = False
            if pd.notna(carrier_handoff):
                variance = (carrier_handoff - limit).total_seconds() / 3600.0
                if variance > 0:
                    late = True
                    late_handoff_sellers.append(s_id)
                    
            seller_analysis.append({
                "seller_id": s_id,
                "shipping_limit_at": limit.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(limit) else None,
                "handoff_variance_hours": round(variance, 2) if variance is not None else None,
                "late_handoff": late
            })
            
        return {
            "delivered_at": delivered_at.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(delivered_at) else None,
            "estimated_delivery_at": estimated_at.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(estimated_at) else None,
            "carrier_handoff_at": carrier_handoff.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(carrier_handoff) else None,
            "delivery_variance_hours": round(delivery_variance, 2) if delivery_variance is not None else None,
            "seller_handoff_analysis": seller_analysis,
            "late_handoff_seller_ids": late_handoff_sellers[:3]
        }

class PolicyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "PolicyAgent",
            """You are the Policy Agent. Evaluate EC_POLICY_V2 rules strictly based on the provided facts.
Output ONLY a valid JSON object describing:
- primary_issue
- secondary_issues (ordered correctly)
- case_status
- confidence
- root_cause_analysis
- evidence_ids
- financial_resolution
- resolution_actions
Use the exact schema requested."""
        )
        
    def process(self, facts: dict):
        prompt = f"""
Facts about the order:
{json.dumps(facts, indent=2)}

Apply EC_POLICY_V2 strictly:
Primary issue priority:
1. canceled_order_paid
2. unavailable_order_paid
3. late_delivery_seller
4. late_delivery_logistics
5. valid_split_payment
6. unsupported_late_claim

Output JSON must follow the schema and rules. Ensure you include evidence_ids format: order:<id>, item:<id>, payment:<id>, seller:<id>, policy:<root_cause_code>.
"""
        response = self.invoke(prompt)
        try:
            # Strip markdown code blocks if any
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            return json.loads(response)
        except json.JSONDecodeError:
            print("Failed to decode PolicyAgent JSON output.")
            return {}

class CoordinatorAgent:
    def __init__(self, data_loader: DataLoader):
        self.customer_agent = CustomerAgent(data_loader)
        self.order_product_agent = OrderProductAgent(data_loader)
        self.payment_agent = PaymentAgent(data_loader)
        self.delivery_agent = DeliveryAgent(data_loader)
        self.policy_agent = PolicyAgent()
        self.db = data_loader
        
    def run_case(self, case_json: dict):
        order_id = case_json['customer_request']['claimed_order_id']
        case_id = case_json['case_id']
        
        # Gather facts via Domain Agents (Hybrid Python execution)
        customer_ctx = self.customer_agent.process(order_id)
        op_ctx = self.order_product_agent.process(order_id)
        payment_ctx = self.payment_agent.process(order_id)
        delivery_ctx = self.delivery_agent.process(order_id)
        
        details = self.db.get_order_details(order_id)
        order_status = details['order']['order_status'] if details else None
        
        facts = {
            "order_id": order_id,
            "order_status": order_status,
            "customer": customer_ctx,
            "order_product": op_ctx,
            "payment": payment_ctx,
            "delivery": delivery_ctx
        }
        
        # Policy Agent evaluates rules
        policy_decision = self.policy_agent.process(facts)
        
        # Assemble Final JSON
        affected_entities = {
            "order_ids": [order_id],
            "item_ids": op_ctx.get('item_ids', []),
            "seller_ids": op_ctx.get('seller_ids', []),
            "payment_ids": payment_ctx.get('payment_ids', []) if payment_ctx else []
        }
        
        final_output = {
            "case_id": case_id,
            "case_assessment": {
                "primary_issue": policy_decision.get("primary_issue", ""),
                "secondary_issues": policy_decision.get("secondary_issues", []),
                "case_status": policy_decision.get("case_status", "no_action"),
                "confidence": policy_decision.get("confidence", 0.95)
            },
            "affected_entities": affected_entities,
            "customer_context": {
                "customer_unique_id": customer_ctx.get("customer_unique_id"),
                "related_order_ids": customer_ctx.get("related_order_ids", [])
            },
            "product_context": {
                "product_ids": op_ctx.get("product_ids", []),
                "category_names": op_ctx.get("category_names", [])
            },
            "delivery_analysis": delivery_ctx,
            "payment_reconciliation": payment_ctx.get("payment_reconciliation") if payment_ctx else None,
            "root_cause_analysis": policy_decision.get("root_cause_analysis", {}),
            "evidence_ids": policy_decision.get("evidence_ids", []),
            "financial_resolution": policy_decision.get("financial_resolution", {}),
            "resolution_actions": policy_decision.get("resolution_actions", [])
        }
        
        return final_output
