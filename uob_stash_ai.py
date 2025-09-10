from dotenv import load_dotenv
import json
import os
import time
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

# ----------------------------
# Model setup (same pattern)
# ----------------------------
# load_dotenv()
# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     model="deepseek/deepseek-r1-0528:free",
#     temperature=0
# )

llm = OllamaLLM(
    model="gpt-oss:20b",  # name of the model you have pulled in Ollama
    # temperature=0.7,      # randomness of outputs (0 = deterministic, 1 = very random)
    # top_p=0.9,            # nucleus sampling, consider tokens up to cumulative prob.
    # num_predict=512,      # max tokens to generate
    # stop=["</s>"],        # stop sequences
)

# ----------------------------
# Data payloads
# ----------------------------
customer_payloads = [
    {"snap_date": "2025-08-31",
     "stash_account": {
         "average_balance_last_month": 51000,
         "average_balance_this_month": 49000
         }
    }
]

product_rules = {
    "Criteria": "Maintain or increase your monthly average balance as compared to the previous month to qualify for bonus interest rate",
    "tiers": {
        "tier_1": "Bonus interest applies on first S$10,000",
        "tier_2": "Bonus interest applies on next S$30,000 (from >S$10,000 to S$40,000)",
        "tier_3": "Bonus interest applies on next S$30,000 (from >S$40,000 to S$70,000)",
        "tier_4": "Bonus interest applies on next S$30,000 (from >S$70,000 to S$100,000)"
    }
}

interest_rate_data = {
    "Base Rate": "0.05%",
    "Bonus Rate": {
        "Tier 1": "0.00%",
        "Tier 2": "1.55%",
        "Tier 3": "2.15%",
        "Tier 4": "2.90%"
    }
}

# ----------------------------
# Prompt builder (mirrors your style)
# ----------------------------
def build_prompt(customer_data, product_rules, interest_rate_data):
    return f"""
## Role
You are a product specialist for the UOB Stash Account. Your task is to analyze customer profiles and generate personalized recommendations to help customers maximize the benefits of the UOB Stash Account. Your output will be used by a supervisor agent, who will combine your recommendations with insights from other products to deliver a personalized recommendation to the customer.

## Constraints
- Do not fabricate numbers or rules. Only use data in "Customer Data", "Product Rules" and "Interest Rate Data".
- Do not recommend losses, lower balance or lower tier.
- Use snap_date to determine the exact number of days in the month/year.
- Round DOWN the final interest to the nearest hundredths.
- all data and output are in SGD and monthly

## Tasks
1. Compute the exact current monthly base and bonus interest. Use the number of days in the month and year from snap_date and round down to the nearest hundredths.
2. Bonus interest is only applicable if the average balance this month is maintained or increased compared to last month.
3. Apply only the applicable bonus tier rate to each corresponding tier balance. The bonus interest is computed progressively based on the amount in each tier, not the highest tier only.
4. Simulate the "what-if" scenarios to calculate potential gains
   1) Check if this month balance is eligible for bonus interest. If not, recommend a top up to qualify and compute the incremental gains. (FIRST priority)
   2) Increase the BALANCE up to the cap of the CURRENT tier. (SECOND priority)
   3) Increase to the next TIER if the current is not Tier 4. (LAST priority)
5. For each scenario, compute exact simulated base and bonus interest using the same day-count and rounding rules.

## Output Requirements
- Identify the customer’s previous balance, current balance, interest tier.
- Clearly explain the incremental gains between current vs simulated interest, showing how each gain is computed.
- Select the recommendation (by priority order) and write next steps to achieve higher interest.

## Customer Data
{json.dumps(customer_data, indent=2)}

## Product Rules
{json.dumps(product_rules, indent=2)}

## Interest Rate Data
{json.dumps(interest_rate_data, indent=2)}

## Return EXACTLY this JSON schema (no extra text):

{{
  "snap_date": "<YYYY-MM-DD>",
  "current": {{
    "average_balance_last_month": 0,
    "average_balance_this_month": 0,
    "tier": "Tier 1 | Tier 2 | Tier 3 | Tier 4",
    "days_in_month": 0,
    "days_in_year": 0,
    "base_interest_month": 0.00,
    "bonus_interest_month_breakdown": {{
      "tier_1_amount": 0.00,
      "tier_2_amount": 0.00,
      "tier_3_amount": 0.00,
      "tier_4_amount": 0.00
      "tier_1_bonus_interest_amount": 0.00,
      "tier_2_bonus_interest_amount": 0.00,
      "tier_3_bonus_interest_amount": 0.00,
      "tier_4_bonus_interest_amount": 0.00
    }},
    "total_interest_month": 0.00
    }},
    "recommended_action": {{
      "reasoning": "Pick a recommendation that yields a positive incremental gain, identify the incremental gain and explain briefly.",
      "recommended_incremental_gain_vs_current" : 0.00,
      "next_steps": [
        "Step 1 ...",
        "Step 2 ...",
        "Step 3 ..."
    ]
  }}
}}
"""

# ----------------------------
# Run loop (same JSON-extract pattern)
# ----------------------------
results = []
for i, cust in enumerate(customer_payloads, 1):
    print(f"[{i}/{len(customer_payloads)}] Processing snap_date {cust.get('snap_date')}")
    try:
        time.sleep(0.8)
        prompt = build_prompt(cust, product_rules, interest_rate_data)
        response = llm.invoke(prompt)

        # Extract strict JSON
        # m = re.search(r"\{.*\}\s*$", response.content, re.DOTALL)
        m = re.search(r"\{.*\}\s*$", response, re.DOTALL)
        if not m:
            raise ValueError("No valid JSON in LLM response.")
        parsed = json.loads(m.group(0))
        results.append(parsed)
        

    except Exception as e:
        print(f"Error: {e}")
        results.append({
            "snap_date": cust.get("snap_date"),
            "error": str(e)
        })

# ----------------------------
# Save
# ----------------------------
from datetime import datetime
today = datetime.today()
strftime = today.strftime("%Y%m%d%H%M%S")

save_file = f"outputs/uob_stash_interest_simulation_{strftime}.json"
with open(save_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Completed. Saved final results to {save_file}")