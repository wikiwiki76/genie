from dotenv import load_dotenv
import json
import os
import time
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

# Initialize the Ollama LLM with your locally running gpt-oss model
llm = OllamaLLM(
    model="gpt-oss:20b",  # name of the model you have pulled in Ollama
    temperature=0.8,      # randomness of outputs (0 = deterministic, 1 = very random)
    # top_p=0.9,            # nucleus sampling, consider tokens up to cumulative prob.
    # num_predict=512,      # max tokens to generate
    # stop=["</s>"],        # stop sequences
)

# Example prompt
prompt = """
## Role
You are a UOB personal banker specializing in One Account helping customers maximize their interest. Output will be used as a conversation starter with the customer.

## Constraints
- Do not fabricate numbers or rules. Only use data in "Customer Data", "Product Rules" and "Interest Rate Data".
- Do not recommend losses, lower balance or lower tier.
- Use snap_date to determine the exact number of days in the month/year.
- Round DOWN all money to the nearest hundredths.

## Tasks
1. Compute the exact current monthly base and bonus interest. Use the number of days in the month and year from snap_date and round down to the nearest hundredths.
2. Simulate the following "what-if" scenarios to calculate potential gains:
   1) Increase to the next LEVEL if the current is not Level 3. (FIRST priority)
   2) Increase the BALANCE up to the cap of the CURRENT tier. (SECOND priority)
   3) Increase to the next TIER if the current is not Tier 3. (LAST priority)
3. For each scenario, compute exact simulated base and bonus interest using the same day-count and rounding rules.

## Output Requirements
- Identify the customer’s current balance, interest tier, and level.
- Clearly explain the incremental gains between current vs simulated interest, showing how each gain is computed.
- Select the recommendation with the first positive gain based on the prioritization above and write next steps to achieve higher interest.

## Customer Data
    {
        "snap_date": "2025-08-31",
        "one_account": {
            "avg_balance": 127000,
            "salary_credit": 2000,
            "card_spend": 700,
            "giro_count": 3
        }
    }


## Product Rules
{
    "levels": {
        "level_1": "Card spend >= S$500",
        "level_2": "Card spend >= S$500 & Perform 3 GIRO debit transactions",
        "level_3": "Card spend >= S$500 & Credit salary of minimum S$1,600 (GIRO not required)"
    },
    "tiers": {
        "tier_1": "Bonus interest applies on first S$75,000",
        "tier_2": "Bonus interest applies on next S$50,000 (from >S$75,000 to S$125,000)",
        "tier_3": "Bonus interest applies on next S$25,000 (from >S$125,000 to S$150,000)"
    }
}


## Interest Rate Data
{
    "Base Rate": "0.05%",
    "Level 1 Bonus": {
        "Tier 1": "0.60%",
        "Tier 2": "0.00%",
        "Tier 3": "0.00%"
    },
    "Level 2 Bonus": {
        "Tier 1": "0.95%",
        "Tier 2": "1.95%",
        "Tier 3": "0.00%"
    },
    "Level 3 Bonus": {
        "Tier 1": "1.45%",
        "Tier 2": "2.95%",
        "Tier 3": "4.45%"
    }
}

## Return EXACTLY this JSON schema (no extra text):

{{
  "snap_date": "<YYYY-MM-DD>",
  "current": {{
    "avg_balance": 0,
    "level": "Level 1 | Level 2 | Level 3",
    "tier": "Tier 1 | Tier 2 | Tier 3",
    "days_in_month": 0,
    "days_in_year": 0,
    "base_interest_month": 0.00,
    "bonus_interest_month_breakdown": {{
      "tier_1_amount": 0.00,
      "tier_2_amount": 0.00,
      "tier_3_amount": 0.00
    }},
    "total_interest_month": 0.00
  }},
  "simulations": [
    {{
      "name": "Upgrade Level",
      "assumption": "Increase to the next level if not Level 3",
      "new_level": "Level 1 | Level 2 | Level 3",
      "new_tier": "Tier 1 | Tier 2 | Tier 3",
      "new_avg_balance": 0,
      "base_interest_month": 0.00,
      "bonus_interest_month_breakdown": {{
        "tier_1_amount": 0.00,
        "tier_2_amount": 0.00,
        "tier_3_amount": 0.00
      }},
      "total_interest_month": 0.00,
      "incremental_gain_vs_current": 0.00,
      "explanation": "How incremental gain is computed"
    }},
    {{
      "name": "Top-up to Tier Cap",
      "assumption": "Increase balance up to cap of current tier",
      "new_level": "same as current | or specify",
      "new_tier": "Tier 1 | Tier 2 | Tier 3",
      "new_avg_balance": 0,
      "base_interest_month": 0.00,
      "bonus_interest_month_breakdown": {{
        "tier_1_amount": 0.00,
        "tier_2_amount": 0.00,
        "tier_3_amount": 0.00
      }},
      "total_interest_month": 0.00,
      "incremental_gain_vs_current": 0.00,
      "explanation": "How incremental gain is computed"
    }},
    {{
      "name": "Upgrade Tier",
      "assumption": "Increase to the next tier if not Tier 3",
      "new_level": "same as current | or specify",
      "new_tier": "Tier 1 | Tier 2 | Tier 3",
      "new_avg_balance": 0,
      "base_interest_month": 0.00,
      "bonus_interest_month_breakdown": {{
        "tier_1_amount": 0.00,
        "tier_2_amount": 0.00,
        "tier_3_amount": 0.00
      }},
      "total_interest_month": 0.00,
      "incremental_gain_vs_current": 0.00,
      "explanation": "How incremental gain is computed"
    }}
  ],
  "recommended_action": {{
    "chosen_scenario": "Upgrade Level | Top-up to Tier Cap | Upgrade Tier | None",
    "reasoning": "Pick the first scenario (by priority order) that yields a positive incremental gain, and explain briefly.",
    "next_steps": [
      "Step 1 ...",
      "Step 2 ...",
      "Step 3 ..."
    ]
  }}
}}
"""

# Run the model
response = llm.invoke(prompt)

from datetime import datetime
today = datetime.today()
strftime = today.strftime("%Y-%m-%d-%H-%M-%S")
print("Formatted date:", strftime)

with open(f"./outputs/test_output.txt", "w") as f:
    f.write(response)
    
# print("LLM Response:\n", response)