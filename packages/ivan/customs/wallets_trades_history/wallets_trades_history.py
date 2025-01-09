"""
A mech tool for querying Flipside data about wallet transactions.
"""

import functools
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from flipside import Flipside

class APIClients:
    """Class for managing API clients."""
    def __init__(self, api_keys: Any):
        self.flipside_key = api_keys["flipside"]
        
        if not self.flipside_key:
            raise ValueError("Missing required Flipside API key")
            
        self.flipside = Flipside(self.flipside_key, "https://api-v2.flipsidecrypto.xyz")

def with_key_rotation(func: Any):
    """Decorator to handle API key rotation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]:
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper

def parse_prompt(prompt: str) -> Tuple[List[str], int]:
    """Parse the prompt to extract wallet addresses and time interval."""
    # Default time interval (7 days)
    default_days = 7

    # Extract wallet addresses using regex
    wallet_pattern = r'0x[a-fA-F0-9]{40}'
    wallets = re.findall(wallet_pattern, prompt)
    
    # Require at least one wallet address
    if not wallets:
        raise ValueError("No valid wallet addresses found in prompt. Please provide at least one Ethereum address (0x...)")
    
    # Extract time interval
    time_patterns = {
        r'(\d+)\s*days?': 1,
        r'(\d+)\s*weeks?': 7,
        r'(\d+)\s*months?': 30,
        r'(\d+)\s*years?': 365
    }
    
    days = default_days
    for pattern, multiplier in time_patterns.items():
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            days = int(match.group(1)) * multiplier
            break
        
    return wallets, days

def generate_sql_query(wallets: List[str], days: int) -> str:
    """Generate SQL query for wallet transactions."""
    wallet_list = "('" + "','".join(wallets) + "')"
    
    return f"""
    SELECT DISTINCT
      to_varchar(amount_in_usd + amount_out_usd, '$999,999,999') AS "USD Value",
      to_varchar(amount_in_usd, '$999,999,999') AS "Amount Bought",
      symbol_in AS "Bought Symbol",
      to_varchar(amount_out_usd, '$999,999,999') AS "Amount Sold",
      symbol_out AS "Sold Symbol",
      block_timestamp AS "Time",
      trader AS "Trader",
      tx_hash AS "Transaction Hash", 
      blockchain AS "Blockchain"
    FROM crosschain.defi.ez_dex_swaps
    WHERE trader IN {wallet_list}
      AND amount_out_usd IS NOT NULL
      AND amount_in_usd IS NOT NULL
      AND block_timestamp > CURRENT_TIMESTAMP() - interval '{days} day'
    ORDER BY 1 DESC
    LIMIT 150;
    """

@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the Flipside query and return results."""
    try:
        # Initialize clients
        clients = APIClients(api_keys)
        
        # Parse wallets and days from prompt
        wallets, days = parse_prompt(prompt)
        
        # Generate SQL query
        sql = generate_sql_query(wallets, days)
        
        # Run query
        query_result = clients.flipside.query(sql)
        
        # Format results
        metadata = {
            "wallets": wallets,
            "days": days,
            "query": sql
        }
        
        return (
            str(query_result.rows),  # Main response
            "",  # Context
            metadata,  # Metadata
            None,  # Additional data
        )
    except Exception as e:
        return str(e), "", None, None