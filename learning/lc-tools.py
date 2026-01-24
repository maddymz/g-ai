from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def calculate_discount(price: float, discount_percentage: float) -> float:
    """
    Calculates the final price after applying a discount.

    Args:
        price (float): The original price of the item.
        discount_percentage (float): The discount percentage (e.g., 20 for 20%).

    Returns:
        float: The final price after the discount is applied.
    """
    if not (0 <= discount_percentage <= 100):
        raise ValueError("Discount percentage must be between 0 and 100")

    discount_amount = price * (discount_percentage / 100)
    final_price = price - discount_amount
    return final_price
    
llm_with_tools = llm.bind_tools([calculate_discount])

result = llm_with_tools.invoke("What is the price of an item that costs $100 after a 20% discount?")
print(result.tool_calls)

args = result.tool_calls[0]['args']
print(calculate_discount.invoke(args))