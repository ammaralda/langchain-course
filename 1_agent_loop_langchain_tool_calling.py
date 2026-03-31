from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI

MAX_ITERATION = 10

@tool
def get_product_price(product:str) -> float:
    """Look up the price of product in the catalog"""
    print(f"Looking for the price of product = {product}")
    prices = {"laptop": 1233.9, "mouse": 32.1, "keyboard": 23.4}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount for price based on user discount tier and return discounted price as a final
    Available tiers: bronze, silver, gold"""
    discount = {"bronze": 10, "silver": 15, "gold": 20}
    print(f"User discount tier = {discount_tier}")
    return round(price * (1 - (discount.get(discount_tier)) / 100), 2)

@traceable(name="Langchain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview",
                             temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are helpful shopping assistant\n"
                "You have access to a product catalog tool "
                "and a discount tool. \n\n"
                "STRICT RULES = MUST FOLLOW THESE EXACTLY:\n"
                "1. Never guess the price or assume any product"
                "2. Must call get_product_price first to get the real price\n"
                "3. Only call apply_discount after you have received a price from get_product_price. pass exact number"
                "4. Never calculate discount using math, always use apply_discount tool"
                "5. If user does not provide tier, always ask the user to make sure"
            )
        ),
        
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n ---- Iteration {iteration} ----")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If dont have tool call, this is final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id   = tool_call.get("id")

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' is not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id = tool_id)
        )
    print("ERROR MAX ITERATION")

if __name__ == "__main__": 
    run_agent("What is the price of laptop with gold discount?") 