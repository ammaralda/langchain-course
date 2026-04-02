from dotenv import load_dotenv
import re
import inspect
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.5:9b"


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    'get_product_price': get_product_price,
    'apply_discount': apply_discount

}

def get_tool_description(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # This used to get original function before it get wrapped by @traceable
        original_function = getattr(tool_function, "__wrapped__", tool_function)

        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        
        descriptions.append(f"{tool_name}{signature} - {docstring}") # Example output -> get_product_price(product: str) -> float - Look up the price of a product in the catalog.
    return "\n".join(descriptions)

tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())


react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""





@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):


    print(f"Question: {question}")
    print("=" * 60)


    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(model=MODEL,
                                      messages=[{"role": "user", "content": full_prompt}],
                                      options={"stop": ["\nObservation"], 
                                               "temperature": 0}
                                               )
        output = response.message.content
        print(f"LLM Output: {output}")
        
        print(f"   [Parsing] Looking for Final Answer in LLM output...")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            
            print(f"   [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or action_input_match:
            print(
                "   [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
        
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        # Split comma-separated args; strip key= prefix if LLM output key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"   [Tool Executing] {tool_name}({args})...")
        
        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")