import yaml
import os
import csv
import json
from dotenv import load_dotenv
from litellm import completion
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime


# Load environment variables
load_dotenv()

MODEL_NAME = 'gpt-4o-mini'

# use pydantic models for structured output
class RecipeDimensionTuple(BaseModel):
    cuisine_type: str
    meal_type: str
    degree_of_simplicity: str

class RecipeDimensionsList(BaseModel):
    tuples: List[RecipeDimensionTuple]

class SyntheticQuery(BaseModel):
    query: str
    tuple_reference: str

class SyntheticQueriesList(BaseModel):
    queries: List[SyntheticQuery]

NUM_TUPLES_TO_GENERATE = 15
NNUM_QUERIES_PER_TUPLE = 7

#  load prompts from yaml
def load_prompts(yaml_file: str) -> List[Dict[str, str]]:
    """Load prompts from YAML file."""
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_prompt_by_subject(prompts_data: List[Dict], subject: str) -> str:
    """Find a specific prompt by its subject."""
    for prompt in prompts_data:
        if prompt['subject'] == subject:
            return prompt['message']
    raise ValueError(f"Prompt with subject '{subject}' not found")

#  call llm 
def call_llm(messages: List[Dict[str, str]], response_format: Any) -> Any:
    """Call LLM with structured output format."""
    try:
        response = completion(
            model=MODEL_NAME,
            messages=messages,
            response_format=response_format
        )
        
        # Parse the JSON response and return the Pydantic model
        response_content = response.choices[0].message.content
        return response_format(**json.loads(response_content))
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise

#  generate recipe dimension tuples
def generate_recipe_tuples(tuple_prompt: str) -> RecipeDimensionsList:
    """Generate recipe dimension tuples using the tuple prompt."""
    
    # Enhance the prompt to ensure structured output
    enhanced_prompt = f"""
    Generate exactly {NUM_TUPLES_TO_GENERATE} unique and divers tuples combinations with the following dimension values.
    Each combination should represent a different user scenario. Ensure balanced coverage across all dimensions - don't over-represent any particular value or combination.

    {tuple_prompt}
    
    Please provide your response as a JSON object with this structure:
    {{
        "tuples": [
            {{
                "cuisine_type": "example_cuisine",
                "meal_type": "example_meal", 
                "degree_of_simplicity": "example_difficulty"
            }}
        ]
    }}
    
    
    """
    
    messages = [{"role": "user", "content": enhanced_prompt}]
    
    result = call_llm(messages, RecipeDimensionsList)
    print(f"Generated {len(result.tuples)} recipe dimension tuples")
    return result

# generate synthetic queries 
def generate_synthetic_queries(query_prompt: str, tuples_data: RecipeDimensionsList) -> SyntheticQueriesList:
    """Generate synthetic queries based on the tuples"""
    all_queries = []
    # Format tuples for the prompt
    for tuple_item in tuples_data.tuples:
        enhanced_prompt=f"""
        You are a recipe chatbot prompt generator. Create {NNUM_QUERIES_PER_TUPLE} dirverse, realistic user queries for this recipe dimension tuple:
        - Cuisine: {tuple_item.cuisine_type}
        - Meal: {tuple_item.meal_type}
        - Difficulty: {tuple_item.degree_of_simplicity}

        Generate queries that include:
        1. Different user personas (busy parent, college student, food enthusiast, beginner cook)
        2. Realistic scenarios (dinner party, quick lunch, weekend cooking, meal prep)
        3. Specific constraints (dietary restrictions, time limits, available ingredients)
        4. Natural language variations (casual, formal, urgent, exploratory)
        5. Contextual details (cooking for family, trying new cuisine, comfort food, healthy options)

        Example personas and scenarios:
        - "I'm a college student with 30 minutes - need a simple [cuisine] [meal] that won't break the bank"
        - "Planning a dinner party and want to impress guests with [difficulty] [cuisine] [meal] - any suggestions?"
        - "New to cooking [cuisine] food - looking for [difficulty] [meal] recipes with clear instructions"
        - "Busy parent needs [difficulty] [cuisine] [meal] that kids will actually eat"

        {query_prompt}
        
        Create varied, natural queries that sound like real people asking for recipe help.

        Provide your response as JSON:
        
        {{
            "queries": [
                {{
                    "query": "natural language query text",
                    "tuple_reference": "cuisine_type: meal_type: difficulty"
                }}
            ]
        }}
        """

        messages = [{"role": "user", "content": enhanced_prompt}]

        result = call_llm(messages, SyntheticQueriesList)
        all_queries.extend(result.queries)
    print(f"Generated {len(all_queries)} synthetic queries")
    return SyntheticQueriesList(queries=all_queries)

# save outputs to the files
def save_tuples_to_file(tuples_data: RecipeDimensionsList, filename: str = "recipe_dimensions.py"):
    """Save generated tuples to a Python file."""

    python_code = '''"""
Recipe dimension tuples generated automatically.
Each tuple contains (cuisine_type, meal_type, degree_of_simplicity)
"""

recipe_dimensions = [
'''
    
    for tuple_item in tuples_data.tuples:
        python_code += f'    ("{tuple_item.cuisine_type}", "{tuple_item.meal_type}", "{tuple_item.degree_of_simplicity}"),\n'
    
    python_code += ''']

# Example usage:
if __name__ == "__main__":
    print("Total number of recipe combinations:", len(recipe_dimensions))
    print("\\nExample combinations:")
    for i, (cuisine, meal, difficulty) in enumerate(recipe_dimensions[:5], 1):
        print(f"{i}. Cuisine: {cuisine}, Meal: {meal}, Difficulty: {difficulty}")
'''

    with open(filename, 'w') as f:
        f.write(python_code)
    
    print(f"Tuples saved to {filename}")

def save_queries_to_csv(queries_data: SyntheticQueriesList, filename: str = "synthetic_queries.csv"):
    """save queries to CSV file"""

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    
        # Write header
        writer.writerow(['id', 'query'])
        
        # Write queries
        for i, query_item in enumerate(queries_data.queries, 1):
            writer.writerow([i, query_item.query])
    
    print(f"Queries saved to {filename}")

def main():
    print("Starting automated queries generation")
    print(f"Using LLM model: {MODEL_NAME}")

    try:
        # load prompts from YAML file
        print("loading prompts from YAML...")
        prompt_data = load_prompts('homeworks/hw2/prompts.yaml')

        # get the prompts we need
        tuple_prompt = get_prompt_by_subject(prompt_data, 'tuple prompt')
        query_prompt = get_prompt_by_subject(prompt_data, 'synthetic query prompt')

        print("Found required prompts")

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output folder
        output_folder = "generated_queries"
        os.makedirs(output_folder, exist_ok=True)

        print(f"Output folder: {output_folder}")
        print(f"Timestamp: {timestamp}")

        #  generate recipe dimension tuples
        print("generating recipe dimension tuples...")
        tuple_data = generate_recipe_tuples(tuple_prompt)

        # save tuples to the file
        tuples_filename = f"{output_folder}/recipe_dimensions_{timestamp}.py"
        save_tuples_to_file(tuple_data, filename=tuples_filename)

        # generate synthetic queries
        print("generating synthetic queries...")
        queries_data = generate_synthetic_queries(query_prompt, tuple_data)

        # save to csv file
        queries_filename = f"{output_folder}/synthetic_queries_{timestamp}.csv"
        save_queries_to_csv(queries_data, filename=queries_filename)

        print("Process completed successfully!")
        print(f"generated {len(tuple_data.tuples)} tuples and {len(queries_data.queries)} queries")
    except Exception as e:
        print(f"Error in main process {e}")

if __name__=="__main__":
    main()