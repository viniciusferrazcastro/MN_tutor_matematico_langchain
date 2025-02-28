from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
import os
import re

# Load environment variables
load_dotenv()

# Configure the LLM model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Class to format the question as JSON
class MathQuestion(BaseModel):
    question: str = Field(description="Mathematical question sent by the user")
    category: str = Field(default="mathematics", description="Question category")

# Function to correctly format the question
def format_question(question: str) -> dict:
    """Creates a JSON object with the correct structure."""
    return MathQuestion(question=question).dict()

# Function to validate if the question is mathematical
def validate_question(question: str) -> bool:
    """
    Checks if the question contains basic mathematical terms.

    :param question: Question sent by the user.
    :return: True if it is a valid question, False otherwise.
    """
    math_pattern = re.compile(r'[\d+\-*/=]')  # Looks for numbers and mathematical operators
    math_terms = ["addition", "subtraction", "multiplication", "division", "equation", "calculate", "solve"] # Looks for mathematical terms

    return bool(math_pattern.search(question) or any(term in question.lower() for term in math_terms))

# Create the prompt template for the LLM
prompt = ChatPromptTemplate.from_template("Solve the following math question: {question}")

# Create the chain that interacts with the model
professor_chain = LLMChain(prompt=prompt, llm=llm)

# Function that sends the question to the LLM and returns the answer
def virtual_professor_llm(question: dict) -> dict:
    """Sends the question to the GPT-3.5 model and returns the response."""
    response = professor_chain.invoke({"question": question["question"]})
    return {"question": question["question"], "response": response["text"]}

# Testing with a math question
question = "Solve the equation 2x + 3 = 7"

if validate_question(question):
    print("Valid question! Sending to the Virtual Professor...")

    question_json = format_question(question)  # Format as JSON
    response = virtual_professor_llm(question_json)  # Send to LLM
    
    print("Virtual Professor's Response:")

    formatted_response = response["response"].replace("\n", "\n")

    print(f'Question: {response["question"]}')
    print(f'Response:\n{formatted_response}')

else:
    print("Invalid question! Please send a math-related question.")
