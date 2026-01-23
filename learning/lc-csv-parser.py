from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

parser_list = CommaSeparatedListOutputParser()
prompt_list = PromptTemplate.from_template(
    template = "Answer the question.\n{format_instructions}\n{question}",
    input_vairables = ["question"],
    partial_variables = {"format_instructions": parser_list.get_format_instructions()},
)

prompt_value = prompt_list.invoke({"question": "List 4 chocolate brands"})
response = llm.invoke(prompt_value)
print(response.content)

returned_object = parser_list.parse(response.content)
print(type(returned_object))