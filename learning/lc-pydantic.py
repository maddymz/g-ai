from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

class Author(BaseModel):
    name: str = Field(description="The name of the author")
    number: int = Field(description="The number of books written by the author")
    books: list[str] = Field(description="The list of books they wrote")

output_parser = PydanticOutputParser(pydantic_object=Author)

prompt_list = PromptTemplate.from_template(
    template = "Answer the question.\n{format_instructions}\n{question}",
    input_vairables = ["question"],
    partial_variables = {"format_instructions": output_parser.get_format_instructions()},
)

prompt_value = prompt_list.invoke({"question": "Generate the books written by Dan Brown"})
response = llm.invoke(prompt_value)

returned_object = output_parser.parse(response.content)
print(f"{returned_object.name} wrote {returned_object.number} books.")
print(returned_object.books)