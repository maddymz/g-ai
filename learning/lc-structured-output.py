from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Keeping the same model; setting temperature low can help with format adherence.
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

class Author(BaseModel):
    name: str = Field(description="The name of the author")
    number: int = Field(description="The number of books written by the author")
    books: list[str] = Field(description="The list of books they wrote")

structured_llm = llm.with_structured_output(Author)

# A tiny nudge in the prompt helps many models obey types without extra code.
returned_object = structured_llm.invoke(
    "Generate the books written by Dan Brown. "
    "Return 'number' as an integer (not a string) and 'books' as a JSON array of strings (not a quoted string)."
)

print(f"{returned_object.name} wrote {returned_object.number} books.")
print(returned_object.books)