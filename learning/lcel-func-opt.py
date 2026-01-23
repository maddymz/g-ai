from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

parse_template = PromptTemplate(
    input_variables=["raw_feedback"],
    template="Parse and clean the following customer feedback for key information:\n\n{raw_feedback}"
)

summary_template = PromptTemplate(
    input_variables=["parsed_feedback"],
    template="Summarize this customer feedback in one concise sentence:\n\n{parsed_feedback}"
)

sentiment_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word as either 'Positive', 'Neutral', or 'Negative':\n\n{feedback}"
)

# Create a function
def format_print_output(key_name, output):
  print (key_name, ":", output.content, '\n')
  return {key_name: output}
  
# Convert the funtion into a RunnableLambda
format_parsed_output = RunnableLambda(lambda output: format_print_output("parsed_feedback" ,output))
format_summary_output = RunnableLambda(lambda output: format_print_output("feedback", output))

# Neutral
user_feedback = "The delivery was late, and the product was damaged when it arrived. However, the customer support team was very helpful in resolving the issue quickly."

# Postive
# user_feedback = "The customer service was fantastic. The representative was friendly, knowledgeable, and resolved my issue quickly."

# Negative
# user_feedback = "I was extremely disappointed with the customer service. The representative was unhelpful and rude."

chain = parse_template | llm | format_parsed_output | summary_template | llm | format_summary_output | sentiment_template| llm | StrOutputParser()
feedback_sentiment = chain.invoke({"raw_feedback": user_feedback})

print(feedback_sentiment)