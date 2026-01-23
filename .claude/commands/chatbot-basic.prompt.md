### Create a chatbot
The goal of this exercise is to provide understanding of the basics of starting a multi-turn chat interaction using the OpenAI API. The exercise will involve using the gpt-3.5-turbo model to interact with the API.

### Task
We are going to implement and create an interaction with the model that will involve asking questions related to a certain topic, which in our case is programming. The goal is to see if the history of the conversation will be maintained with respect to the context.

We can use the following steps to implement this task.

### Steps
1 : Begin by importing the openai library and setting the API key.
2 : Next, initialize the system message, which helps set the role and behavior of the assistant. In this case, we are instructing it to behave like a programming expert.
3: Now we need to define the user input, which will be the question we want to ask the model.
  For example: 
  ```
  user_qestions = [
    "What's a popular choice for a first programming language?",
    "What are some advantages of learning it as my first language?",
    "Can you show me a simple 'Hello World' program written in that language?",
    ]
  ```

4: From the above user_questions list, the first question is used to establish a topic of discussion. The second question builds upon the first by seeking more information on the topic. The third question then further builds upon the previous discussion by requesting a practical example related to topic in discussion.
5: Next, we are going to create a loop that will iterate through the user questions. The continuation of context is maintained by keeping the entire conversation history in the messages list. After that, we will capture the user's question and append it to the conversation history.

This simulates a chatbot where a user types in questions one at a time.

6: Finally, we will add the assistant's response to the list of messages and print the response.