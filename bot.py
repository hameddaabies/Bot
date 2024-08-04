from flask import Flask, request, jsonify, session, render_template_string
import os
from flask_session import Session

# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'Key')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Set environment variables
os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_sk"
os.environ["LANGCHAIN_PROJECT"] = "Bargainb Multi-agent Supervisor"

from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_core.tools import tool
from algoliasearch.search_client import SearchClient
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# Define the tool for Algolia search
@tool("algolia_search", return_direct=False)
def algolia_search(query, hitsPerPage=10):
    """Returns the products after search use it one time for each search term"""
    if query.lower() == "not-related":
        return f"Sorry, your search term {query} is out-of-scope. Please try another search."
    else:
        STORE_ID_MAPPING = {
            1: 'Albert Heijn',
            2: 'Jumbo',
            3: 'Hoogvliet',
            4: 'Dirk'
        }
        
        client = SearchClient.create('DG62X9U03X', 'api')
        index = client.init_index('dev_PRODUCTS')
        attributes = ['name', 'english_name', 'price', 'old_price', 'unit', 'offer', 'store_id']
        res = index.search(query, {'attributesToRetrieve': attributes, 'hitsPerPage': hitsPerPage})
        
        formatted_results = []
        for hit in res['hits']:
            try:
                price = float(hit.get('price', 0))
                old_price = hit.get('old_price')
                if old_price is not None:
                    old_price = float(old_price)
                    discount = old_price - price
                else:
                    discount = None
                    
                store_name = STORE_ID_MAPPING.get(hit.get('store_id'), 'Unknown')
                
                hit_info = ", ".join(f"{key}: {hit.get(key, 'N/A')}" for key in attributes if key != 'store_id')
                hit_info += f", discount: {discount if discount is not None else 'N/A'}"
                hit_info += f", store_name: {store_name}"
                formatted_results.append(hit_info)
            
            except Exception as e:
                return f"Error processing hit: {e}"

        formatted_output = f"Algolia Search Results for '{query}':\n\n" + "\n".join(formatted_results)
        return formatted_output

tools = [algolia_search]

# Setup the LangChain agent
llm = ChatOpenAI(model="gpt-4")

def get_react_prompt_template():
    return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}
format the output in production way
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

prompt_template = get_react_prompt_template()

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Define the chat API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    user_input = request.json.get('input')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Add the user input to the conversation history
    session['conversation_history'].append({"role": "user", "content": user_input})

    # Pass the conversation history to the agent
    response = agent_executor.invoke({"input": user_input})
    
    # Add the response to the conversation history
    session['conversation_history'].append({"role": "assistant", "content": response['output']})

    return jsonify({"output": response['output']}), 200

# Define a simple web page to interact with the chat
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BargainB Bot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .chat-container {
            max-width: 600px;
            margin: 0 auto;
            background: #fff;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        input[type="text"] {
            width: calc(100% - 90px);
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            color: #fff;
            background-color: #007BFF;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        p#response {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
        }
    </style>
    <script>
        async function sendMessage() {
            const input = document.getElementById("userInput").value;
            if (input.trim() === "") {
                alert("Please enter a message before sending.");
                return;
            }
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ input })
                });
                const data = await response.json();
                document.getElementById("response").textContent = "Bot: " + data.output;
            } catch (error) {
                console.error("Error:", error);
                document.getElementById("response").textContent = "An error occurred. Please try again.";
            }
        }
    </script>
</head>
<body>
    <div class="chat-container">
        <h1>Chat with BargainB Bot</h1>
        <input type="text" id="userInput" placeholder="Type your message here...">
        <button onclick="sendMessage()">Send</button>
        <p id="response"></p>
    </div>
</body>
</html>
 ''')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
