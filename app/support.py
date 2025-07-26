from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph
import json
from langgraph.types import Command

DB_URI = 'mongodb+srv://newuse1:aman123@cluster1.hn5lfzo.mongodb.net/'
config = {"configurable": {"thread_id": "6"}}
 
def init():
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
    
        state = graph_with_mongo.get_state(config=config)
        # for message in state.values['messages']:
        #     message.pretty_print()
        
        last_message = state.values['messages'][-1]
        tool_calls = last_message.additional_kwargs.get("tool_calls", [])

        user_query = None

        for call in tool_calls:
            if call.get("function", {}).get("name") == "human_assistance_tool":
                args = call["function"].get("arguments", "{}")
                try:
                    args_dict = json.loads(args)
                    user_query = args_dict.get("query")
                except json.JSONDecodeError:
                    print("Failed to decode function arguments.")
        
        print("User is Tying to Ask:", user_query)
        ans = input("Resolution > ")

        # OpenAI Call to mimic human

        resume_command = Command(resume={"data": ans})
        
        for event in graph_with_mongo.stream(resume_command, config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()

init()