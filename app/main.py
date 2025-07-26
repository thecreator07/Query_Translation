# from graph import graph
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph 

DB_URI='mongodb+srv://newuse1:aman123@cluster1.hn5lfzo.mongodb.net/'
config = {"configurable": {"thread_id": "6"}}
def init():
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph_with_mongo=create_chat_graph(checkpointer=checkpointer)
    
        while True:
            user_input=input("Ask question: ")
            # result=graph.invoke({"messages":[{"role":"user","content":user_input}]})
            for event in graph_with_mongo.stream({"messages":[{"role":"user","content":user_input}]},stream_mode="values",config=config):
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        
init()