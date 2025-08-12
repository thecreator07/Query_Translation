# from graph import graph
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph 
import os 
import speech_recognition as sr

DB_URI=os.environ.get('DB_URL')
config = {"configurable": {"thread_id": "10"}}
def init():
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph_with_mongo=create_chat_graph(checkpointer=checkpointer)
        r=sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold=2
            
            while True:
                print("Say Something")
                audio=r.listen(source)
                try:
                    print("Processing audio...")
                    sst=r.recognize_google(audio)
                    
                    print("You Said: ",sst)
                    # result=graph.invoke({"messages":[{"role":"user","content":user_input}]})
                    for event in graph_with_mongo.stream({"messages":[{"role":"user","content":sst}]},stream_mode="values",config=config):
                        if "messages" in event:
                            event["messages"][-1].pretty_print()
                except sr.UnknownValueError:
                    print("Sorry, coul not understand")
                except sr.RequestError as e:
                    print(f"API request error: {e}")                
        
init()