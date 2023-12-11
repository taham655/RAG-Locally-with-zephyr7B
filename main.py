from QuestionAnsweringSystem import QuestionAnsweringSystem
from ingest import ingest 


create = ingest("Muhammad Taha Profile.pdf")
qa_system = QuestionAnsweringSystem(model_path="zephyr-7b-beta.Q5_K_S.gguf")  # Creating an instance of the QA system


while True:
    query = input("Enter query: ")  # Taking user input for the query
    response = qa_system.answer_query(query)  # Getting the response
    if response == "Exiting...":  # Check if the response indicates an exit
        break  # Break the loop to exit
    print(response)  # Print the response
