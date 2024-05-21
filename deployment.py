from dbgpt_hub.llm_base.chat_model import ChatModel
from dbgpt_hub.predict.predict import prepare_dataset
import gradio as gr
import random
import json


def ask(text):

  

  # Prediction
  args = {
      "model_name_or_path": "codellama/CodeLlama-7b-Instruct-hf",
      "template": "llama2",
      "finetuning_type": "lora",
      "checkpoint_dir": "dbgpt_hub/output/adapter/CodeLlama-7b-sql-lora-25e/checkpoint-5410",
      "predicted_out_filename": "pred_sql.sql",
     # "quantization_bit": 4,
  }
  
  model = ChatModel(args) # TODO: See if we can use TensorRT to speed this up
  response, _ = model.chat(query=text, history=[])
  return response

def generate_question():
  # Preprocessing
  with open('dbgpt_hub/data/eval_data/dev_sql.json') as f:
    data = json.load(f) # TODO: Fix so that the capitalization is correct, get the line from the sql file instead
  predict_data = prepare_dataset("dbgpt_hub/data/eval_data/dev_sql.json")
  index = random.randint(0, len(predict_data)-1)
  item = predict_data[index]
  print(item)
  print(f"Index: {index}")
  answer = data[index]
  print(answer)
  input_part = item["input"].split("###Input:")[1].split("###Response:")[0].strip()

  return input_part, item["input"], answer['output']

def run_sql(gold_query, pred_query): # TODO: Fix this evaluation step, present results from queries and execution acccuracy
  return gold_query, pred_query

with gr.Blocks() as server:
  # TODO: Add Description
  # TODO: Add Database Schema
  with gr.Tab("Text-to-SQL for spider database"):
    generate_question_button = gr.Button("Generate Example Question")

    model_input = gr.Textbox(label="Your Question:", 
                             value="", interactive=False)
    
    full_model_input = gr.Textbox(label="Full model input:", 
                          value="", interactive=False)

    ask_button = gr.Button("Ask")
    model_output = gr.Textbox(label="The SQL query:", 
                              interactive=False, value="")
    gold_sql = gr.Textbox(label="The SQL gold query:", 
                              interactive=False, value="")
    
    run_queries = gr.Button("Run SQL queries against the Spider database")

    database_answer_gold = gr.Textbox(label="The gold query database answer:", 
                              interactive=False, value="")
    
    database_answer_pred = gr.Textbox(label="The predicted query database answer:", 
                          interactive=False, value="")
    
  generate_question_button.click(generate_question, outputs=[model_input, full_model_input, gold_sql])
  ask_button.click(ask, inputs=[full_model_input], outputs=[model_output])
  run_queries.click(run_sql, inputs=[gold_sql, model_output], outputs=[database_answer_gold, database_answer_pred])

server.launch()