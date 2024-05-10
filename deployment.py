from dbgpt_hub.llm_base.chat_model import ChatModel
from dbgpt_hub.predict.predict import prepare_dataset
import gradio as gr
import random


def ask(text):

  

  # Prediction
  args = {
      "model_name_or_path": "codellama/CodeLlama-7b-Instruct-hf",
      "template": "llama2",
      "finetuning_type": "lora",
      "checkpoint_dir": "dbgpt_hub/output/adapter/CodeLlama-7b-sql-lora-25e/checkpoint-5410",
      "predict_file_path": "dbgpt_hub/data/eval_data/dev_sql.json",
      "predict_out_dir": "dbgpt_hub/output/",
      "predicted_out_filename": "pred_sql.sql",
      "quantization_bit": 4,
  }
  
  # model = ChatModel(args)
  # response, _ = model.chat(query=text, history=[])
  return text.upper()

def generate_question():
  # Preprocessing
  predict_data = prepare_dataset("dbgpt_hub/data/eval_data/dev_sql.json")
  item = random.choice(predict_data)
  input_part = item["input"].split("###Input:")[1].split("###Response:")[0].strip()

  return input_part, item["input"]

with gr.Blocks() as server:
  # Description
  # Database Schema
  with gr.Tab("Text-to-SQL for spider database"):
    generate_question_button = gr.Button("Generate Example Question")

    model_input = gr.Textbox(label="Your Question:", 
                             value="", interactive=False)
    
    full_model_input = gr.Textbox(label="Full model input:", 
                          value="", interactive=False)

    ask_button = gr.Button("Ask")
    model_output = gr.Textbox(label="The SQL query:", 
                              interactive=False, value="")
    
    database_answer = gr.Textbox(label="The database answer:", 
                              interactive=False, value="")
    
  generate_question_button.click(generate_question, outputs=[model_input, full_model_input])
  ask_button.click(ask, inputs=[full_model_input], outputs=[model_output])

server.launch()