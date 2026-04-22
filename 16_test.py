from mp_api.client import MPRester
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read API key from environment variable
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY:
    raise ValueError("MP_API_KEY is not set in .env file.")

task_id = "mp-8352"

with MPRester(API_KEY) as mpr:
    task_doc = mpr.materials.tasks.search(
        task_ids=[task_id],
        fields=["task_id", "calc_type", "input", "output"]
    )[0]

print("task_id =", task_doc.task_id)
print("calc_type =", task_doc.calc_type)

initial_structure = task_doc.input.structure
final_structure = task_doc.output.structure

initial_structure.to(filename="mp-8352_input_from_task.cif")
final_structure.to(filename="mp-8352_output_from_task.cif")

print("saved mp-8352_input_from_task.cif")
print("saved mp-8352_output_from_task.cif")