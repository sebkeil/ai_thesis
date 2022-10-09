import os
import pickle

os.chdir("../")

model_path = os.getcwd() + "\\models\\v_bs16_lr1e-05_methodrandom"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("success")
