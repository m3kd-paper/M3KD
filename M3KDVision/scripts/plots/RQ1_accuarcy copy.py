import os
import json
import glob
eval_path = "evalplus_results/humaneval"


one_epoch_models = []
one_epoch_models_2ndtime = []
real_2_epoch_models = []
SFT_models = []
QV = []
selecTKD = []
_1000_models = []
orth = []

json_files = glob.glob(os.path.join(eval_path, "*eval_results.json"))
for json_file in json_files:
    with open(json_file, "r") as f:
        json_content = json.load(f)
        acc = json_content["pass_at_k"]

        pass_at_1_base = acc["base"]["pass@1"]
        pass_at_1_plus = acc["plus"]["pass@1"]

        if "qwen2.5-coder-1.5b-distillm" in json_file or "Top" in json_file or "HPD" in json_file:
            one_epoch_models.append((json_file, pass_at_1_base, pass_at_1_plus))
        elif "qwen2.5-coder-1.5b-2epochs-distillm" in json_file:
            one_epoch_models_2ndtime.append((json_file, pass_at_1_base, pass_at_1_plus))
        elif "Real2epoch" in json_file:
            real_2_epoch_models.append((json_file, pass_at_1_base, pass_at_1_plus))
        elif "sft" in json_file:
            SFT_models.append((json_file, pass_at_1_base, pass_at_1_plus))
        if "QV" in json_file:
            QV.append((json_file, pass_at_1_base, pass_at_1_plus))
        if "Selec" in json_file:
            selecTKD.append((json_file, pass_at_1_base, pass_at_1_plus))
        if "orth" in json_file and "Real2epoch" not in json_file and "QV" not in json_file:
            orth.append((json_file, pass_at_1_base, pass_at_1_plus))
print("#---------------#")
print("One epoch models:")
print("#---------------#")
one_epoch_models.sort(key=lambda x: x[1], reverse=True)
for json_file, pass_at_1_base, pass_at_1_plus in one_epoch_models:
    print("------------------------------")
    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")
#
#print()
#print()
#
#print("#---------------#")
#print("One epoch models (2ndtime):")
#print("#---------------#")
#one_epoch_models_2ndtime.sort(key=lambda x: x[1], reverse=True)
#for json_file, pass_at_1_base, pass_at_1_plus in one_epoch_models_2ndtime:
#    print("------------------------------")
#    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")
#
#
#print()
#print()
#
#print("#---------------#")
#print("Real 2 epoch models:")
#print("#---------------#")
#real_2_epoch_models.sort(key=lambda x: x[1], reverse=True)
#for json_file, pass_at_1_base, pass_at_1_plus in real_2_epoch_models:
#    print("------------------------------")
#    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")


print()
print()
print("#---------------#")
print("SFT models:")
print("#---------------#")
SFT_models.sort(key=lambda x: x[1], reverse=True)
for json_file, pass_at_1_base, pass_at_1_plus in SFT_models:
    print("------------------------------")
    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")


#print()
#print()
#print("#---------------#")
#print("QV models:")
#print("#---------------#")
#QV.sort(key=lambda x: x[1], reverse=True)
#for json_file, pass_at_1_base, pass_at_1_plus in QV:
#    print("------------------------------")
#    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")
#
#print()
#print()
#print("#---------------#")
#print("SelecTKD models:")
#print("#---------------#")
#selecTKD.sort(key=lambda x: x[1], reverse=True)
#for json_file, pass_at_1_base, pass_at_1_plus in selecTKD:
#    print("------------------------------")
#    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")

#print()
#print()
#print("#---------------#")
#print("1000 models:")
#print("#---------------#")
#_1000_models.sort(key=lambda x: x[1], reverse=True)
#for json_file, pass_at_1_base, pass_at_1_plus in _1000_models:
#    print("------------------------------")
#    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")


print() 
print()
print("#---------------#")
print("Orth models:")
print("#---------------#")
orth.sort(key=lambda x: x[1], reverse=True)
for json_file, pass_at_1_base, pass_at_1_plus in orth:
    print("------------------------------")
    print(f"File: {json_file}:\n Pass@1 Base: {pass_at_1_base}, Pass@1 Plus: {pass_at_1_plus}")