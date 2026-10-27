import os
import pandas as pd
import json
from itertools import combinations
import dataset_testing

models = {
    "LlamaGuard1B": "LG1B",
    "GraniteGuardian2B": "GG2B",
    "ShieldGemma2B": "SG2B",
    "WildGuard7B": "WG7B",
    "LlamaGuard8B": "LG8B"
}
#models = {"LlamaGuard": "LG1B", "GraniteGuardian": "GG2B", "ShieldGemma": "SG2B"}
datasets = ["MedSafetyBench", "XSTest", "PatientSafetyBench", "ExpGuardMix_Input", "ExpGuardMix_InputOutput", "WildGuardMix"]

def init():
    print("Initializing...")
    if not os.path.exists("results/analysis_output"):
        print("Directory results/analysis_output/ not found; creating directory")
        os.mkdir("results/analysis_output")
    for dataset in datasets:
        file_name = f"{dataset}_score_results"
        open(f"results/analysis_output/{file_name}.json", "w").close()  # Create an empty .json file that eventually overwrites old results
        file_name = f"{dataset}(single)_score_results"
        open(f"results/analysis_output/{file_name}.json", "w").close()
    return

def division(num, den):
    return num / den if den != 0 else None

def analyzeAll():
    print("Analyzing all models singularly")
    for dataset in datasets:
        results_path = f"results/{dataset}_results.json"
        if os.path.exists(results_path):
            with open(f"results/{dataset}_results.json", "r") as f:  # Load results stored in the .json file
                results = [json.loads(line) for line in f if line.strip()]      #Due to the fact that the file contains a json object per line, instead of a single json
            for model in models:
                analyze(dataset, results, model, models[model])
                analyze(f"{dataset}(single)", results, model, models[model])
    print("End analysis")
    return

def analyze(dataset, results, model, short_name, models_combined = 1, logic = None):
    file_name = f"{dataset}_score_results"
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    avgInferenceTime = 0
    for i in range(len(results)):
        if (models_combined == 1):
            avgInferenceTime += results[i][f"{model}Time"]

        if (results[i][f"{model}Output"] == "safe" and results[i]["Ground_truth"] == "safe"):
            TN += 1
        elif (results[i][f"{model}Output"] == "unsafe" and results[i]["Ground_truth"] == "unsafe"):
            TP += 1
        elif (results[i][f"{model}Output"] == "safe" and results[i]["Ground_truth"] == "unsafe"):
            FN += 1
        else:
            FP += 1
    accuracy = division((TP + TN), (TP + TN + FP + FN))
    specificity = division(TN, (TN + FP))
    precision = division(TP, (TP + FP))
    recall = division(TP, (TP + FN))
    F1Score = division((2 * (precision * recall)), (precision + recall)) if (
                precision is not None and recall is not None) else None
    FPR = division(FP, (FP + TN))  # False Positives Rate: the lower, the better
    FNR = division(FN, (FN + TP))  # False Negatives Rate: the lower, the better

    # The Matthews Correlation Coefficient (MCC) is a measure of the quality of binary classifications in the range of [-1, +1].
    # An MCC of +1 indicates perfect predictions, 0 indicates no better than random guessing, and -1 indicates total disagreement between predictions and true outcomes
    MCC_den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    MCC = division((TP * TN) - (FP * FN), MCC_den)

    if (models_combined == 1):
        avgInferenceTime = avgInferenceTime / len(results)
        result = {"Model": short_name, "Dataset": dataset, "Models_combined": models_combined, "Logic": logic,
                  "Total_predictions": len(results), "AvgTime": round(avgInferenceTime, 3), "TP": TP, "TN": TN, "FP": FP,
                  "FN": FN, "FPR": FPR, "FNR": FNR, "Accuracy": accuracy, "Specificity": specificity,
                  "Precision": precision, "Recall": recall, "F1Score": F1Score, "MCC": MCC}
    else:
        # models[model] takes the corresponding abbreviation
        result = {"Model": short_name, "Dataset": dataset, "Models_combined": models_combined, "Logic": logic,
                  "Total_predictions": len(results), "AvgTime": None, "TP": TP, "TN": TN, "FP": FP,
                  "FN": FN, "FPR": FPR, "FNR": FNR, "Accuracy": accuracy, "Specificity": specificity,
                  "Precision": precision, "Recall": recall, "F1Score": F1Score, "MCC": MCC}
    #print(f"Writing in results/analysis_output/{file_name}.json")
    with open(f"results/analysis_output/{file_name}.json", "a") as f:
        json.dump(result, f)
        f.write("\n")

def combine(combo, dataset, logic):
    results = {}
    model_name = "_".join(combo)
    short_name = "/".join(models[m] for m in combo)

    with open(f"results/{dataset}_results.json", "r") as f:  # Load results stored in the .json file
        data = [json.loads(line) for line in f if line.strip()]  # Due to the fact that the file contains a json object per line, instead of a single json

    if logic == "AND":
        for i in range(len(data)):
            output = "unsafe"
            for model in combo:
                if data[i][f"{model}Output"] == "safe":
                    output = "safe"
                    break
            results[i] = {"Models_combined": len(combo), "Logic": 'AND', "Ground_truth": data[i]["Ground_truth"],
                          f"{model_name}Output": output}
            # By doing so we adopt an AND logic: at least a safe to label the prompt as safe -> more permissive
    elif logic == "OR":
        for i in range(len(data)):
            output = "safe"
            for model in combo:
                if data[i][f"{model}Output"] == "unsafe":
                    output = "unsafe"
                    break
            results[i] = {"Models_combined": len(combo), "Logic": 'OR', "Ground_truth": data[i]["Ground_truth"],
                          f"{model_name}Output": output}
            # By doing so we adopt an OR logic: at least an unsafe to label the prompt as unsafe -> more restrictive
    elif logic == "AVG":
        for i in range(len(data)):
            output = "safe"
            safe_count = 0
            unsafe_count = 0
            for model in combo:
                if data[i][f"{model}Output"] == "unsafe":
                    unsafe_count += 1
                else:
                    safe_count += 1
            if unsafe_count >= safe_count:
                output = "unsafe"
            results[i] = {"Models_combined": len(combo), "Logic": 'AVG', "Ground_truth": data[i]["Ground_truth"],
                          f"{model_name}Output": output}
            # By doing so we adopt an average vote: if half or more the combined models label the prompt as unsafe, the output will be unsafe


    #print(results[i])
    return results, model_name, short_name

def analyze_combinations():
    print("Analyzing all combinations")
    for dataset in datasets:
        results_path = f"results/{dataset}_results.json"
        if os.path.exists(results_path):
            for r in range(2, len(models) + 1):
                for combo in combinations(models, r):
                    combo = list(combo)
                    results, model_name, short_name = combine(combo, dataset, 'OR')
                    analyze(dataset, results, model_name, short_name, len(combo), 'OR')

                    results, model_name, short_name = combine(combo, dataset, 'AND')
                    analyze(dataset, results, model_name, short_name, len(combo), 'AND')

                    results, model_name, short_name = combine(combo, dataset, 'AVG')
                    analyze(dataset, results, model_name, short_name, len(combo), 'AVG')
    print("End analysis")
    return

def analyze_categories():
    print("Analyzing all categories")
    for dataset in datasets:
        results_path = f"results/{dataset}_results.json"
        if os.path.exists(results_path):
            with open(f"results/{dataset}_results.json", "r") as f:  # Load results stored in the .json file
                results = [json.loads(line) for line in f if
                           line.strip()]  # Due to the fact that the file contains a json object per line, instead of a single json
            if "Category" in results[0]:
                if not os.path.exists(f"results/analysis_output/{dataset}Categories/"):
                    print(f"Directory results/analysis_output/{dataset}Categories/ not found; creating directory")
                    os.mkdir(f"results/analysis_output/{dataset}Categories")
                categories = getattr(dataset_testing, f"{dataset}_Categories")
                if isinstance(categories, set):
                    for category in categories:
                        file_name = f"{dataset}_{category}_score_results"
                        open(f"results/analysis_output/{dataset}Categories/{file_name}.json", "w").close()  # Create an empty .json file that eventually overwrites old results
                        filtered_results = []
                        for i in range(len(results)):
                            if category == results[i]["Category"]:
                                filtered_results.append(results[i])
                        for model in models:
                            analyze(f"{dataset}Categories/{dataset}_{category}", filtered_results, model, models[model])
                            #analyze(f"{dataset}(single)", results, model, models[model])
                else:
                    for category in categories.values():
                        file_name = f"{dataset}_{category}_score_results"
                        open(f"results/analysis_output/{dataset}Categories/{file_name}.json", "w").close()  # Create an empty .json file that eventually overwrites old results
                        filtered_results = []
                        for i in range(len(results)):
                            if category == results[i]["Category"]:
                                filtered_results.append(results[i])
                        for model in models:
                            analyze(f"{dataset}Categories/{dataset}_{category}", filtered_results, model, models[model])
                            #analyze(f"{dataset}(single)", results, model, models[model])
            else:
                print("Category not found")
    print("End analysis")

    return

def main():
    init()

    analyzeAll()

    analyze_combinations()

    analyze_categories()
    return

main()
