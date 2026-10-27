import os
import pandas as pd
import json
import sys
from tabulate import tabulate, tabulate_formats

headers1 = []
headers2 = []

def create_table(filename):
    with open(f"results/analysis_output/{filename}.json", "r") as f:  # Load results stored in the .json file
        results = [json.loads(line) for line in f if line.strip()]  # Due to the fact that the file contains a json object per line, instead of a single json
    #print(results[0].keys())

    table1 = []
    table2 = []

    '''
    check_TP = False
    check_TN = False
    for result in results:
        if result['TP'] != 0:
            check_TP = True
        if result['TN'] != 0:
            check_TN = True
    '''

    check_TP = any(result['TP'] != 0 for result in results)
    check_TN = any(result['TN'] != 0 for result in results)

    if (check_TN and check_TP):
        for result in results:
            table1.append(
                [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['TP'], result['TN'], result['FP'], result['FN'], result['FPR'],
                 result['FNR']])
            table2.append(
                [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['Accuracy'], result['Specificity'], result['Precision'],
                 result['Recall'], result['F1Score'],
                 result['MCC']])

        headers1 = ['Model', 'Logic', 'TP', 'TN', 'FP', 'FN', 'FPR', 'FNR']
        headers2 = ['Model', 'Logic', 'Accuracy', 'Specificity', 'Precision', 'Recall', 'F1Score', 'MCC']
    elif (check_TP and not check_TN):
        for result in results:
            table1.append(
                [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['TP'], result['FN'], result['FNR']])
            table2.append(
                [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['Accuracy'], result['Recall'], result['F1Score']])

        headers1 = ['Model', 'Logic', 'TP', 'FN', 'FNR']
        headers2 = ['Model', 'Logic', 'Accuracy', 'Recall', 'F1Score']
    elif (not check_TP and check_TN):
        for result in results:
            table1.append(
                [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['TN'], result['FP'], result['FPR']])
            table2.append(
                [result['Model'], result['Logic'] if result['Logic'] is not None else "-", result['Accuracy'], result['Specificity'], result['F1Score']])

        headers1 = ['Model', 'Logic', 'TN', 'FP', 'FPR']
        headers2 = ['Model', 'Logic', 'Accuracy', 'Specificity', 'F1Score']
    else:
        print("Error in checking the number of TPs and TNs")

    # Convert to data frame for further processing (bold best result per category)
    df1 = pd.DataFrame(table1, columns=headers1)
    df2 = pd.DataFrame(table2, columns=headers2)

    # Calculate best values for table 2
    for col in ['Accuracy', 'Specificity', 'Precision', 'Recall', 'F1Score', 'MCC']:
        if col not in df2.columns:
            continue
        best = df2[col].max()
        worst = df2[col].min()
        df2[col] = df2[col].apply(
            lambda x: "--" if x == None
            else f"\\textbf{{{x * 100:.2f}}}" if x == best
            else f"\\underline{{{x * 100:.2f}}}" if x == worst
            else f"{x * 100:.2f}"
        )

    # Calculate best FPR and best FNR for table 1: Lower is better
    for col in ['FPR', 'FNR']:
        if col not in df1.columns:
            continue
        best = df1[col].min()
        worst = df1[col].max()
        df1[col] = df1[col].apply(
            lambda x: "--" if x == None
            else f"\\textbf{{{x * 100:.2f}}}" if x == best
            else f"\\underline{{{x * 100:.2f}}}" if x == worst
            else f"{x * 100:.2f}"
        )

    #table1 = tabulate(df1, headers1, tablefmt="latex_booktabs_raw", showindex=False)
    #table2 = tabulate(df2, headers2, tablefmt="latex_booktabs_raw", showindex=False)

    table1 = tabulate(df1, headers1, tablefmt="latex_raw", showindex=False)
    table2 = tabulate(df2, headers2, tablefmt="latex_raw", showindex=False)
    return table1, table2, headers1, headers2

def create_table_single(filename):
    with open(f"results/analysis_output/{filename}.json", "r") as f:  # Load results stored in the .json file
        results = [json.loads(line) for line in f if line.strip()]  # Due to the fact that the file contains a json object per line, instead of a single json
    #print(results[0].keys())

    table1 = []
    table2 = []

    check_TP = any(result['TP'] != 0 for result in results)
    check_TN = any(result['TN'] != 0 for result in results)

    if (check_TN and check_TP):
        for result in results:
            table1.append(
                [result['Model'], result['TP'], result['TN'], result['FP'], result['FN'], result['FPR'],
                 result['FNR'], result['AvgTime']])
            table2.append(
                [result['Model'], result['Accuracy'], result['Specificity'], result['Precision'],
                 result['Recall'], result['F1Score'],
                 result['MCC']])

        headers1 = ['Model', 'TP', 'TN', 'FP', 'FN', 'FPR', 'FNR', 'AvgTime (s)']
        headers2 = ['Model', 'Accuracy', 'Specificity', 'Precision', 'Recall', 'F1Score', 'MCC']
    elif (check_TP and not check_TN):
        for result in results:
            table1.append(
                [result['Model'], result['TP'], result['FN'], result['FNR'], result['AvgTime']])
            table2.append(
                [result['Model'], result['Accuracy'], result['Recall'], result['F1Score']])

        headers1 = ['Model', 'TP', 'FN', 'FNR', 'AvgTime (s)']
        headers2 = ['Model', 'Accuracy', 'Recall', 'F1Score']
    elif (not check_TP and check_TN):
        for result in results:
            table1.append(
                [result['Model'], result['TN'], result['FP'], result['FPR'], result['AvgTime']])
            table2.append(
                [result['Model'], result['Accuracy'], result['Specificity'], result['F1Score']])

        headers1 = ['Model', 'TN', 'FP', 'FPR', 'AvgTime (s)']
        headers2 = ['Model', 'Accuracy', 'Specificity', 'F1Score']
    else:
        print("Error in checking the number of TPs and TNs")

    # Convert to data frame for further processing (bold best result per category)
    df1 = pd.DataFrame(table1, columns=headers1)
    df2 = pd.DataFrame(table2, columns=headers2)

    # Calculate best values for table 2
    for col in ['Accuracy', 'Specificity', 'Precision', 'Recall', 'F1Score', 'MCC']:
        if col not in df2.columns:
            continue
        best = df2[col].max()
        worst = df2[col].min()
        df2[col] = df2[col].apply(
            lambda x: "--" if x == None
            else f"\\textbf{{{x * 100:.2f}}}" if x == best
            else f"\\underline{{{x * 100:.2f}}}" if x == worst
            else f"{x * 100:.2f}"
        )

    # Calculate best FPR and best FNR for table 1: Lower is better
    for col in ['FPR', 'FNR']:
        if col not in df1.columns:
            continue
        best = df1[col].min()
        worst = df1[col].max()
        df1[col] = df1[col].apply(
            lambda x: "--" if x == None
            else f"\\textbf{{{x * 100:.2f}}}" if x == best
            else f"\\underline{{{x * 100:.2f}}}" if x == worst
            else f"{x * 100:.2f}"
        )

    #table1 = tabulate(df1, headers1, tablefmt="latex_booktabs_raw", showindex=False)
    #table2 = tabulate(df2, headers2, tablefmt="latex_booktabs_raw", showindex=False)

    table1 = tabulate(df1, headers1, tablefmt="latex_raw", showindex=False)
    table2 = tabulate(df2, headers2, tablefmt="latex_raw", showindex=False)
    return table1, table2, headers1, headers2

def extract_rows(tabulate_output):
    lines = tabulate_output.strip().split("\n")
    return "\n".join(lines[4:-2])

def printTables(dataset, single = False):
    if single:
        table1, table2, headers1, headers2 = create_table_single(f"{dataset}_score_results")
        content1 = extract_rows(table1)
        content2 = extract_rows(table2)
        full_table1 = generate_latex_table(content1, headers1, f"{dataset} - Table 1", f"tab:{dataset.lower()}_t1", long=False)
        full_table2 = generate_latex_table(content2, headers2, f"{dataset} - Table 2", f"tab:{dataset.lower()}_t2", long=False)
    else:
        table1, table2, headers1, headers2 = create_table(f"{dataset}_score_results")
        content1 = extract_rows(table1)
        content2 = extract_rows(table2)
        full_table1 = generate_latex_table(content1, headers1, f"{dataset} - Table 1", f"tab:{dataset.lower()}_t1", long=True)
        full_table2 = generate_latex_table(content2, headers2, f"{dataset} - Table 2", f"tab:{dataset.lower()}_t2", long=True)

    print(f"\n\t*** Dataset: {dataset} - TABLE 1:***\n")
    print(full_table1)
    print(f"\n\t*** Dataset: {dataset} - TABLE 2: ***\n")
    print(full_table2)

    '''
    print(f"\n\t*** Dataset: {dataset} - TABLE 1:***\n")
    print(f"Headers:{headers1}")
    print(table1)
    print(f"\n\t*** Dataset: {dataset} - TABLE 2: ***\n")
    print(f"Headers:{headers2}")
    print(table2)
    '''
    return

def generate_latex_table(content, headers, caption, label, long=False):
    # Build column spec: 'l' for text columns (Model, Logic), 'r' for numeric ones
    col_spec = ""
    for h in headers:
        if h in ('Model', 'Logic'):
            col_spec += "l"
        else:
            col_spec += "r"

    # Build header row with (\%) suffix for numeric columns
    header_row = " & ".join(
        f"{h}(\\%)" if h in ('FPR', 'FNR', 'Accuracy', 'Specificity', 'Precision', 'Recall', 'F1Score', 'MCC') else h
        for h in headers
    ) + " \\\\"

    num_cols = len(headers)

    if long:
        table = f"""{{\\small
\\begin{{longtable}}{{{col_spec}}}
\\caption{{{caption}}} \\label{{{label}}} \\\\
\\toprule
{header_row}
\\midrule
\\endfirsthead
\\multicolumn{{{num_cols}}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued from previous page}} \\\\
\\toprule
{header_row}
\\midrule
\\endhead
\\midrule
\\multicolumn{{{num_cols}}}{{r}}{{Continued on next page}} \\\\
\\endfoot
\\bottomrule
\\endlastfoot
{content}
\\end{{longtable}}
}}"""
    else:
        table = f"""\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header_row}
\\midrule
{content}
\\bottomrule
\\end{{tabular}}
\\caption{{\\label{{{label}}}{caption}}}
\\end{{table}}"""

    return table

def printAllTables():
    datasets_single_models = ["XSTest(single)", "MedSafetyBench(single)", "PatientSafetyBench(single)", "ExpGuardMix_Input(single)", "ExpGuardMix_InputOutput(single)", "WildGuardMix(single)"]
    datasets = ["XSTest", "MedSafetyBench", "PatientSafetyBench", "ExpGuardMix_Input", "ExpGuardMix_InputOutput", "WildGuardMix"]
    # Print all single-models tables
    print("\n\t--- SINGLE-MODEL TABLES: ---")
    for dataset in datasets_single_models:
        printTables(dataset, True)

    input("Press Enter to continue...")
    os.system('clear')

    print("\n\t--- SINGLE-MODEL PER-CATEGORY TABLES: ---")
    # Print all per-category tables
    for dataset in datasets:
        path = f"results/analysis_output/{dataset}Categories/"
        if os.path.exists(path):
            for file in os.scandir(path):
                if file.name.endswith(".json"):
                    file = file.name.removesuffix("_score_results.json")
                    file = f"{dataset}Categories/{file}"
                    printTables(file, True)
        input("Press Enter to continue...")
        os.system('clear')

    print("\n\t--- LONG TABLES: ---")
    for dataset in datasets:
        dataset = f"{dataset}"
        printTables(dataset)
    return

def main():
    #dataset = "XSTest"
    #dataset = "MedSafetyBench"
    #dataset = "PatientSafetyBench"
    #dataset = "ExpGuardMix_Input"
    #dataset = "ExpGuardMix_InputOutput"

    #printTables(dataset)

    printAllTables()

    """
    # Print all single-models tables
    for dataset in datasets_single_models:
        printTables(dataset)
    """



    '''
    headers = ["Uno", "Due", "Tre"]
    table = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
    print("\t***\n")
    print(tabulate(table, headers, tablefmt="latex_booktabs"))
    '''

    '''
    # Table template
    
    \begin{table}[h]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular} ... \end{tabular}
    }
    \caption{Your caption here}
    \label{tab:your_label}
    \end{table}
    '''

    """
# Long table template (with small)
{\small
\begin{longtable}{llrrrrrr}
\caption{5 Models benchmark on XSTest} \label{tab:5models} \\
 
\toprule
Model & Logic & TP & TN & FP & FN & FPR & FNR \\
\midrule
\endfirsthead
 
\multicolumn{8}{c}{\tablename\ \thetable{} -- continued from previous page} \\
\toprule
Model & Logic & TP & TN & FP & FN & FPR & FNR \\
\midrule
\endhead
 
\midrule
\multicolumn{8}{r}{Continued on next page} \\
\endfoot
 
\bottomrule
\endlastfoot
        # HERE CONTENT OF TABLE
\end{longtable}
}
    """

    return

main()

"""
\begin{table}[H]
\centering
\begin{tabular}{llrrrrrr}
\toprule
 Model          & Logic   &   TP &   TN &   FP &   FN &   FPR &   FNR \\
\midrule

# CONTENT HERE

\bottomrule
\end{tabular}
\caption{\label{tab:widgets}XSTEST Table 1}
\end{table}

\begin{table}[H]
\begin{tabular}{llrrrrrr}
\toprule
 Model          & Logic   &   Accuracy &   Specificity &   Precision &   Recall &   F1Score &      MCC \\
\midrule

# CONTENT HERE

\bottomrule
\end{tabular}
\caption{\label{tab:xstest2}XSTEST Table 2}
\end{table}

%--------------------------------------------------

"""