import guardrail_models
import dataset_testing
import sys

def benchmarkAll():
    print("--- [Benchmarking on all datasets] ---")
    dataset_testing.MedSafetyBench()
    dataset_testing.XSTest()
    return

datasets = {
    "xstest": dataset_testing.XSTest,
    "msb": dataset_testing.MedSafetyBench,
    "psb": dataset_testing.PatientSafetyBench,
    "egm": dataset_testing.ExpGuardMix,
    "wgm": dataset_testing.WildGuardMix,
    "All": benchmarkAll,
    "Test": None
}

def main():
    input_dataset = sys.argv[1]
    input_model = sys.argv[2]
    print("Starting benchmarks...")

    if input_dataset == "Test":
        print("Starting Test")

        guardrail_models.main()

        print("Test ended")
        return
    elif input_dataset not in datasets:
        print("Input dataset not supported")
    else: datasets[input_dataset](input_model)

    print("End")

    return

main()
