import re
import csv

# result_path = 'results/current_best_early_stop'
result_path = 'results/Uformer_audio-tedlium-10052023_235146'
# result_path = '../hidden/runs/audio_librispeech 2023.06.14--22-05-46'
data_file = open(result_path + '/sample_result.txt', 'r')
data = data_file.readlines()
data = ''.join(data)
print(data)

def process_data_to_csv(data):
    # Extract relevant information using regular expressions
    pattern = r"Result on (.*), attack: (.*): Total clips: (.*), MSE loss (.*), WM loss: (.*), WM loss after attack: (.*), SNR score: (.*), PESQ score: (.*)"

    results = re.findall(pattern, data)

    # Create a list of dictionaries to store the data in a structured format
    structured_data = []
    for result in results:
        structured_data.append({
            "Set": result[0],
            "Attack": result[1],
            "Total Clips": int(result[2]),
            "MSE Loss": float(result[3]),
            "WM Loss": float(result[4]),
            "WM Loss After Attack": float(result[5]),
            "SNR Score": float(result[6]),
            "PESQ Score": float(result[7])
        })

    # Write the structured data to a CSV file
    fieldnames = ["Set", "Attack", "Total Clips", "MSE Loss", "WM Loss", "WM Loss After Attack", "SNR Score", "PESQ Score"]

    with open(result_path + "/results.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in structured_data:
            writer.writerow(row)

# Call the function to process data and generate a CSV table
process_data_to_csv(data)