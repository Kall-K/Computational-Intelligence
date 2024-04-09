import numpy as np

# Generate random data for tokenized words (40 columns)
num_samples = 1000  # Number of samples in the dataset
num_words = 40  # Number of tokenized words columns
tokenized_words = np.random.randint(0, 100, size=(num_samples, num_words))

# Generate random data for date-related columns
min_date = np.random.randint(1900, 2022, size=(num_samples, 1))  # Min date column
max_date = np.random.randint(1900, 2022, size=(num_samples, 1))  # Max date column

# Ensure min_date is less than or equal to max_date
min_date, max_date = np.minimum(min_date, max_date), np.maximum(min_date, max_date)

average_date = (min_date + max_date) // 2  # Average date column

# Normalize the date columns to [-1, 1]
min_date_norm = (min_date - 1900) / (2022 - 1900) * 2 - 1
max_date_norm = (max_date - 1900) / (2022 - 1900) * 2 - 1
average_date_norm = (average_date - 1900) / (2022 - 1900) * 2 - 1

tokenized_words_norm = (tokenized_words - np.min(tokenized_words)) / (np.max(tokenized_words) - np.min(tokenized_words)) * 2 - 1

# Concatenate tokenized words, normalized min_date, max_date, and average_date to form the dataset
dataset = np.concatenate((tokenized_words_norm, average_date_norm, min_date_norm, max_date_norm), axis=1)

# Define headers
headers = [str(i) for i in range(num_words)] + ['min_date', 'max_date', 'average_date']

# Display the shape of the dataset
print("Shape of dataset:", dataset.shape)

# Save the dataset to a CSV file with headers
np.savetxt('dataset.csv', dataset, delimiter='\t', fmt='%f', header='\t'.join(headers), comments='')