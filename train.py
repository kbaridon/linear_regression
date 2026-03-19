import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools


def get_new_thetas(t0: float, t1: float, lr: float, mileage, price):
	"""Function using the gradient descent derivative formula"""
	m = len(mileage)

	sum0 = 0.0
	sum1 = 0.0

	for i in range(m):
		pred = t0 + t1 * mileage[i]
		error = pred - price[i]

		sum0 += error
		sum1 += error * mileage[i]

	tmp_t0 = lr * (1/m) * sum0
	tmp_t1 = lr * (1/m) * sum1

	return t0 - tmp_t0, t1 - tmp_t1


def MSE(t0: float, t1: float, mileage: list, price: list):
	"""Mean squared error function"""
	m = len(mileage)
	total = 0.0

	for i in range(m):
		pred = t0 + t1 * mileage[i]
		error = pred - price[i]
		total += error ** 2
	return (total / m)


def gradient_descent(data: pd.DataFrame):
	"""Take a dataframe to return the optimal theta0 and theta1 of it"""
	learning_rate = 0.01  # "Pas" every iteration
	epsilon = 0.001  # Seuil de tolerance --> Si une iteration change de - de 0.001 alors on stop
	scale = max(abs(data["km"].max()), 1.0)
	km = (data["km"].astype(float) / scale).tolist()
	price = data["price"].astype(float)
	theta0 = 0.0
	theta1 = 0.0

	spinner = itertools.cycle(["◴", "◷", "◶", "◵"])
	iter_count = 0

	old_cost = MSE(theta0, theta1, km, price)
	while True:
		theta0, theta1 = get_new_thetas(theta0, theta1, learning_rate, km, price)
		new_cost = MSE(theta0, theta1, km, price)

		if not (math.isfinite(old_cost) and math.isfinite(new_cost)):
			print("\nNumerical issue detected during training; try a smaller learning rate or scale your data.")
			break

		iter_count += 1
		if iter_count % 10000 == 0:
			iter_count %= 5
			sys.stdout.write(f"\rTraining {next(spinner)}")
			sys.stdout.flush()

		if abs(old_cost - new_cost) < epsilon:
			sys.stdout.write("\rTraining ✓\n")
			break
		old_cost = new_cost

	theta1 = theta1 / scale

	return (theta0, theta1)


def evaluate_metrics(t0: float, t1: float, mileage: list, price: list):
	"""Compute and print evaluation metrics for a linear model."""
	m = len(mileage)

	preds = [t0 + t1 * xi for xi in mileage]
	errors = [preds[i] - price[i] for i in range(m)]

	ss_res = sum(e * e for e in errors)
	mse = ss_res / m
	rmse = math.sqrt(mse)
	mae = sum(abs(e) for e in errors) / m

	y_mean = sum(price) / m
	ss_tot = sum((yi - y_mean) ** 2 for yi in price)
	r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

	print("Precision metrics:")
	print(f"	MSE : {mse:.0f}")
	print(f"	RMSE: {rmse:.2f}")
	print(f"	MAE : {mae:.2f}")
	print(f"	R2  : {r2:.4f}")


def save_thetas(t0: float, t1: float):
	"""Function saving 2 floats into a text file."""
	content = f"{t0},{t1}"
	try:
		f = open("training_results.txt", "w")
		f.write(content)
		f.close()
	except (PermissionError):
		print("Not able to write in the file training_results.txt. Please update permissions.")
		sys.exit(-1)


def do_graph(data: pd.DataFrame, theta0: float, theta1: float):
	"""Function drawing a graph of a dataframe + result of linear regression"""
	x = data["km"]
	y = data["price"]

	plt.scatter(x, y, label="Dataset")
	y_pred = theta0 + theta1 * x

	plt.plot(x, y_pred, color='red', label='Linear Regression')

	plt.xlabel("Mileage (km)")
	plt.ylabel("Price")
	plt.title("Price of a car depending on its mileage")
	plt.legend()
	plt.show()


def main():
	"""Start the training with a specific csv"""
	try:
		if (len(sys.argv) != 2):
			raise ValueError
		dataset = pd.read_csv(sys.argv[1])
		print("Loading dataset...")
	except (FileNotFoundError, PermissionError, ValueError):
		print("Please load with a propoer csv.")
		sys.exit(-1)
	theta0, theta1 = gradient_descent(dataset)
	evaluate_metrics(theta0, theta1, dataset["km"].astype(float).tolist(), dataset["price"].astype(float).tolist())
	save_thetas(theta0, theta1)
	do_graph(dataset, theta0, theta1)


if __name__ == "__main__":
	main()