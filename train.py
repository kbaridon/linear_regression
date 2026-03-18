import sys
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_new_thetas(t0: float, t1: float, lr: float, mileage, price):
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
	m = len(mileage)
	total = 0.0

	for i in range(m):
		pred = t0 + t1 * mileage[i]
		error = pred - price[i]
		total += error ** 2
	return (total / (2 * m))


def gradient_descent(data: pd.DataFrame):
	learning_rate = 0.01  # Pas a chaque iteration
	epsilon = 0.001  # Seuil de tolerance --> Si une iteration change de - de 0.001 alors on stop
	scale = max(abs(data["km"].max()), 1.0)
	km = (data["km"].astype(float) / scale).tolist()
	price = data["price"].astype(float)

	theta0 = 0.0
	theta1 = 0.0

	old_cost = MSE(theta0, theta1, km, price)
	while (True):
		theta0, theta1 = get_new_thetas(theta0, theta1, learning_rate, km, price)
		new_cost = MSE(theta0, theta1, km, price)
		if not (math.isfinite(old_cost) and math.isfinite(new_cost)):
			print("Numerical issue detected during training; try a smaller learning rate or scale your data.")
			break
		if abs(old_cost - new_cost) < epsilon:
			break
		old_cost = new_cost

	theta1 = theta1 / scale

	return (theta0, theta1)


def save_thetas(t0: float, t1: float):
	content = f"{t0},{t1}"
	try:
		f = open("training_results.txt", "w")
		f.write(content)
		f.close()
	except (PermissionError):
		print("Not able to write in the file training_results.txt. Please update permissions.")
		sys.exit(-1)


def do_graph(data: pd.DataFrame, theta0: float, theta1: float):
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
	save_thetas(theta0, theta1)
	do_graph(dataset, theta0, theta1)


if __name__ == "__main__":
	main()