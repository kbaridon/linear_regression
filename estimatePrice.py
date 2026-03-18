import sys


def get_thetas():
	"""Get thetas from the previous training"""
	try:
		f = open("training_results.txt")
		line = f.readline().strip()
		parts = line.split(",")
		theta0 = float(parts[0])
		theta1 = float(parts[1])
		f.close()
		return (theta0, theta1)
	except:
		print("Please, begin by launching the training...")
		return (0, 0)


def main():
	"""Take a mileage a return an estimated price of the car."""
	try:
		if (len(sys.argv) != 2 | int(sys.argv[1]) < 0):
			raise ValueError
	except:
		print("Please respect the format: ./estimatePrice mileage (mileage is an int >0)")
		sys.exit(-1)
	mileage = int(sys.argv[1])
	if (mileage > 396691):
		print("Do not even try to sell your care. It's value is zero.")
		sys.exit(-1)
	theta0, theta1 = get_thetas()
	print(f"Predicting the price of {mileage}...")
	estimatePrice = theta0 + (theta1 * mileage)
	print(f"The price of your car should be at {estimatePrice:,.2f} euros.".replace(",", " "))


if __name__ == "__main__":
	main()