import matplotlib.pyplot as plt

months = [1, 2, 3, 4, 5]
sales = [100, 150, 130, 200, 250]

plt.bar(months, sales)

plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.show()