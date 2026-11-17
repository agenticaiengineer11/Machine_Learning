import matplotlib.pyplot as plt

hours = [1, 2, 3, 4, 5]
marks = [20, 40, 60, 80, 100]

plt.scatter(hours, marks)

plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.show()