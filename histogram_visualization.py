import matplotlib.pyplot as plt

marks = [40, 50, 60, 70, 70, 80, 90, 90, 100]

plt.hist(marks, bins=5)

plt.title("Marks Distribution")
plt.show()