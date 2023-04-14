from basiccreation import *
from plot_functions import *


opinions = a_random_initial_opinion_distribution(num_agents=10)

grid = np.array([[x, y] for x in np.linspace(0, 1, 5) for y in np.linspace(-1, 1, 5)  # 11 and 21
				if (((y-x) < 0.000001) and ((y+x) > -0.000001))]).round(decimals=3)

_ = create_many_opinions(num_agents=100, file_name='example_opinions', grid=grid, show_result=False)

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(111)
plot_histogram(ax1, opinions)
plt.gcf().canvas.draw()
plt.show()

