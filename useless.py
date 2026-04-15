import matplotlib.pyplot as plt

pomcp_times = [
    15.48,5.12,12.18,5.56,17.73,10.73,4.84,16.25,8.04,26.96,
    16.91,4.53,8.34,14.81,26.20,8.37,6.50,6.61,27.67,4.80,
    4.97,8.32,5.33,9.69,21.80,17.95,8.46,6.61,8.84,7.82,
    10.01,6.82,28.56,3.76,5.54,9.31,12.68,7.60,9.76
]

hybrid_times = [
    18.71,6.39,3.76,5.25,11.86,12.25,3.72,9.23,4.77,7.23,
    3.56,6.66,7.58,8.00,6.31,12.61,10.91,3.56,29.90,3.60,
    9.19,6.43,11.51,7.59,7.16,6.49,3.68,19.73,18.14,15.54,
    6.47,17.52,8.72,7.33,3.72,4.86,6.12,16.77,8.17,6.53,
    7.63,13.30,6.35,4.64,4.04,6.35,3.68
]

avg_pomcp = sum(pomcp_times) / len(pomcp_times)
avg_hybrid = sum(hybrid_times) / len(hybrid_times)

plt.figure()
plt.plot(pomcp_times, label="Pure POMCP", color="tab:blue")
plt.plot(hybrid_times, label="Hybrid Planner", color="tab:orange")

plt.axhline(avg_pomcp, linestyle="--", color="tab:blue", label="POMCP Avg")
plt.axhline(avg_hybrid, linestyle="--", color="tab:orange", label="Hybrid Avg")

plt.xlabel("Episode Index")
plt.ylabel("Time to Intercept (s)")
plt.title("Time to Intercept Comparison")
plt.legend()
plt.show()
