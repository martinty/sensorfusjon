#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats

from utils import dynamicmodels, measurementmodels
from filters import pda, ekf

# %% plot config check and style setup

# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )

# %%
use_pregen = True
data_file_name = "../data/data_for_pda.mat"
if use_pregen:
    loaded_data = scipy.io.loadmat(data_file_name)
    K = loaded_data["K"].item()
    Ts = loaded_data["Ts"].item()
    Xgt = loaded_data["Xgt"].T
    Z = [zk.T for zk in loaded_data["Z"].ravel()]
    true_association = loaded_data["a"].ravel()
else:
    x0 = np.array([0, 0, 1, 1, 0])
    P0 = np.diag([50, 50, 10, 10, np.pi / 4]) ** 2
    # model parameters
    sigma_a_true = 0.25
    sigma_omega_true = np.pi / 15
    sigma_z = 3
    # sampling interval a length
    K = 1000
    Ts = 0.1
    # detection and false alarm
    PDtrue = 0.9
    lambdatrue = 3e-4
    np.rando.rng(10)
    # [Xgt, Z, a] = sampleCTtrack(K, Ts, x0, P0, qtrue, rtrue,PDtrue, lambdatrue);
    raise NotImplementedError
# %%

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
# %%
# play measurement movie.
# Can you find the target?
# I do not think you can run this with inline plotting. '%matplotlib' in the console to make it external
# Remember that you can exit the figure.
# comment this out when you are
"""
fig2, ax2 = plt.subplots(num=2, clear=True)
sh = ax2.scatter(np.nan, np.nan)
th = ax2.set_title(f"measurements at step 0")
ax2.axis([0, 700, -100, 300])
plotpause = 0.003
# sets a pause in between time steps if it goes to fast
for k, Zk in enumerate(Z):
    sh.set_offsets(Zk)
    th.set_text(f"measurements at step {k}")
    fig2.canvas.draw_idle()
    plt.show(block=False)
    plt.pause(plotpause)
# %%
"""

# Model parameters EKF
sigma_a = 6  # From EKF exc 3
sigma_z = 4  # From EKF exc 3

# PDA relevant
PD = 0.6
clutter_intensity = 10e-10  # TODO # Basically estimated from poisson clutter model, makes no sense to have fixed one
gate_size = 2

CV = dynamicmodels.WhitenoiseAccelleration(sigma_a)
measurement_model = measurementmodels.CartesianPosition(sigma_z)
ekf_filter = ekf.EKF(CV, measurement_model)

tracker = pda.PDA(ekf_filter, clutter_intensity, PD, gate_size)

# allocate
NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

# initialize
x_bar_init = np.array([*Z[0][true_association[0] - 1], 0, 0])

P_bar_init = np.zeros((4, 4))
P_bar_init[[0, 1], [0, 1]] = 2 * sigma_z ** 2
P_bar_init[[2, 3], [2, 3]] = 10 ** 2

init_state = tracker.init_filter_state({"mean": x_bar_init, "cov": P_bar_init})

tracker_update = init_state
tracker_update_list = []
tracker_predict_list = []
# estimate
for k, (Zk, x_true_k) in enumerate(zip(Z, Xgt)):
    tracker_predict = tracker.predict(tracker_update, Ts=Ts)
    tracker_update = tracker.update(Zk, tracker_predict)
    x_est, P_est = tracker_update.mean, tracker_predict.cov
    NEES[k] = tracker.state_filter.NEES_from_gt(x_est, x_true_k[:4], P_est)  #  mahalanobis_distance_squared(x_true_k, x_bar, P_bar)
    NEESpos[k] = tracker.state_filter.NEES_from_gt(x_est[:2], x_true_k[:2], P_est[:2, :2])
    NEESvel[k] = tracker.state_filter.NEES_from_gt(x_est[2:4], x_true_k[2:4], P_est[2:, 2:])

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)

x_hat = np.array([upd.mean for upd in tracker_update_list])
# calculate a performance metric
pos_bar_squared = (x_hat[:,0]-Xgt[:,0])**2 + (x_hat[:,1]-Xgt[:,1])**2
vel_bar_squared = (x_hat[:,2]-Xgt[:,2])**2 + (x_hat[:,3]-Xgt[:,3])**2

posRMSE = np.sqrt(np.mean(pos_bar_squared, axis=0, dtype=np.float64))
velRMSE = np.sqrt(np.mean(vel_bar_squared, axis=0, dtype=np.float64))

# %% plots
fig3, ax3 = plt.subplots(num=3, clear=True)
ax3.plot(*x_hat.T[:2], label=r"$\hat x$")
ax3.plot(*Xgt.T[:2], label="$x$")
ax3.set_title(
    rf"$\sigma_a = {sigma_a:.3f}$, \sigma_z = {sigma_z:.3f}, posRMSE = {posRMSE:.2f}, velRMSE = {velRMSE:.2f}"
)

fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)

confprob = 0.95
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

axs4[0].plot(np.arange(K) * Ts, NEESpos)
axs4[0].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[1].plot(np.arange(K) * Ts, NEESvel)
axs4[1].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[2].plot(np.arange(K) * Ts, NEESpos)
axs4[2].plot([0, (K - 1) * Ts], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES")
inCI = np.mean((CI2[0] <= NEES) * (NEES <= CI2[1]))
axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

confprob = 0.95
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos, axis=0, dtype=np.float64)
ANEESvel = np.mean(NEESvel, axis=0, dtype=np.float64)
ANEES = np.mean(NEES, axis=0, dtype=np.float64)

print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

fig5, axs5 = plt.subplots(2, num=5, clear=True)
axs5[0].plot(np.arange(K) * Ts, np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
axs5[0].set_ylabel("position error")

axs5[1].plot(np.arange(K) * Ts, np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
axs5[1].set_ylabel("velocity error")
# %%
plt.show()