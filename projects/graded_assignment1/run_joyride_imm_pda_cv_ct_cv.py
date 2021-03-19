# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data.data_fetchers import get_joyride_data
from filters import pda, imm, ekf
from utils import measurementmodels, dynamicmodels
from utils import estimationstatistics as estats
# %% plot config check and style setup


# to see your plot config
from utils.gaussparams import GaussParams
from utils.mixturedata import MixtureParameters

print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


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


# %% load data and plot
K, Ts, T, Xgt, Z = get_joyride_data()
Ts = np.insert(Ts, 0, 0)  # Add first measurement at timestep 0
# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45  # pre-gated measurements
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

# %% play measurement movie. Remember that you can cross out the window
play_movie = False
play_slice = slice(0, K)
if play_movie:
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    fig2, ax2 = plt.subplots(num=2, clear=True)
    sh = ax2.scatter(np.nan, np.nan)
    th = ax2.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = 0.1
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)

# %% setup and track

# ts 2.5?
# State-model and measurement-model parameters
sigma_z = 25  # From EKF exc 3
sigma_a_CV = 0.25  # From EKF exc 3
sigma_a_CT = 0.25  # From EKF exc 3
sigma_a_CV_high = 3.5  # From EKF exc 3
sigma_omega = 0.002

state_dim = 5  # CV + CT states
modes = 3

# IMM
p11 = 0.95  # CV
p22 = 0.90  # CT
p33 = 0.95  # CV High

               # CV CT CV High
PI = np.array([[p11, 0.5*(1-p11), 0.5*(1-p11)],   # CV
               [0.2*(1-p22), p22, 0.8*(1-p22)],   # CT
               [0.2*(1-p33), 0.8*(1-p33), p33]])  # CV High
# p_init = 0.8
# init_weights = np.array([p_init, 0.5*(1-p_init), 0.5*(1-p_init)])
init_weights = np.array([0.4, 0.2, 0.4])

# PDA
PD = 0.9
clutter_intensity = 10e-6
gate_size = 4

# IMM Init from initial position measurement / WHY NOT VELOCITY?
Z_0 = Z[0]
init_mean = np.array([Z_0[0, 0], Z_0[0, 1], 2, 0, 0])

# IMM Init Covariance
init_cov = np.diag([sigma_z, sigma_z, 3, 3, 0.0005]) ** 2  # THIS WILL NOT BE GOOD

# Models and filter instantiation
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=state_dim)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=state_dim)
CV_high = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV_high, n=state_dim)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)
ekf_filters = [ekf.EKF(CV, measurement_model), ekf.EKF(CT, measurement_model), ekf.EKF(CV_high, measurement_model)]

# Transition matrix test
assert np.allclose(PI.sum(axis=1), 1), "rows of PI must sum to 1"
assert PI.shape[0] == modes, 'Dimension transition matrix not same as modes'

# IMM filter instantiation
imm_filter = imm.IMM(ekf_filters, PI)

# IMM init mode probabilities test
assert np.allclose(np.sum(init_weights), 1), \
    "initial mode probabilities must sum to 1"

init_mode_states = [GaussParams(init_mean, init_cov)] * modes
init_immstate = MixtureParameters(init_weights, init_mode_states)  # Mixture of two mode states
tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

# allocate
NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

tracker_update = init_immstate
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []

# estimate
for k, (Zk, x_true_k, ts) in enumerate(zip(Z, Xgt, Ts)):
    tracker_predict = tracker.predict(tracker_update, Ts=ts)
    tracker_update = tracker.update(Zk, tracker_predict)

    # You can look at the prediction estimate as well
    tracker_estimate = tracker.estimate(tracker_update)

    NEES[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(4))
    NEESpos[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2))
    NEESvel[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2, 4))

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)

x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.weights for upd in tracker_update_list])

# calculate a performance metrics
poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1)
# todo calculate omega gt from velocities
# omega_err = np.linalg.norm(x_hat[:, 4] - Xgt[:, 4], axis=0
posRMSE = np.sqrt(np.mean(poserr ** 2))  # not true RMSE (which is over monte carlo simulations)
velRMSE = np.sqrt(np.mean(velerr ** 2))
# omegaRMSE = np.sqrt(np.mean(omega_err ** 2))

# not true RMSE (which is over monte carlo simulations)
peak_pos_deviation = poserr.max()
peak_vel_deviation = velerr.max()


# consistency - same conf interval for NEES and NIS
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K

ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)

# NIS -> Mean double measurements over rows
gated_bbol = [tracker.gate(z_k, pred_k) for z_k, pred_k in zip(Z,tracker_predict_list)]

Z_g = [z[gate] for z, gate in zip(Z, gated_bbol)]
# valid_predictions = pred_k[any(gated_bbol)]
Z_meaned = [zk[0] if zk.shape[0] is 1 else np.mean(zk, axis=0) for zk in Z]
NISes_comb = [tracker.state_filter.NISes(z_mean, pred_k) for
              z_mean, pred_k in zip(Z_meaned, tracker_predict_list)]
NIS, NISes = [np.array(n) for n in zip(*NISes_comb)]
ANIS = NIS.mean()

# %% plots
# trajectory
fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
axs3[0].plot(*x_hat.T[:2], label=r"$\hat x$")
axs3[0].plot(*Xgt.T[:2], label="$x$")
axs3[0].set_title(
    f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
)
axs3[0].axis("equal")
axs3[0].legend(loc='upper right')
time_cum = np.cumsum(Ts)
# probabilities
axs3[1].plot(time_cum, prob_hat[:, 0], label='CV')
axs3[1].plot(time_cum, prob_hat[:, 1], label='CT')
axs3[1].plot(time_cum, prob_hat[:, 2], label='CV High')
axs3[1].legend(loc='upper right')
axs3[1].set_ylim([0, 1])
axs3[1].set_ylabel("mode probability")
axs3[1].set_xlabel("time")

# NEES
fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
axs4[0].plot(time_cum, NEESpos)
axs4[0].plot([0, np.sum(Ts[:-1])], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[1].plot(time_cum, NEESvel)
axs4[1].plot([0, np.sum(Ts[:-1])], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[2].plot(time_cum, NEES)
axs4[2].plot([0, np.sum(Ts[:-1])], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES")
inCI = np.mean((CI4[0] <= NEES) * (NEES <= CI4[1]))
axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")


# errors
fig5, axs5 = plt.subplots(2, num=5, clear=True)
axs5[0].plot(time_cum, np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
axs5[0].set_ylabel("position error")

axs5[1].plot(time_cum, np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
axs5[1].set_ylabel("velocity error")

# NIS
fig6, axs6 = plt.subplots(2)
axs6[0].plot(time_cum, NIS, label='NIS')
axs6[0].legend()
axs6[0].plot([0, np.sum(Ts[:-1])], np.repeat(CI4[None], 2, 0), "--r")
axs6[0].set_ylabel("NIS")
inCI_nis = np.mean((CI4[0] <= NIS) * (NIS <= CI4[1]))
axs6[0].set_title(f"{inCI_nis*100:.1f}% inside {confprob*100:.1f}% CI")

axs6[1].plot(time_cum, NISes[:, 0], label='CV')
axs6[1].plot(time_cum, NISes[:, 1], label='CT')
axs6[1].plot(time_cum, NISes[:, 2], label='CV High')
axs6[1].legend()
axs6[1].plot([0, np.sum(Ts[:-1])], np.repeat(CI4[None], 2, 0), "--r")
axs6[1].set_ylabel("NISes")
# inCI = np.mean((CI4[0] <= NIS) * (NIS <= CI4[1]))
# axs6[0].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

fig7, axs7 = plt.subplots()
axs7.plot(time_cum, x_hat[:, 4], label='"$\omega$" est')
# axs7[1].plot(time_cum, Xgt[:, 4], label='r"$\omega$" GT')
axs7.legend()
axs7.set_ylabel("$\omega$")
# axs7[1].set_title(f"{omegaRMSE:.3f}")
axs7.set_title("Estimated turn rate")

print(f"ANIS = {ANIS:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")
print(f"RMSE pos = {posRMSE:.3f}")
print(f"RMSE vel = {velRMSE:.3f}")
plt.show()


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


# write key values to log file
log_file = 'imm_pda_cv_ct_cv.log'
writeValuesToFile = True
if writeValuesToFile:
    append_new_line(log_file, f"sigma_z = {sigma_z}")
    append_new_line(log_file, f"sigma_a_CT = {sigma_a_CT}")
    append_new_line(log_file, f"sigma_a_CV = {sigma_a_CV}")
    append_new_line(log_file, f"sigma_a_CV_high = {sigma_a_CV_high}")
    append_new_line(log_file, f"sigma_omega = {sigma_omega}")
    append_new_line(log_file, f"clutter_intensity = {clutter_intensity}")
    append_new_line(log_file, f"PD = {PD}")
    append_new_line(log_file, f"gate_size = {gate_size}")
    append_new_line(log_file, f"PI11 = {p11}. PI22 = {p22}. P133 = {p33}")
    append_new_line(log_file, f"init mode prob: CV: {init_weights[0]:.3f}. "
                              f"CT: {init_weights[1]:.3f}. CV High: {init_weights[2]:.3f}")
    append_new_line(log_file, f"ANEESpos = {ANEESpos:.2f} with CI2K = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    append_new_line(log_file, f"ANEESvel = {ANEESvel:.2f} with CI2K = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    append_new_line(log_file, f"ANEES = {ANEES:.2f} with CI4K = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")
    append_new_line(log_file, f"ANEES = {ANIS:.2f} with CI4K = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")
    append_new_line(log_file, f"NEES pos: {inCIpos*100:.1f}% inside {confprob*100:.1f}% CI2")
    append_new_line(log_file, f"NEES vel: {inCIvel*100:.1f}% inside {confprob*100:.1f}% CI2")
    append_new_line(log_file, f"NEES: {inCI*100:.1f}% inside {confprob*100:.1f}% CI4")
    append_new_line(log_file, f"NIS: {inCI_nis * 100:.1f}% inside {confprob * 100:.1f}% CI4")
    append_new_line(log_file, f"RMSE pos = {posRMSE:.3f}")
    append_new_line(log_file, f"RMSE vel = {velRMSE:.3f}")
    append_new_line(log_file, "\n")


