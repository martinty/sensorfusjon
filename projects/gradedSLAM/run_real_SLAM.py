# %% Imports
from scipy.io import loadmat
from scipy.stats import chi2

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]


import numpy as np
from gradedSLAM.EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from gradedSLAM.plotting import ellipse
from gradedSLAM.vp_utils import detectTrees, odometry, Car
from gradedSLAM.utils import rotmat2d

# %% plot config check and style setup


# to see your plot config
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

# %% Load data
VICTORIA_PARK_PATH = "./victoria_park/"
realSLAM_ws = {
    **loadmat(VICTORIA_PARK_PATH + "aa3_dr"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_lsr2"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_gpsx"),
}

timeOdo = (realSLAM_ws["time"] / 1000).ravel()
timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

steering = realSLAM_ws["steering"].ravel()
speed = realSLAM_ws["speed"].ravel()
LASER = (
    realSLAM_ws["LASER"] / 100
)  # Divide by 100 to be compatible with Python implementation of detectTrees
La_m = realSLAM_ws["La_m"].ravel()
Lo_m = realSLAM_ws["Lo_m"].ravel()

K = timeOdo.size
mK = timeLsr.size
Kgps = timeGps.size

# %% Parameters

L = 2.83  # axel distance
H = 0.76  # center to wheel encoder
a = 0.95  # laser distance in front of first axel
b = 0.5  # laser distance to the left of center

car = Car(L, H, a, b)

# TODO: Q matrix
sigma_x = 0.5
sigma_y = 0.5
sigma_psi = 0.7 * np.pi/180

sigmas = np.array([sigma_x, sigma_y, sigma_psi])
CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)

# TODO: R matrix
sigma_r = 0.06
sigma_phi = 0.22 * np.pi/180

R = np.diag([sigma_r, sigma_phi]) ** 2

# TODO: JCBB alphas
alpha_joint = 5e-6
alpha_individual = 5e-3

JCBBalphas = np.array([alpha_joint, alpha_individual])

sensorOffset = np.array([car.a + car.L, car.b])
doAsso = True

slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas, sensor_offset=sensorOffset)

# For consistency testing
alpha = 0.05
confidence_prob = 1 - alpha

xupd = np.zeros((mK, 3))
a = [None] * mK
NIS = np.zeros(mK)
NISnorm = np.zeros(mK)
CI = np.zeros((mK, 2))
CInorm = np.zeros((mK, 2))

NISnorm_asso_only = np.zeros(mK)
CInorm_asso_only = np.zeros((mK, 2))

# Initialize state
eta = np.array([Lo_m[0], La_m[1], 42 * np.pi / 180]) # you might want to tweak these for a good reference
P = np.diag([1, 1, 1])  # np.zeros((3, 3))  # TODO: Check this!

mk_first = 1  # first seems to be a bit off in timing
mk = mk_first
t = timeOdo[0]

# %%  run
N = K  # TODO: set as K when done
total_num_asso = 0

doPlot = False

lh_pose = None

if doPlot:
    fig, ax = plt.subplots(num=1, clear=True)

    lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
    sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
    sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

do_raw_prediction = True
if do_raw_prediction:  # TODO: further processing such as plotting
    odos = np.zeros((K, 3))
    odox = np.zeros((K, 3))
    odox[0] = eta

    for k in range(min(N, K - 1)):
        odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
        odox[k + 1], _ = slam.predict(odox[k], P.copy(), odos[k + 1])

for k in tqdm(range(N)):
    if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
        # Force P to symmetric: there are issues with long runs (>10000 steps)
        # seem like the prediction might be introducing some minor asymetries,
        # so best to force P symetric before update (where chol etc. is used).
        # TODO: remove this for short debug runs in order to see if there are small errors
        P = (P + P.T) / 2
        dt = timeLsr[mk] - t
        if dt < 0:  # avoid assertions as they can be optimized avay?
            raise ValueError("negative time increment")

        t = timeLsr[mk]  # ? reset time to this laser time for next post predict
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = slam.predict(eta, P, z_odo=odo)  # TODO predict

        z = detectTrees(LASER[mk])
        eta, P, NIS[mk], a[mk] = slam.update(eta, P, z)  # TODO update

        num_asso = np.count_nonzero(a[mk] > -1)

        if num_asso > 0:
            CI[mk] = chi2.interval(confidence_prob, 2 * num_asso)
            NISnorm[mk] = NIS[mk] / (2 * num_asso)
            CInorm[mk] = CI[mk] / (2 * num_asso)
            # Only associated measurements
            NISnorm_asso_only[mk] = NIS[mk] / (2 * num_asso)
            CInorm_asso_only[mk] = CI[mk] / (2 * num_asso)
            total_num_asso += 2 * num_asso

        else:
            NISnorm[mk] = 1
            CInorm[mk].fill(1)

        xupd[mk] = eta[:3]

        if doPlot:
            sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
            if len(z) > 0:
                zinmap = (
                    rotmat2d(eta[2])
                    @ (
                        z[:, 0] * np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                        + slam.sensor_offset[:, None]
                    )
                    + eta[0:2, None]
                )
                sh_Z.set_offsets(zinmap.T)
            lh_pose.set_data(*xupd[mk_first:mk, :2].T)

            ax.set(
                xlim=[-200, 200],
                ylim=[-200, 200],
                title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
            )
            plt.draw()
            plt.pause(0.00001)

        mk += 1

    if k < K - 1:
        dt = timeOdo[k + 1] - t
        t = timeOdo[k + 1]
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = slam.predict(eta, P, odo)

# %% Plotting
fig_size_x = 7
fig_size_y = 7

# %% Consistency
# NIS
insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * (NISnorm[:mk] <= CInorm[:mk, 1])

fig3, ax3 = plt.subplots(num=3, clear=True, constrained_layout=True, figsize=[fig_size_x, fig_size_y - 2])
ax3.plot(CInorm[:mk, 0], "--")
ax3.plot(CInorm[:mk, 1], "--")
ax3.plot(NISnorm[:mk], lw=0.5)

ax3.set_title(f"NIS: {insideCI.mean()*100:.2f}% inside {100 * confidence_prob:.0f}% CI")

# ANIS
CI_ANIS = np.array(chi2.interval(confidence_prob, total_num_asso))/total_num_asso
ANIS = np.sum(NIS)/total_num_asso
print(f"ANIS = {ANIS:.2f} with CI = [{CI_ANIS[0]:.2f}, {CI_ANIS[1]:.2f}]")

# %% NIS associated only
none_idxs = np.isnan(NISnorm_asso_only[:mk])
asso_indxs = np.where(none_idxs == False)[0]  # All indexes for associated measurements
insideCI_associated = (CInorm_asso_only[asso_indxs, 0] <= NISnorm_asso_only[asso_indxs]) * \
                      (NISnorm_asso_only[asso_indxs] <= CInorm_asso_only[asso_indxs, 1])

fig8, ax8 = plt.subplots(num=8, clear=True, constrained_layout=False, figsize=[fig_size_x, fig_size_y - 2])
ax8.plot(CInorm_asso_only[asso_indxs, 0], '--')
ax8.plot(CInorm_asso_only[asso_indxs, 1], '--')
ax8.plot(NISnorm_asso_only[asso_indxs], lw=0.5)

ax8.set_title(f'NIS associated only: {insideCI_associated.mean()*100:.1f}% inside {100 * confidence_prob:.0f}%  CI')
ANIS_asso_only = np.mean(NIS[asso_indxs])  # The mean of NIS over its associated measurements
CI_ANIS_asso_only = np.array(chi2.interval(confidence_prob, total_num_asso)) / len(asso_indxs)
fig8.tight_layout()
print(f"ANIS associated only = {ANIS_asso_only:.2f} with CI = [{CI_ANIS_asso_only[0]:.2f}, {CI_ANIS_asso_only[1]:.2f}]")

# %% slam

if do_raw_prediction:
    fig5, ax5 = plt.subplots(num=5, clear=True, constrained_layout=True, figsize=[fig_size_x, fig_size_y])
    ax5.scatter(
        Lo_m[timeGps < timeOdo[N - 1]],
        La_m[timeGps < timeOdo[N - 1]],
        c="r",
        marker=".",
        label="GPS",
    )
    ax5.plot(*odox[:N, :2].T, label="odom")
    ax5.grid()
    ax5.set_title("GPS vs odometry integration")
    ax5.legend()

# %%
fig6, ax6 = plt.subplots(num=6, clear=True, constrained_layout=True, figsize=[fig_size_x, fig_size_y])
ax6.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x", label="landmark")
ax6.plot(*xupd[mk_first:mk, :2].T, label="est")
ax6.set(
    title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
)
ax6.legend()

# %% Trajectory vs GPS
fig7, ax7 = plt.subplots(num=7, clear=True, constrained_layout=True, figsize=[fig_size_x, fig_size_y])
ax7.plot(*xupd[mk_first:mk, :2].T, label="est")
ax7.scatter(
        Lo_m[timeGps < timeOdo[N - 1]],
        La_m[timeGps < timeOdo[N - 1]],
        c="r",
        marker=".",
        label="GPS",
    )
ax7.grid()
ax7.set_title("Estimated trajectory vs GPS")
ax7.legend()

# %%
plt.show()

# %%
saveFigures = False
if saveFigures:
    fig3.savefig("plots/vp_NIS.png")
    fig6.savefig("plots/vp_result.png")
    if do_raw_prediction:
        fig5.savefig("plots/vp_raw_prediction.png")
    fig7.savefig("plots/vp_est_GPS.png")
    fig8.savefig("plots/vp_NIS_asso_only.png")

# %%
