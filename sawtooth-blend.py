from pylab import *
from numpy import *
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

def f(x, s=0):
    return (2 * x + s * sin(2 * pi * x)) % 1

def dfdx(x, s=0):
    return 2 + s * 2 * pi * cos(2 * pi * x)

def dfds(x, s=0):
    return sin(2 * pi * x)

def J(x):
    return cos(2 * pi * x)

def dJdx(x):
    return -2 * pi * sin(2 * pi * x)

# ------------------- FD --------------------
s_arr = linspace(1E-9, 0.1, 21)
Js_arr = []
for s in s_arr:
    x = random.rand(100)
    for i in range(10):
        x = f(x, s)
    J_s, x_arr = [], []
    for i in range(1000):
        x = f(x, s)
        J_s.append(J(x).mean())
        x_arr.append(x.copy())
    Js_arr.append(mean(J_s))

Js_arr = array(Js_arr)
figure(0)
meanLegend = plot(s_arr, Js_arr, 'o')[0]

# ------------------- LSS -------------------
Js_lss_arr = []
Js_lssr_arr = []
for s in s_arr:
# for s in [0]:
    x = random.rand(10)
    for i in range(10):
        x = f(x, s)
    x_hist = []
    for i in range(1000):
        x = f(x, s)
        x_hist.append(x.copy())
    x_hist = array(x_hist)
    fx, fs = dfdx(x_hist, s), dfds(x_hist, s)

    v = []
    for i in range(x_hist.shape[1]):
        N = fx.shape[0]
        B = sp.spdiags(ones(N), 1, N-1, N) - sp.spdiags(fx[:,i], 0, N-1, N)
        S = B * B.T
        w = splinalg.spsolve(S, fs[:-1,i])
        v.append(B.T * w)
    v = transpose(v)
    Js_lss_arr.append((dJdx(x_hist) * v).mean())

    vr = []
    for i in range(x_hist.shape[1]):
        N = fx.shape[0]
        dx = x_hist[1:,i] - x_hist[1:,i:i+1]
        dx = abs((dx + 0.5) % 1 - 0.5)

        EPS = 3E-2
        n_reconnect = (dx < EPS).sum(0)
        i_reconnect, j_reconnect = (dx < EPS).nonzero()

        fx_i, fx_j = fx[i_reconnect,i], fx[j_reconnect,i]
        B = sp.lil_matrix((N-1, N))
        B[i_reconnect, j_reconnect] = -1.0 / n_reconnect[i_reconnect] * fx_j
        B += sp.spdiags(ones(N), 1, N-1, N)
        b = np.dot(dx < EPS, fs[:-1,i]) / n_reconnect

        B = B.tocsr()
        S = B * B.T
        w = splinalg.spsolve(S, b)
        vr.append(B.T * w)
    vr = transpose(vr)
    Js_lssr_arr.append((dJdx(x_hist) * vr).mean())

    figure(1)
    clf()
    vLegend = plot(x_hist, v, '.b')[0]
    vrLegend = plot(x_hist, vr, '.r')[0]
    xlabel('x')
    ylabel('v')
    legend([vLegend, vrLegend], ['lss', 'lssr'])
    grid()
    savefig('sawtooth_plus_s_sin2pix_s_{0:5.3f}.png'.format(s))
    # stop

Js_lss_arr = array(Js_lss_arr)
Js_lssr_arr = array(Js_lssr_arr)

ds_vis = 0.005
figure(0)
lssLegend = plot([s_arr - ds_vis, s_arr + ds_vis],
     [Js_arr - Js_lss_arr * ds_vis, Js_arr + Js_lss_arr * ds_vis], '-k')[0]
lssrLegend = plot([s_arr - ds_vis, s_arr + ds_vis],
     [Js_arr - Js_lssr_arr * ds_vis, Js_arr + Js_lssr_arr * ds_vis], '-r')[0]
xlabel('s')
ylabel(r'$E(\cos(2\pi x))$')
legend([meanLegend, lssLegend, lssrLegend],
       ['mean', 'lss', 'lssr'], loc='upper left')
grid()
savefig('sawtooth_plus_s_sin2pix')
show()
