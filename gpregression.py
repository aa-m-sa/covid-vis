#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# gaussian process regression on Roos' data

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic

# naively stratification(?), i.e. separate regressions per continent.

#gpr_eu = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(0.1), n_restarts_optimizer=9)
#gpr_am = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(0.1), n_restarts_optimizer=9)
#gpr_ap = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(0.1), n_restarts_optimizer=9)

gpr_eu = GaussianProcessRegressor(kernel=RationalQuadratic()+WhiteKernel(), n_restarts_optimizer=9)
gpr_am = GaussianProcessRegressor(kernel=RationalQuadratic()+WhiteKernel(), n_restarts_optimizer=9)
gpr_ap = GaussianProcessRegressor(kernel=RationalQuadratic()+WhiteKernel(), n_restarts_optimizer=9)

regXa = np.array(regX)
y = np.array(y)
regXeu = regXa[regXa[:,1]=="Europe",0].reshape(-1,1)
yeu = y[regXa[:,1]=="Europe"]
regXam = regXa[regXa[:,1]=="Americas",0].reshape(-1,1)
yam = y[regXa[:,1]=="Americas"]
regXap = regXa[regXa[:,1]=="Asia & Pacific",0].reshape(-1,1)
yap = y[regXa[:,1]=="Asia & Pacific"]

gpr_eu.fit(regXeu, yeu)
gpr_am.fit(regXam, yam)
gpr_ap.fit(regXap, yap)

xt2 = np.linspace(np.min(regXa[:,0]), np.max(regXa[:,0]+np.log(4)), 100).reshape(-1,1)
xt = np.linspace(np.min(regXa[:,0]), np.max(regXa[:,0]), 100).reshape(-1,1)


y_pred_eu, sigma_eu = gpr_eu.predict(xt, return_std=True)
y_pred_am, sigma_am = gpr_am.predict(xt, return_std=True)
y_pred_ap, sigma_ap = gpr_ap.predict(xt, return_std=True)

plt.figure()

#### repeat plotting setup

 
if region_system in ['ITU', 'ITUcomb']:
    width = 10
else:
    width = 12
f, ax = plt.subplots(figsize=(width, 7))
sns.set_style("dark")
sns.set(font_scale=1.15)
p = sns.scatterplot(x = x, y = y, ax=ax)

ax.set(xscale='log', yscale='linear')

# format axis labels

formatter = FuncFormatter(lambda x, _: '{:,.16g}m'.format(x)) # https://stackoverflow.com/a/49306588/3904031
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=(1.0, 0.5)))

formatter = FuncFormatter(lambda y, _: '{:,.16g}'.format(10**y)) # https://stackoverflow.com/a/49306588/3904031
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_major_locator(plt.FixedLocator([1,2]))
  
# create legend with a color sample for each region

texts = [plt.text(x[i], y[i], lab[i], ha='center', va='bottom', color=color[reg[i]]) for i in range(len(lab))]


ax.set_xlabel("Country population")
ax.set_ylabel("Daily deaths (7-day avg) {} days after 50th death".format(lag))

if region_system in ['ITU', 'ITUcomb']:
    leg = plt.legend(handles=[mpatches.Patch(color=colors[i], label=regions[i]) for i in range(len(regions))], facecolor='white', loc='lower right') #
else:
    box = ax.get_position() # get position of figure
    ax.set_position([box.x0, box.y0, box.width * 0.70, box.height]) # resize position
    leg = plt.legend(handles=[mpatches.Patch(color=colors[i], label=reglabels[i]) for i in range(len(regions))], facecolor='white', bbox_to_anchor=(1.05, 1), loc=2) #, borderaxespad=0.)

if region_system in ['UNtop', 'UN2nd']:
    leg.set_title('Region (UN)')
elif region_system == 'ITUcomb':
    leg.set_title('Region')
else:   
    leg.set_title("Region ({})".format(region_system))


# gp plotting

plt.plot(np.power(10, xt), y_pred_eu, linewidth=3.5, color=color[regions[0]], alpha=0.6)
plt.plot(np.power(10, xt), y_pred_am, linewidth=3.5, color=color[regions[1]], alpha=0.6)
plt.plot(np.power(10, xt), y_pred_ap, linewidth=3.5, color=color[regions[2]], alpha=0.6)

ax.annotate("Sources: Covid data from ourworldindata.org, population data from worldometers.info\n Code at github.com/aa-m-sa/covid-vis , a fork of github.com/teemuroos/covid-vis" ,
            xy=(10, 10), xycoords='figure pixels', color='gray', fontsize=10)

# add linear trend

plt.plot(np.power(10, xt), xt, linewidth=1, color="black", alpha=0.5, linestyle="dashed")

plt.savefig("figures/gpregression.png", dpi=300)


# posterior interval

plt.plot(np.power(10,xt),
         (y_pred_eu - 1.9600 * sigma_eu).reshape(-1,1),
         alpha=0.4, color=color[regions[0]], linewidth=2, linestyle="dashed")

plt.plot(np.power(10,xt),
         (y_pred_eu + 1.9600 * sigma_eu).reshape(-1,1),
         alpha=0.4, color=color[regions[0]], linewidth=2, linestyle="dashed")

plt.savefig("figures/gpregression_with_europe_CI.png", dpi=300)


# # posterior interval

# plt.plot(np.power(10,xt),
#          (y_pred_am - 1.9600 * sigma_am).reshape(-1,1),
#          alpha=0.4, color=color[regions[1]], linewidth=2, linestyle="dotted")

# plt.plot(np.power(10,xt),
#          (y_pred_am + 1.9600 * sigma_am).reshape(-1,1),
#          alpha=0.4, color=color[regions[1]], linewidth=2, linestyle="dotted")

# # posterior interval

# plt.plot(np.power(10,xt),
#          (y_pred_ap - 1.9600 * sigma_ap).reshape(-1,1),
#          alpha=0.4, color=color[regions[2]], linewidth=2, linestyle="dashdot")

# plt.plot(np.power(10,xt),
#          (y_pred_ap + 1.9600 * sigma_ap).reshape(-1,1),
#          alpha=0.4, color=color[regions[2]], linewidth=2, linestyle="dashdot")



# plt.savefig("figures/gpregression_withCI.png", dpi=300)


## No Russia in set of Europe region?

regXa_norus = regXa[np.array(lab)!="RUS",:]
y_norus = y[np.array(lab)!="RUS"]

#regXa_norus = regXa[np.bitwise_and(np.array(lab)!="RUS", np.array(lab)!="UKR"),:]
#y_norus = y[np.bitwise_and(np.array(lab)!="RUS", np.array(lab)!="UKR")]

regXeu_norus = regXa_norus[regXa_norus[:,1]=="Europe",0].reshape(-1,1)
yeu_norus = y_norus[regXa_norus[:,1]=="Europe"]

gpr_eu_norus = GaussianProcessRegressor(kernel=RationalQuadratic()+WhiteKernel(), n_restarts_optimizer=9)

gpr_eu_norus.fit(regXeu_norus, yeu_norus)
y_pred_eu_norus, sigma_eu_norus = gpr_eu_norus.predict(xt, return_std=True)

plt.plot(np.power(10, xt), y_pred_eu_norus, linewidth=2, color="black", alpha=0.3)

# posterior interval

plt.plot(np.power(10,xt),
         (y_pred_eu_norus - 1.9600 * sigma_eu_norus).reshape(-1,1),
         alpha=0.4, color="black", linewidth=2, linestyle="dashed")

plt.plot(np.power(10,xt),
         (y_pred_eu_norus + 1.9600 * sigma_eu_norus).reshape(-1,1),
         alpha=0.4, color="black", linewidth=2, linestyle="dashed")

plt.savefig("figures/gpregression_with_europe_CI_noRUS.png", dpi=300)
plt.show()

f, ax = plt.subplots(figsize=(width, 7))
sns.set_style("dark")
sns.set(font_scale=1.15)
p = sns.scatterplot(x = regXeu_norus[:,0], y = yeu_norus, ax=ax)

ax.set(xscale='linear', yscale='linear')

plt.plot(xt, y_pred_eu_norus, linewidth=3.5, color=color[regions[0]], alpha=0.6)
plt.savefig("figures/gpregression_with_europe_CI_noRUS2.png", dpi=300)
plt.show()