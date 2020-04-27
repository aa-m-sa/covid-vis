# linear model for Europe only

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

## model

linreg_eu = LinearRegression()
model_eu = linreg_eu.fit(regXeu, yeu)

linreg_eu_norus = LinearRegression()
model_eu_norus = linreg_eu_norus.fit(regXeu_norus, yeu_norus)

linreg_am = LinearRegression()
model_am = linreg_am.fit(regXam, yam)

linreg_ap = LinearRegression()
model_ap = linreg_ap.fit(regXap, yap)

y_pred_eu = model_eu.predict(xt)
y_pred_eu_norus = model_eu_norus.predict(xt)
y_pred_am = model_am.predict(xt)
y_pred_ap = model_ap.predict(xt)


plt.plot(np.power(10, xt), y_pred_eu_norus, linewidth=2, color="black", alpha=0.3)
plt.plot(np.power(10, xt), y_pred_eu, linewidth=3.5, color=color[regions[0]], alpha=0.6)
plt.plot(np.power(10, xt), y_pred_am, linewidth=3.5, color=color[regions[1]], alpha=0.6)
plt.plot(np.power(10, xt), y_pred_ap, linewidth=3.5, color=color[regions[2]], alpha=0.6)

plt.plot(np.power(10, xt), xt, linewidth=1, color="black", alpha=0.5, linestyle="dashed")

plt.savefig("figures/gpregression.png", dpi=300)

plt.savefig("figures/lmregression_with_europe_noRUS.png", dpi=300)
plt.show()