<html>
<head>
<title>Q1.py</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style type="text/css">
.s0 { color: #808080;}
.s1 { color: #a9b7c6;}
.s2 { color: #cc7832;}
.s3 { color: #6897bb;}
.s4 { color: #6a8759;}
</style>
</head>
<body bgcolor="#2b2b2b">
<table CELLSPACING=0 CELLPADDING=5 COLS=1 WIDTH="100%" BGCOLOR="#606060" >
<tr><td><center>
<font face="Arial, Helvetica" color="#000000">
Q1.py</font>
</center></td></tr></table>
<pre><span class="s0"># Required Libraries for Program</span>
<span class="s0"># Suggested running this program on PyCharm IDE</span>
<span class="s2">import </span><span class="s1">matplotlib.pyplot </span><span class="s2">as </span><span class="s1">plt</span>
<span class="s2">from </span><span class="s1">mlxtend.regressor </span><span class="s2">import </span><span class="s1">LinearRegression</span>
<span class="s2">from </span><span class="s1">numpy </span><span class="s2">import </span><span class="s1">random</span>
<span class="s2">import </span><span class="s1">numpy </span><span class="s2">as </span><span class="s1">np</span>
<span class="s2">from </span><span class="s1">sklearn </span><span class="s2">import </span><span class="s1">metrics</span>

<span class="s0"># Q1- Generate random data for vector X and Vector ESP and using those Vector we generate Vector Y.</span>
<span class="s0"># Given linear equation for Y = -1 + (0.5 * X) - (2 * X)^2 + (0.3 * X)^3</span>
<span class="s1">x = random.normal(loc=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">scale=</span><span class="s3">1</span><span class="s2">, </span><span class="s1">size=</span><span class="s3">5000</span><span class="s1">)</span>
<span class="s1">esp = random.normal(loc=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">scale=</span><span class="s3">0.25</span><span class="s2">, </span><span class="s1">size=</span><span class="s3">5000</span><span class="s1">)</span>
<span class="s1">y = -</span><span class="s3">1 </span><span class="s1">+ (</span><span class="s3">0.5 </span><span class="s1">* x) - (</span><span class="s3">2 </span><span class="s1">* (x ** </span><span class="s3">2</span><span class="s1">)) + (</span><span class="s3">0.3 </span><span class="s1">* (x ** </span><span class="s3">3</span><span class="s1">)) + esp</span>
<span class="s0"># true weights</span>
<span class="s1">true_weight_vector = [-</span><span class="s3">1</span><span class="s2">, </span><span class="s3">0.5</span><span class="s2">, </span><span class="s1">-</span><span class="s3">2</span><span class="s2">, </span><span class="s3">0.3</span><span class="s1">]</span>
<span class="s0"># ---------------------- END OF DATA GENERATION ------------------------</span>

<span class="s0"># Plotting graph using synthetic data of Vector X and Vector Y</span>
<span class="s1">plt.scatter(x</span><span class="s2">, </span><span class="s1">y</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;stars&quot;</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">&quot;green&quot;</span><span class="s2">, </span><span class="s1">marker=</span><span class="s4">&quot;.&quot;</span><span class="s2">, </span><span class="s1">s=</span><span class="s3">30</span><span class="s1">)</span>
<span class="s1">plt.title(</span><span class="s4">&quot;Synthetic Dataset of Vectors&quot;</span><span class="s1">)</span>
<span class="s1">plt.xlabel(</span><span class="s4">&quot;X&quot;</span><span class="s1">)</span>
<span class="s1">plt.ylabel(</span><span class="s4">&quot;y&quot;</span><span class="s2">, </span><span class="s1">rotation=</span><span class="s3">0</span><span class="s1">)</span>
<span class="s1">plt.show()</span>
<span class="s0"># --------------- END OF SYNTHETIC DATA PLOT --------------------------</span>


<span class="s0"># Convert Vector X into matrix for fitting/training model</span>
<span class="s1">X = np.asanyarray(x).reshape(-</span><span class="s3">1</span><span class="s2">, </span><span class="s3">1</span><span class="s1">)</span>

<span class="s0"># Using LinearRegression model for implementing BATCH GRADIANT DECENT in ADALINE NEURAL NETWORK.</span>
<span class="s0"># method is sgd - stochastic gradient descent with Minibatch = 1 act as Batch Gradiant Decent</span>
<span class="s0"># ets = Learning Rate | epochs = Dataset read cycles</span>
<span class="s1">adaline_BGD1 = LinearRegression(method=</span><span class="s4">'sgd'</span><span class="s2">, </span><span class="s1">eta=</span><span class="s3">0.0001</span><span class="s2">, </span><span class="s1">epochs=</span><span class="s3">20</span><span class="s2">, </span><span class="s1">random_seed=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">minibatches=</span><span class="s3">1</span><span class="s1">)</span>
<span class="s1">adaline_BGD2 = LinearRegression(method=</span><span class="s4">'sgd'</span><span class="s2">, </span><span class="s1">eta=</span><span class="s3">0.01</span><span class="s2">, </span><span class="s1">epochs=</span><span class="s3">20</span><span class="s2">, </span><span class="s1">random_seed=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">minibatches=</span><span class="s3">1</span><span class="s1">)</span>
<span class="s0"># Training Model</span>
<span class="s1">adaline_BGD1.fit(X</span><span class="s2">, </span><span class="s1">y)</span>
<span class="s1">adaline_BGD2.fit(X</span><span class="s2">, </span><span class="s1">y)</span>
<span class="s0"># Making predictions</span>
<span class="s1">prediction1 = adaline_BGD1.predict(X)</span>
<span class="s1">prediction2 = adaline_BGD2.predict(X)</span>
<span class="s0"># calculating Mean Square Error</span>
<span class="s1">mean_sq_error = metrics.mean_squared_error(prediction1</span><span class="s2">, </span><span class="s1">y)</span>
<span class="s1">mean_sq_error2 = metrics.mean_squared_error(prediction2</span><span class="s2">, </span><span class="s1">y)</span>

<span class="s1">eta1 = </span><span class="s3">0.0001</span>
<span class="s1">eta2 = </span><span class="s3">0.01</span>

<span class="s0"># Printing numeric output</span>
<span class="s1">print(</span><span class="s4">&quot;Adaline Batch Gradient Decent&quot;</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">&quot;-----------------------------------------------------&quot;</span><span class="s1">)</span>

<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\t</span><span class="s4">Learning Rate: &quot;</span><span class="s2">, </span><span class="s1">eta1</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Intercept: %.2f' </span><span class="s1">% adaline_BGD1.w_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Slope: %.2f' </span><span class="s1">% adaline_BGD1.b_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">MSE: '</span><span class="s2">, </span><span class="s1">mean_sq_error</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>

<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\n\t</span><span class="s4">Learning Rate: &quot;</span><span class="s2">, </span><span class="s1">eta2</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Intercept: %.2f' </span><span class="s1">% adaline_BGD2.w_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Slope: %.2f' </span><span class="s1">% adaline_BGD2.b_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">MSE: '</span><span class="s2">, </span><span class="s1">mean_sq_error2</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>

<span class="s1">xp1 = np.linspace(x.min()</span><span class="s2">, </span><span class="s1">x.max()</span><span class="s2">, </span><span class="s3">5000</span><span class="s1">)</span>
<span class="s1">xp2 = np.linspace(x.min()</span><span class="s2">, </span><span class="s1">x.max()</span><span class="s2">, </span><span class="s3">5000</span><span class="s1">)</span>

<span class="s0"># Plotting graph and for both Learning Rate time for Adaline Neural network Structure.</span>
<span class="s0"># With Batch Gradiant Decent method</span>

<span class="s0"># Plotting Regression line on the graph side by side</span>
<span class="s1">fig</span><span class="s2">, </span><span class="s1">ax = plt.subplots(nrows=</span><span class="s3">1</span><span class="s2">, </span><span class="s1">ncols=</span><span class="s3">2</span><span class="s2">, </span><span class="s1">figsize=(</span><span class="s3">8</span><span class="s2">, </span><span class="s3">4</span><span class="s1">))</span>
<span class="s0"># Graph number 1</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].scatter(X</span><span class="s2">, </span><span class="s1">y</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'green'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">'Data Point'</span><span class="s2">, </span><span class="s1">marker=</span><span class="s4">&quot;.&quot;</span><span class="s2">, </span><span class="s1">s=</span><span class="s3">30</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].plot(xp1</span><span class="s2">, </span><span class="s1">adaline_BGD1.predict(xp1.reshape(-</span><span class="s3">1</span><span class="s2">, </span><span class="s3">1</span><span class="s1">))</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'blue'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;Regression Line&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_title(</span><span class="s4">&quot;Adaline BGD (LR:&quot; </span><span class="s1">+ str(eta1) + </span><span class="s4">&quot;)&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_xlabel(</span><span class="s4">&quot;X&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_ylabel(</span><span class="s4">&quot;y&quot;</span><span class="s2">, </span><span class="s1">rotation=</span><span class="s3">0</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].legend()</span>
<span class="s0"># Graph number 2</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].scatter(X</span><span class="s2">, </span><span class="s1">y</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'green'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;Data Point&quot;</span><span class="s2">, </span><span class="s1">marker=</span><span class="s4">&quot;.&quot;</span><span class="s2">, </span><span class="s1">s=</span><span class="s3">30</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].plot(xp2</span><span class="s2">, </span><span class="s1">adaline_BGD2.predict(xp2.reshape(-</span><span class="s3">1</span><span class="s2">, </span><span class="s3">1</span><span class="s1">))</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'blue'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;Regression Line&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_title(</span><span class="s4">&quot;Adaline BGD (LR:&quot; </span><span class="s1">+ str(eta2) + </span><span class="s4">&quot;)&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_xlabel(</span><span class="s4">&quot;X&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_ylabel(</span><span class="s4">&quot;y&quot;</span><span class="s2">, </span><span class="s1">rotation=</span><span class="s3">0</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].legend()</span>
<span class="s1">plt.tight_layout()</span>
<span class="s1">plt.show()</span>
<span class="s0"># ----------- Plotting Regression line on the graph side by side END ---------------------</span>

<span class="s0"># Plotting Learning rate graphs side by side</span>
<span class="s1">fig</span><span class="s2">, </span><span class="s1">ax1 = plt.subplots(nrows=</span><span class="s3">1</span><span class="s2">, </span><span class="s1">ncols=</span><span class="s3">2</span><span class="s2">, </span><span class="s1">figsize=(</span><span class="s3">8</span><span class="s2">, </span><span class="s3">4</span><span class="s1">))</span>
<span class="s0"># Graph number 1</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].plot(range(</span><span class="s3">1</span><span class="s2">, </span><span class="s1">adaline_BGD1.epochs + </span><span class="s3">1</span><span class="s1">)</span><span class="s2">, </span><span class="s1">np.log10(adaline_BGD1.cost_))</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].set_xlabel(</span><span class="s4">'Epochs'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].set_ylabel(</span><span class="s4">'Cost: Log (Mean Squared Error)'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].set_title(</span><span class="s4">'Adaline BGD Learning curve' </span><span class="s1">+ str(eta1))</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].plot(range(</span><span class="s3">1</span><span class="s2">, </span><span class="s1">adaline_BGD2.epochs + </span><span class="s3">1</span><span class="s1">)</span><span class="s2">, </span><span class="s1">np.log10(adaline_BGD2.cost_))</span>
<span class="s0"># Graph number 2</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].set_xlabel(</span><span class="s4">'Epochs'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].set_ylabel(</span><span class="s4">'Cost: Log (Mean Squared Error)'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].set_title(</span><span class="s4">'Adaline BGD Learning curve' </span><span class="s1">+ str(eta2))</span>
<span class="s1">plt.tight_layout()</span>
<span class="s1">plt.show()</span>
<span class="s0"># ---------- Plotting Learning rate graphs side by side END ---------------------</span>
<span class="s0"># ----------- BATCH GRADIANT DECENT END _______________________________</span>

<span class="s0"># STOCHASTIC GRADIANT DECENT in ADALINE NEURAL NETWORK.</span>
<span class="s0"># Using LinearRegression model</span>
<span class="s0"># method is sgd - stochastic gradient descent with Minibatch = 1 act as Batch Gradiant Decent</span>
<span class="s0"># ets = Learning Rate | epochs = Dataset read cycles</span>
<span class="s1">adaline_SGD1 = LinearRegression(method=</span><span class="s4">'sgd'</span><span class="s2">, </span><span class="s1">eta=</span><span class="s3">0.0001</span><span class="s2">, </span><span class="s1">epochs=</span><span class="s3">20</span><span class="s2">, </span><span class="s1">random_seed=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">minibatches=len(y))</span>
<span class="s1">adaline_SGD2 = LinearRegression(method=</span><span class="s4">'sgd'</span><span class="s2">, </span><span class="s1">eta=</span><span class="s3">0.01</span><span class="s2">, </span><span class="s1">epochs=</span><span class="s3">20</span><span class="s2">, </span><span class="s1">random_seed=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">minibatches=len(y))</span>
<span class="s0"># Training Model</span>
<span class="s1">adaline_SGD1.fit(X</span><span class="s2">, </span><span class="s1">y)</span>
<span class="s1">adaline_SGD2.fit(X</span><span class="s2">, </span><span class="s1">y)</span>
<span class="s0"># Making predictions</span>
<span class="s1">prediction1 = adaline_SGD1.predict(X)</span>
<span class="s1">prediction2 = adaline_SGD2.predict(X)</span>
<span class="s0"># calculating Mean Square Error</span>
<span class="s1">mean_sq_error = metrics.mean_squared_error(prediction1</span><span class="s2">, </span><span class="s1">y)</span>
<span class="s1">mean_sq_error2 = metrics.mean_squared_error(prediction2</span><span class="s2">, </span><span class="s1">y)</span>

<span class="s1">eta1 = </span><span class="s3">0.0001</span>
<span class="s1">eta2 = </span><span class="s3">0.01</span>

<span class="s0"># Printing numeric output</span>
<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\n</span><span class="s4">Adaline Stochastic Gradient Decent&quot;</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">&quot;-----------------------------------------------------&quot;</span><span class="s1">)</span>

<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\t</span><span class="s4">Learning Rate: &quot;</span><span class="s2">, </span><span class="s1">eta1</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Intercept: %.2f' </span><span class="s1">% adaline_SGD1.w_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Slope: %.2f' </span><span class="s1">% adaline_SGD1.b_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">MSE: '</span><span class="s2">, </span><span class="s1">mean_sq_error</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>

<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\n\t</span><span class="s4">Learning Rate: &quot;</span><span class="s2">, </span><span class="s1">eta2</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Intercept: %.2f' </span><span class="s1">% adaline_SGD2.w_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">Slope: %.2f' </span><span class="s1">% adaline_SGD2.b_</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">'</span><span class="s2">\t</span><span class="s4">MSE: '</span><span class="s2">, </span><span class="s1">mean_sq_error2</span><span class="s2">, </span><span class="s1">end=</span><span class="s4">'</span><span class="s2">\n</span><span class="s4">'</span><span class="s1">)</span>

<span class="s1">xp1 = np.linspace(x.min()</span><span class="s2">, </span><span class="s1">x.max()</span><span class="s2">, </span><span class="s3">5000</span><span class="s1">)</span>
<span class="s1">xp2 = np.linspace(x.min()</span><span class="s2">, </span><span class="s1">x.max()</span><span class="s2">, </span><span class="s3">5000</span><span class="s1">)</span>

<span class="s0"># Plotting graph and for both Learning Rate time for Adaline Neural network Structure.</span>
<span class="s0"># With Batch Gradiant Decent method</span>

<span class="s0"># Plotting Regression line on the graph side by side</span>
<span class="s1">fig</span><span class="s2">, </span><span class="s1">ax = plt.subplots(nrows=</span><span class="s3">1</span><span class="s2">, </span><span class="s1">ncols=</span><span class="s3">2</span><span class="s2">, </span><span class="s1">figsize=(</span><span class="s3">8</span><span class="s2">, </span><span class="s3">4</span><span class="s1">))</span>
<span class="s0"># Graph number 1</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].scatter(X</span><span class="s2">, </span><span class="s1">y</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'green'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">'Data Point'</span><span class="s2">, </span><span class="s1">marker=</span><span class="s4">&quot;.&quot;</span><span class="s2">, </span><span class="s1">s=</span><span class="s3">30</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].plot(xp1</span><span class="s2">, </span><span class="s1">adaline_SGD1.predict(xp1.reshape(-</span><span class="s3">1</span><span class="s2">, </span><span class="s3">1</span><span class="s1">))</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'red'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;Regression Line&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_title(</span><span class="s4">&quot;Adaline SGD (LR:&quot; </span><span class="s1">+ str(eta1) + </span><span class="s4">&quot;)&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_xlabel(</span><span class="s4">&quot;X&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_ylabel(</span><span class="s4">&quot;y&quot;</span><span class="s2">, </span><span class="s1">rotation=</span><span class="s3">0</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">0</span><span class="s1">].legend()</span>
<span class="s0"># Graph number 2</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].scatter(X</span><span class="s2">, </span><span class="s1">y</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'green'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;Data Point&quot;</span><span class="s2">, </span><span class="s1">marker=</span><span class="s4">&quot;.&quot;</span><span class="s2">, </span><span class="s1">s=</span><span class="s3">30</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].plot(xp2</span><span class="s2">, </span><span class="s1">adaline_SGD2.predict(xp2.reshape(-</span><span class="s3">1</span><span class="s2">, </span><span class="s3">1</span><span class="s1">))</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'red'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;Regression Line&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_title(</span><span class="s4">&quot;Adaline SGD (LR:&quot; </span><span class="s1">+ str(eta2) + </span><span class="s4">&quot;)&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_xlabel(</span><span class="s4">&quot;X&quot;</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_ylabel(</span><span class="s4">&quot;y&quot;</span><span class="s2">, </span><span class="s1">rotation=</span><span class="s3">0</span><span class="s1">)</span>
<span class="s1">ax[</span><span class="s3">1</span><span class="s1">].legend()</span>
<span class="s1">plt.tight_layout()</span>
<span class="s1">plt.show()</span>
<span class="s0"># ----------- Plotting Regression line on the graph side by side END ---------------------</span>

<span class="s0"># Plotting Learning rate graphs side by side</span>
<span class="s1">fig</span><span class="s2">, </span><span class="s1">ax1 = plt.subplots(nrows=</span><span class="s3">1</span><span class="s2">, </span><span class="s1">ncols=</span><span class="s3">2</span><span class="s2">, </span><span class="s1">figsize=(</span><span class="s3">8</span><span class="s2">, </span><span class="s3">4</span><span class="s1">))</span>
<span class="s0"># Graph number 1</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].plot(range(</span><span class="s3">1</span><span class="s2">, </span><span class="s1">adaline_SGD1.epochs + </span><span class="s3">1</span><span class="s1">)</span><span class="s2">, </span><span class="s1">np.log10(adaline_SGD1.cost_))</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].set_xlabel(</span><span class="s4">'Epochs'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].set_ylabel(</span><span class="s4">'Cost: Log (Mean Squared Error)'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">0</span><span class="s1">].set_title(</span><span class="s4">'Adaline SGD Learning curve' </span><span class="s1">+ str(eta1))</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].plot(range(</span><span class="s3">1</span><span class="s2">, </span><span class="s1">adaline_SGD2.epochs + </span><span class="s3">1</span><span class="s1">)</span><span class="s2">, </span><span class="s1">np.log10(adaline_SGD2.cost_))</span>
<span class="s0"># Graph number 2</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].set_xlabel(</span><span class="s4">'Epochs'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].set_ylabel(</span><span class="s4">'Cost: Log (Mean Squared Error)'</span><span class="s1">)</span>
<span class="s1">ax1[</span><span class="s3">1</span><span class="s1">].set_title(</span><span class="s4">'Adaline SGD Learning curve' </span><span class="s1">+ str(eta2))</span>
<span class="s1">plt.tight_layout()</span>
<span class="s1">plt.show()</span>
<span class="s0"># ---------- Plotting Learning rate graphs side by side END ---------------------</span>
<span class="s0"># ----------- STOCHASTIC BATCH GRADIANT DECENT END _______________________________</span>

<span class="s0"># CROSS VALIDATION</span>
<span class="s0"># importing extra Libraries here to avoid Conflict between import names</span>
<span class="s2">from </span><span class="s1">sklearn.linear_model </span><span class="s2">import </span><span class="s1">LinearRegression</span>
<span class="s2">from </span><span class="s1">sklearn.model_selection </span><span class="s2">import </span><span class="s1">train_test_split</span>
<span class="s2">from </span><span class="s1">sklearn.model_selection </span><span class="s2">import </span><span class="s1">KFold</span>
<span class="s2">from </span><span class="s1">sklearn.model_selection </span><span class="s2">import </span><span class="s1">cross_val_score</span>
<span class="s2">from </span><span class="s1">sklearn.preprocessing </span><span class="s2">import </span><span class="s1">PolynomialFeatures</span>

<span class="s0"># Splitting dataset into TRAINING( size = 70% ) AND TESTING DATASET( size = 30% )</span>
<span class="s1">x_train</span><span class="s2">, </span><span class="s1">x_test</span><span class="s2">, </span><span class="s1">y_train</span><span class="s2">, </span><span class="s1">y_test = train_test_split(X</span><span class="s2">, </span><span class="s1">y</span><span class="s2">, </span><span class="s1">test_size=</span><span class="s3">0.3</span><span class="s1">)</span>
<span class="s1">lm = LinearRegression()</span>

<span class="s0"># using 10-fold cross validation method on our modelS</span>
<span class="s1">cross_val = KFold(n_splits=</span><span class="s3">10</span><span class="s2">, </span><span class="s1">shuffle=</span><span class="s2">True</span><span class="s1">)</span>

<span class="s1">min_mse = </span><span class="s3">23432142134.2343</span>
<span class="s1">min_degree = </span><span class="s3">1</span>

<span class="s0"># loop fit and transform our split sets. also calculating, comparing and testing regression performances</span>
<span class="s2">for </span><span class="s1">itr </span><span class="s2">in </span><span class="s1">range(</span><span class="s3">1</span><span class="s2">, </span><span class="s3">11</span><span class="s1">):</span>
    <span class="s1">poly = PolynomialFeatures(degree=itr)</span>
    <span class="s1">coss_val_model = poly.fit_transform(x_train)</span>
    <span class="s1">poly.fit(coss_val_model</span><span class="s2">, </span><span class="s1">y_train)</span>
    <span class="s1">model = lm.fit(coss_val_model</span><span class="s2">, </span><span class="s1">y_train)</span>
    <span class="s1">scores = cross_val_score(model</span><span class="s2">, </span><span class="s1">coss_val_model</span><span class="s2">, </span><span class="s1">y_train</span><span class="s2">, </span><span class="s1">scoring=</span><span class="s4">&quot;neg_mean_squared_error&quot;</span><span class="s2">, </span><span class="s1">cv=cross_val</span><span class="s2">, </span><span class="s1">n_jobs=</span><span class="s3">1</span><span class="s1">)</span>
    <span class="s1">mean_sq_err = np.mean(np.abs(scores))</span>
    <span class="s1">print(</span><span class="s4">&quot;Degree: &quot; </span><span class="s1">+ str(itr) + </span><span class="s4">&quot;, </span><span class="s2">\n</span><span class="s4">Polynomial MSE: &quot; </span><span class="s1">+ str(mean_sq_err) + </span><span class="s4">&quot;, STD: &quot; </span><span class="s1">+ str(np.std(scores)))</span>
    <span class="s2">if </span><span class="s1">(min_mse &gt; mean_sq_err):</span>
        <span class="s1">min_mse = mean_sq_err</span>
        <span class="s1">min_degree = itr</span>

    <span class="s0"># converting our dataset into array using numpy &quot;asarray&quot; and then reshaping it into matrix</span>
    <span class="s1">x_train_array = np.asarray(x_train).reshape(-</span><span class="s3">1</span><span class="s1">)</span>
    <span class="s1">y_train_array = np.asarray(y_train).reshape(-</span><span class="s3">1</span><span class="s1">)</span>
    <span class="s1">x_test_array = np.asarray(x_test).reshape(-</span><span class="s3">1</span><span class="s1">)</span>
    <span class="s1">y_test_array = np.asarray(y_test).reshape(-</span><span class="s3">1</span><span class="s1">)</span>
    <span class="s1">weights = np.polyfit(x_train_array</span><span class="s2">, </span><span class="s1">y_train_array</span><span class="s2">, </span><span class="s1">itr)</span>

    <span class="s0"># generating model with the given weights</span>
    <span class="s1">model = np.poly1d(weights)</span>
    <span class="s1">new_train = np.linspace(x_train_array.min()</span><span class="s2">, </span><span class="s1">x_train_array.max())</span>
    <span class="s1">new_test = np.linspace(x_test_array.min()</span><span class="s2">, </span><span class="s1">x_test_array.max()</span><span class="s2">, </span><span class="s3">70</span><span class="s1">)</span>
    <span class="s1">predict_plot_train = model(new_train)</span>
    <span class="s1">predict_plot_test = model(new_test)</span>

    <span class="s0"># printing the weight vectors</span>
    <span class="s1">print(</span><span class="s4">&quot;Weights:&quot;</span><span class="s1">)</span>
    <span class="s2">for </span><span class="s1">j </span><span class="s2">in </span><span class="s1">range(</span><span class="s3">0</span><span class="s2">, </span><span class="s1">len(weights)):</span>
        <span class="s1">print(</span><span class="s4">&quot;w&quot; </span><span class="s1">+ str(j) + </span><span class="s4">&quot; = &quot; </span><span class="s1">+ str(weights[j]))</span>

    <span class="s0"># plotting graphs for the regeneration performance and degree</span>

    <span class="s1">fig</span><span class="s2">, </span><span class="s1">ax = plt.subplots(nrows=</span><span class="s3">1</span><span class="s2">, </span><span class="s1">ncols=</span><span class="s3">2</span><span class="s2">, </span><span class="s1">figsize=(</span><span class="s3">8</span><span class="s2">, </span><span class="s3">4</span><span class="s1">))</span>

    <span class="s1">ax[</span><span class="s3">0</span><span class="s1">].scatter(x_train</span><span class="s2">, </span><span class="s1">y_train</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'green'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">'Data Point'</span><span class="s2">, </span><span class="s1">marker=</span><span class="s4">'+'</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">0</span><span class="s1">].plot(new_train</span><span class="s2">, </span><span class="s1">predict_plot_train</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'blue'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">'Regression Line'</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_title(</span><span class="s4">&quot;Plot (Training Set), Degree=&quot; </span><span class="s1">+ str(itr))</span>
    <span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_xlabel(</span><span class="s4">&quot;X_train&quot;</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">0</span><span class="s1">].set_ylabel(</span><span class="s4">&quot;y_train&quot;</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">0</span><span class="s1">].legend()</span>

    <span class="s1">ax[</span><span class="s3">1</span><span class="s1">].scatter(x_test</span><span class="s2">, </span><span class="s1">y_test</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'green'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">'Data Point'</span><span class="s2">, </span><span class="s1">marker=</span><span class="s4">'+'</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">1</span><span class="s1">].plot(new_test</span><span class="s2">, </span><span class="s1">predict_plot_test</span><span class="s2">, </span><span class="s1">color=</span><span class="s4">'blue'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">'Regression Line'</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_title(</span><span class="s4">&quot;Plot (Testing Set), Degree=&quot; </span><span class="s1">+ str(itr))</span>
    <span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_xlabel(</span><span class="s4">&quot;X_test&quot;</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">1</span><span class="s1">].set_ylabel(</span><span class="s4">&quot;y_test&quot;</span><span class="s1">)</span>
    <span class="s1">ax[</span><span class="s3">1</span><span class="s1">].legend()</span>

    <span class="s1">plt.tight_layout()</span>
    <span class="s1">plt.show()</span>

<span class="s1">print(</span><span class="s4">&quot;Best Values:   &quot;</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">&quot;Mean Square Error: &quot; </span><span class="s1">+ str(min_mse))</span>
<span class="s1">print(</span><span class="s4">&quot;Best Degree: &quot; </span><span class="s1">+ str(min_degree))</span>
<span class="s0"># ------------------ CROSS VALIDATION ENDS ------------------------</span>
</pre>
</body>
</html>