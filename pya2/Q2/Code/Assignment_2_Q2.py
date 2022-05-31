<html>
<head>
<title>TEST.py</title>
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
TEST.py</font>
</center></td></tr></table>
<pre><span class="s0">#  Required Libraries for Program</span>
<span class="s0"># Suggested running this program on PyCharm IDE</span>
<span class="s2">from </span><span class="s1">sklearn.datasets </span><span class="s2">import </span><span class="s1">make_classification</span>
<span class="s2">from </span><span class="s1">mlxtend.plotting </span><span class="s2">import </span><span class="s1">plot_decision_regions</span>
<span class="s2">import </span><span class="s1">matplotlib.pyplot </span><span class="s2">as </span><span class="s1">plot</span>
<span class="s2">import </span><span class="s1">numpy </span><span class="s2">as </span><span class="s1">num</span>
<span class="s2">from </span><span class="s1">sklearn.model_selection </span><span class="s2">import </span><span class="s1">cross_val_score</span>
<span class="s2">from </span><span class="s1">sklearn.model_selection </span><span class="s2">import </span><span class="s1">train_test_split</span>
<span class="s2">from </span><span class="s1">sklearn.linear_model </span><span class="s2">import </span><span class="s1">Perceptron</span>
<span class="s2">from </span><span class="s1">sklearn.preprocessing </span><span class="s2">import </span><span class="s1">StandardScaler</span>
<span class="s2">from </span><span class="s1">sklearn </span><span class="s2">import </span><span class="s1">metrics</span>

<span class="s0"># crate 5000 data points into two classes 2500 each class are represented by 0 and 1 in array.</span>
<span class="s0"># class 0 = 2500 data points | class 1 = 2500 data points</span>
<span class="s1">X</span><span class="s2">, </span><span class="s1">y = make_classification(</span>
    <span class="s1">n_samples=</span><span class="s3">5000</span><span class="s2">, </span><span class="s1">n_features=</span><span class="s3">2</span><span class="s2">,</span>
    <span class="s1">n_redundant=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">n_informative=</span><span class="s3">2</span><span class="s2">,</span>
    <span class="s1">n_clusters_per_class=</span><span class="s3">2</span><span class="s2">, </span><span class="s1">class_sep=</span><span class="s3">1.5</span><span class="s2">,</span>
    <span class="s1">flip_y=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">random_state=</span><span class="s3">0</span><span class="s2">, </span><span class="s1">shuffle=</span><span class="s2">False</span><span class="s1">)</span>

<span class="s0"># converting Class 0 into class -1 as required by the question</span>
<span class="s2">for </span><span class="s1">itr</span><span class="s2">, </span><span class="s1">j </span><span class="s2">in </span><span class="s1">enumerate(num.asarray(y)):</span>
    <span class="s2">if </span><span class="s1">j == </span><span class="s3">0</span><span class="s1">:</span>
        <span class="s1">y[itr] = -</span><span class="s3">1</span>

<span class="s0"># counting and separating class 1 and class -1</span>
<span class="s1">elements</span><span class="s2">, </span><span class="s1">elements_counts = num.unique(y</span><span class="s2">, </span><span class="s1">return_counts=</span><span class="s2">True</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\n</span><span class="s4">Frequency of unique class of the array:</span><span class="s2">\n</span><span class="s4">&quot;</span><span class="s1">)</span>
<span class="s1">print(num.asarray((elements</span><span class="s2">, </span><span class="s1">elements_counts)))</span>

<span class="s0"># plotting graph for the class -1 and class 1</span>
<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\n</span><span class="s4">Dataset of Class 1 and Class -1:</span><span class="s2">\n </span><span class="s4">&quot;</span><span class="s1">)</span>
<span class="s1">plot.plot(X[:</span><span class="s2">, </span><span class="s3">0</span><span class="s1">][y == -</span><span class="s3">1</span><span class="s1">]</span><span class="s2">, </span><span class="s1">X[:</span><span class="s2">, </span><span class="s3">1</span><span class="s1">][y == -</span><span class="s3">1</span><span class="s1">]</span><span class="s2">, </span><span class="s4">'g^'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">'Class: -1'</span><span class="s1">)</span>
<span class="s1">plot.plot(X[:</span><span class="s2">, </span><span class="s3">0</span><span class="s1">][y == </span><span class="s3">1</span><span class="s1">]</span><span class="s2">, </span><span class="s1">X[:</span><span class="s2">, </span><span class="s3">1</span><span class="s1">][y == </span><span class="s3">1</span><span class="s1">]</span><span class="s2">, </span><span class="s4">'o'</span><span class="s2">, </span><span class="s1">label=</span><span class="s4">&quot;Class: 1&quot;</span><span class="s1">)</span>
<span class="s1">plot.title(</span><span class="s4">&quot;Dataset&quot;</span><span class="s1">)</span>
<span class="s1">plot.xlabel(</span><span class="s4">&quot;X1&quot;</span><span class="s1">)</span>
<span class="s1">plot.xlabel(</span><span class="s4">&quot;X2&quot;</span><span class="s1">)</span>
<span class="s1">plot.legend()</span>
<span class="s1">plot.margins()</span>
<span class="s1">plot.show()</span>
<span class="s0"># ----------------- Classification END ----------------------------------------</span>


<span class="s0"># ------------------------------------------------------------------------------</span>
<span class="s0"># Implementing the perceptron on the data set created above</span>
<span class="s0"># Splitting class into train test dataset</span>
<span class="s1">X_train</span><span class="s2">, </span><span class="s1">X_test</span><span class="s2">, </span><span class="s1">y_train</span><span class="s2">, </span><span class="s1">y_test = train_test_split(X</span><span class="s2">, </span><span class="s1">y</span><span class="s2">, </span><span class="s1">test_size=</span><span class="s3">0.25</span><span class="s2">, </span><span class="s1">random_state=</span><span class="s3">0</span><span class="s1">)</span>

<span class="s0"># Scaling, fitting and transforming our data set into matrix for our model to reduce shape issues</span>
<span class="s1">sc = StandardScaler()</span>
<span class="s1">X_train = sc.fit_transform(X_train)</span>
<span class="s1">X_test = sc.transform(X_test)</span>

<span class="s0"># Training perceptron model using the training set we have creates</span>
<span class="s1">model = Perceptron(random_state=</span><span class="s3">42</span><span class="s2">, </span><span class="s1">alpha=</span><span class="s3">0.01</span><span class="s2">, </span><span class="s1">eta0=</span><span class="s3">0.2</span><span class="s2">, </span><span class="s1">max_iter=</span><span class="s3">100</span><span class="s1">)</span>
<span class="s1">model.fit(X_train</span><span class="s2">, </span><span class="s1">y_train)</span>

<span class="s0"># calculating performance of perceptron on training set</span>
<span class="s1">accuracy1 = cross_val_score(estimator=model</span><span class="s2">, </span><span class="s1">X=X_train</span><span class="s2">, </span><span class="s1">y=y_train</span><span class="s2">, </span><span class="s1">cv=</span><span class="s3">10</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\n</span><span class="s4">Accuracy and Variance on the Training dataset of Perceptron :</span><span class="s2">\n</span><span class="s4">&quot;</span><span class="s1">)</span>
<span class="s1">accuracy2 = accuracy1.mean() * </span><span class="s3">100</span>
<span class="s1">print(</span><span class="s4">'Mean Accuracy: %.2f' </span><span class="s1">% accuracy2</span><span class="s2">, </span><span class="s4">'%'</span><span class="s1">)</span>
<span class="s1">print(</span><span class="s4">&quot;Standard Deviation: &quot;</span><span class="s2">, </span><span class="s1">accuracy1.std())</span>

<span class="s1">print(</span><span class="s4">&quot;</span><span class="s2">\n</span><span class="s4">Result of Test dataset: </span><span class="s2">\n</span><span class="s4">&quot;</span><span class="s1">)</span>
<span class="s0"># Predicting the Test set results of perceptron</span>
<span class="s1">prediction = model.predict(X_test)</span>

<span class="s0"># Creating table to present the performance of a classification model</span>
<span class="s0"># confusion matrix</span>
<span class="s1">con_matrix = metrics.confusion_matrix(y_test</span><span class="s2">, </span><span class="s1">prediction)</span>
<span class="s1">print(</span><span class="s4">&quot;Confusion Matrix:</span><span class="s2">\n </span><span class="s4">&quot;</span><span class="s2">, </span><span class="s1">con_matrix)</span>
<span class="s1">print(</span><span class="s4">&quot;{0}&quot;</span><span class="s1">.format(metrics.classification_report(y_test</span><span class="s2">, </span><span class="s1">prediction)))</span>
<span class="s1">testing_accuracy = metrics.accuracy_score(y_test</span><span class="s2">, </span><span class="s1">prediction) * </span><span class="s3">100</span>
<span class="s1">print(</span><span class="s4">'Accuracy:%.2f' </span><span class="s1">% testing_accuracy</span><span class="s2">, </span><span class="s4">&quot;%</span><span class="s2">\n</span><span class="s4">&quot;</span><span class="s1">)</span>

<span class="s0"># plotting graph for the perception's result</span>
<span class="s1">fig</span><span class="s2">, </span><span class="s1">axes = plot.subplots(nrows=</span><span class="s3">1</span><span class="s2">, </span><span class="s1">ncols=</span><span class="s3">2</span><span class="s2">, </span><span class="s1">figsize=(</span><span class="s3">8</span><span class="s2">, </span><span class="s3">4</span><span class="s1">))</span>
<span class="s1">fig1 = plot_decision_regions(X_train</span><span class="s2">, </span><span class="s1">y_train</span><span class="s2">, </span><span class="s1">clf=model</span><span class="s2">, </span><span class="s1">ax=axes[</span><span class="s3">0</span><span class="s1">]</span><span class="s2">, </span><span class="s1">legend=</span><span class="s3">0</span><span class="s1">)</span>
<span class="s1">fig2 = plot_decision_regions(X_test</span><span class="s2">, </span><span class="s1">y_test</span><span class="s2">, </span><span class="s1">clf=model</span><span class="s2">, </span><span class="s1">ax=axes[</span><span class="s3">1</span><span class="s1">]</span><span class="s2">, </span><span class="s1">legend=</span><span class="s3">0</span><span class="s1">)</span>

<span class="s1">axes[</span><span class="s3">0</span><span class="s1">].set_title(</span><span class="s4">'Perceptron (Training set)'</span><span class="s1">)</span>
<span class="s1">axes[</span><span class="s3">0</span><span class="s1">].set_xlabel(</span><span class="s4">'x1'</span><span class="s1">)</span>
<span class="s1">axes[</span><span class="s3">0</span><span class="s1">].set_ylabel(</span><span class="s4">'x2'</span><span class="s1">)</span>
<span class="s1">axes[</span><span class="s3">1</span><span class="s1">].set_title(</span><span class="s4">'Perceptron (Test set)'</span><span class="s1">)</span>
<span class="s1">axes[</span><span class="s3">1</span><span class="s1">].set_xlabel(</span><span class="s4">'x1'</span><span class="s1">)</span>
<span class="s1">axes[</span><span class="s3">1</span><span class="s1">].set_ylabel(</span><span class="s4">'x2'</span><span class="s1">)</span>

<span class="s1">holder</span><span class="s2">, </span><span class="s1">labels = fig1.get_legend_handles_labels()</span>
<span class="s1">fig1.legend(holder</span><span class="s2">, </span><span class="s1">[</span><span class="s4">'class -1'</span><span class="s2">, </span><span class="s4">'class 1'</span><span class="s1">])</span>
<span class="s1">fig2.legend(holder</span><span class="s2">, </span><span class="s1">[</span><span class="s4">'class -1'</span><span class="s2">, </span><span class="s4">'class 1'</span><span class="s1">])</span>
<span class="s1">plot.tight_layout()</span>
<span class="s1">plot.show()</span>
<span class="s0"># --------------------------------- PERCEPTRON END -------------------------------</span>
</pre>
</body>
</html>