// questions-complexes.js
// Questions complexes sur l'IA

const questionsComplexes = [
    {
        question: "Comment XGBoost gère-t-il les problèmes de surapprentissage par rapport à un algorithme de boosting standard comme AdaBoost ?",
        options: [
            "XGBoost n'implémente aucune régularisation, il nécessite un élagage manuel des arbres",
            "XGBoost utilise uniquement un sous-échantillonnage des données pour éviter le surapprentissage",
            "XGBoost intègre une régularisation dans sa fonction objectif en pénalisant la complexité des arbres via le nombre de feuilles et la magnitude des poids",
            "XGBoost évite le surapprentissage uniquement en limitant le nombre d'itérations"
        ],
        correctIndex: 2,
        explanation: "XGBoost intègre explicitement la régularisation dans sa fonction objectif, contrairement aux algorithmes de boosting traditionnels. Sa fonction objectif inclut un terme de perte (erreur d'entraînement) et un terme de régularisation Ω(f) qui pénalise la complexité du modèle. Ce terme de régularisation est défini comme Ω(f) = γT + λ∑w² où T est le nombre de feuilles, w représente les poids des feuilles, γ et λ sont des hyperparamètres de régularisation. En plus, XGBoost implémente d'autres techniques comme le sous-échantillonnage des lignes/colonnes et l'arrêt précoce.",
        difficulty: "hard"
    },
    {
        question: "Quelle est la signification géométrique du paramètre C dans les SVM à marges souples (Soft Margin SVM) ?",
        options: [
            "C définit la distance minimale entre l'hyperplan et les vecteurs de support",
            "C contrôle le compromis entre la maximisation de la marge et la minimisation de l'erreur de classification",
            "C détermine le nombre maximum de vecteurs de support autorisés",
            "C spécifie la dimension du noyau utilisé pour la projection"
        ],
        correctIndex: 1,
        explanation: "Dans les SVM à marges souples, le paramètre C contrôle le compromis entre deux objectifs contradictoires : maximiser la marge (qui favorise la généralisation) et minimiser l'erreur de classification sur les données d'entraînement. Mathématiquement, le problème d'optimisation devient : min(w,b) (1/2)||w||² + C∑ξᵢ, où ξᵢ sont les variables d'écart qui permettent des violations de la marge. Un C faible permet plus de violations (marge plus large, plus d'erreurs), tandis qu'un C élevé pénalise davantage les erreurs (marge plus étroite, moins d'erreurs).",
        difficulty: "hard"
    },
    {
        question: "Comment la régularisation Ridge affecte-t-elle un modèle linéaire sujet au problème de multicolinéarité par rapport à une régression linéaire standard ?",
        options: [
            "Ridge n'a aucun effet sur la multicolinéarité, elle réduit uniquement le surapprentissage",
            "Ridge élimine complètement les variables redondantes en mettant leurs coefficients à zéro",
            "Ridge distribue les coefficients entre les variables corrélées au lieu d'attribuer un coefficient élevé à une seule variable",
            "Ridge augmente automatiquement les coefficients des variables les plus informatives"
        ],
        correctIndex: 2,
        explanation: "La multicolinéarité (forte corrélation entre prédicteurs) pose problème dans la régression linéaire standard car elle peut conduire à des coefficients instables et de grande magnitude. La régularisation Ridge ajoute une pénalité L2 (somme des carrés des coefficients) à la fonction objectif, ce qui a pour effet de rétrécir tous les coefficients vers zéro, mais sans jamais les annuler complètement. Pour des variables fortement corrélées, Ridge tend à distribuer le poids entre elles de manière plus équilibrée, au lieu d'attribuer un coefficient très élevé à l'une et très faible à l'autre, comme pourrait le faire la régression linéaire standard.",
        difficulty: "hard"
    },
    {
        question: "Pourquoi l'algorithme de backpropagation utilise-t-il le calcul du gradient par rétropropagation plutôt que par différenciation numérique directe ?",
        options: [
            "La rétropropagation est toujours plus précise mathématiquement que la différenciation numérique",
            "La différenciation numérique ne peut pas calculer les dérivées de fonctions d'activation non linéaires",
            "La rétropropagation est computationnellement beaucoup plus efficace grâce à la réutilisation des calculs intermédiaires",
            "La différenciation numérique nécessite des échantillons supplémentaires que la rétropropagation n'utilise pas"
        ],
        correctIndex: 2,
        explanation: "La différenciation numérique directe (approximation par différences finies) nécessiterait au minimum 2N évaluations du réseau pour calculer les dérivées partielles par rapport à N paramètres. La backpropagation, en revanche, calcule toutes ces dérivées en une seule passe arrière à travers le réseau, grâce à l'application répétée de la règle de chaîne et à la réutilisation des gradients intermédiaires. Cette optimisation algorithmique rend la backpropagation exponentiellement plus efficace en termes de calculs, surtout pour les réseaux avec millions de paramètres.",
        difficulty: "hard"
    },
    {
        question: "Si on veut estimer la sensibilité des modèles face à des données nouvelles plus précisément, quelle statistique peut-on utiliser dans le contexte de la décomposition biais-variance ?",
        options: [
            "Utiliser le coefficient de détermination (R²) sur les données de test pour mesurer la variabilité expliquée",
            "Calculer l'erreur quadratique moyenne (MSE) totale et la décomposer en (biais)² + variance + bruit irréductible",
            "Estimer la variance conditionnelle de l'erreur de prédiction par validation croisée k-fold",
            "Mesurer l'autocorrélation des résidus pour détecter les patterns non-capturés par le modèle"
        ],
        correctIndex: 1,
        explanation: "La décomposition de l'erreur de prédiction en biais et variance est précisément l'outil statistique pour comprendre la sensibilité des modèles face à de nouvelles données. L'erreur quadratique moyenne (MSE) peut être décomposée mathématiquement en trois termes : MSE = (Biais)² + Variance + Bruit irréductible. Le biais mesure l'erreur systématique du modèle (sous-ajustement), la variance mesure la sensibilité du modèle aux fluctuations des données d'entraînement (sur-ajustement), et le bruit irréductible représente la variabilité inhérente au problème. Cette décomposition permet d'identifier si un modèle échoue principalement à cause d'un biais élevé ou d'une variance élevée, guidant ainsi les efforts d'amélioration.",
        difficulty: "hard"
    },
    {
        question: "Quelle méthode permettrait d'estimer l'intervalle de confiance des prédictions d'un modèle Random Forest ?",
        options: [
            "def predict_with_confidence(forest, X_test, alpha=0.05):\n    # Prédictions individuelles de chaque arbre\n    preds = np.array([tree.predict(X_test) for tree in forest.estimators_])\n    \n    # Moyenne des prédictions (prédiction du RandomForest)\n    y_pred = np.mean(preds, axis=0)\n    \n    # Calcul de l'écart-type des prédictions\n    std = np.std(preds, axis=0)\n    \n    # Intervalle de confiance basé sur la distribution t\n    n_trees = len(forest.estimators_)\n    t_score = stats.t.ppf(1 - alpha/2, n_trees-1)\n    \n    # Limites de l'intervalle de confiance\n    lower = y_pred - t_score * std / np.sqrt(n_trees)\n    upper = y_pred + t_score * std / np.sqrt(n_trees)\n    \n    return y_pred, lower, upper",
            "def predict_with_confidence(forest, X_test):\n    # Obtenir les prédictions\n    y_pred = forest.predict(X_test)\n    \n    # Calculer la probabilité de la classe prédite\n    probas = forest.predict_proba(X_test)\n    confidence = np.max(probas, axis=1)\n    \n    return y_pred, confidence",
            "from sklearn.ensemble import BaggingRegressor\nfrom sklearn.tree import DecisionTreeRegressor\n\ndef bootstrap_prediction_intervals(X_train, y_train, X_test, n_estimators=100, alpha=0.05):\n    # Créer et entraîner un modèle de Bagging avec des arbres de décision\n    model = BaggingRegressor(\n        DecisionTreeRegressor(),\n        n_estimators=n_estimators,\n        bootstrap=True\n    )\n    model.fit(X_train, y_train)\n    \n    # Obtenir les prédictions de chaque estimateur sur les données de test\n    preds = np.array([est.predict(X_test) for est in model.estimators_])\n    \n    # Calcul des percentiles pour obtenir l'intervalle de confiance\n    lower = np.percentile(preds, alpha/2 * 100, axis=0)\n    upper = np.percentile(preds, (1 - alpha/2) * 100, axis=0)\n    y_pred = np.mean(preds, axis=0)\n    \n    return y_pred, lower, upper",
            "def jackknife_variance_estimation(model, X, y, X_test):\n    n = len(X)\n    predictions = []\n    \n    # Entraîner le modèle sur n sous-ensembles en excluant une observation à chaque fois\n    for i in range(n):\n        # Exclure l'observation i\n        X_jack = np.delete(X, i, axis=0)\n        y_jack = np.delete(y, i)\n        \n        # Entraîner le modèle et prédire\n        model.fit(X_jack, y_jack)\n        pred = model.predict(X_test)\n        predictions.append(pred)\n    \n    # Calculer la moyenne et la variance\n    predictions = np.array(predictions)\n    mean_pred = np.mean(predictions, axis=0)\n    \n    # Variance jackknife\n    var_jack = (n-1) * np.mean((predictions - mean_pred)**2, axis=0)\n    \n    # Écart-type\n    std_jack = np.sqrt(var_jack)\n    \n    # Intervalle de confiance à 95%\n    lower = mean_pred - 1.96 * std_jack\n    upper = mean_pred + 1.96 * std_jack\n    \n    return mean_pred, lower, upper"
        ],
        correctIndex: 2,
        explanation: "La méthode bootstrap est la plus appropriée pour estimer l'intervalle de confiance des prédictions d'un Random Forest. Cette approche exploite directement la nature d'ensemble du Random Forest en considérant la distribution des prédictions individuelles pour quantifier l'incertitude. En calculant les percentiles des prédictions (par exemple, 2.5ème et 97.5ème pour un intervalle de confiance à 95%), on obtient un intervalle non-paramétrique qui ne fait pas d'hypothèses sur la distribution sous-jacente des prédictions, contrairement à l'approche basée sur la distribution t.",
        difficulty: "hard"
    },
    {
        question: "Quelle implémentation de l'algorithme de bootstrap permettrait de calculer l'intervalle de confiance pour le score d'un modèle de machine learning ?",
        options: [
            "def bootstrap_confidence_interval(model, X, y, n_bootstrap=1000, alpha=0.05):\n    n_samples = len(X)\n    scores = []\n    \n    for _ in range(n_bootstrap):\n        # Échantillonnage avec remise\n        indices = np.random.choice(n_samples, n_samples, replace=True)\n        X_boot, y_boot = X[indices], y[indices]\n        \n        # Calcul du score sur l'échantillon bootstrap\n        score = model.score(X_boot, y_boot)\n        scores.append(score)\n    \n    # Calcul des percentiles pour l'intervalle de confiance\n    lower = np.percentile(scores, alpha/2 * 100)\n    upper = np.percentile(scores, (1 - alpha/2) * 100)\n    point_estimate = np.mean(scores)\n    \n    return point_estimate, lower, upper",
            "def bootstrap_confidence_interval(model, X, y, n_bootstrap=1000, alpha=0.05):\n    scores = []\n    \n    for _ in range(n_bootstrap):\n        # Créer un nouvel échantillon d'apprentissage avec remplacement\n        indices = np.random.choice(len(X), len(X), replace=True)\n        X_train, y_train = X[indices], y[indices]\n        \n        # Calculer les indices hors-sac (out-of-bag)\n        oob_indices = np.array([i for i in range(len(X)) if i not in indices])\n        \n        if len(oob_indices) > 0:\n            X_test, y_test = X[oob_indices], y[oob_indices]\n            \n            # Entraîner un nouveau modèle sur les données bootstrap\n            model_boot = clone(model)\n            model_boot.fit(X_train, y_train)\n            \n            # Évaluer sur les données hors-sac\n            score = model_boot.score(X_test, y_test)\n            scores.append(score)\n    \n    # Calcul de l'intervalle de confiance\n    lower = np.percentile(scores, alpha/2 * 100)\n    upper = np.percentile(scores, (1 - alpha/2) * 100)\n    point_estimate = np.mean(scores)\n    \n    return point_estimate, lower, upper",
            "from scipy import stats\n\ndef parametric_confidence_interval(model, X, y, alpha=0.05):\n    # Calculer le score du modèle\n    score = model.score(X, y)\n    \n    # Calculer l'erreur standard approximative\n    n_samples = len(X)\n    std_error = np.sqrt(score * (1 - score) / n_samples)\n    \n    # Calculer l'intervalle de confiance paramétrique\n    z_score = stats.norm.ppf(1 - alpha/2)\n    margin_of_error = z_score * std_error\n    \n    lower = max(0, score - margin_of_error)\n    upper = min(1, score + margin_of_error)\n    \n    return score, lower, upper",
            "from sklearn.model_selection import KFold\n\ndef cross_validation_confidence_interval(model, X, y, n_splits=10, alpha=0.05):\n    # Validation croisée\n    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)\n    scores = []\n    \n    for train_idx, val_idx in kf.split(X):\n        X_train, X_val = X[train_idx], X[val_idx]\n        y_train, y_val = y[train_idx], y[val_idx]\n        \n        # Entrainer et évaluer le modèle\n        model_cv = clone(model)\n        model_cv.fit(X_train, y_train)\n        score = model_cv.score(X_val, y_val)\n        scores.append(score)\n    \n    # Calcul de l'intervalle de confiance\n    mean_score = np.mean(scores)\n    t_score = stats.t.ppf(1 - alpha/2, n_splits-1)\n    std_error = np.std(scores, ddof=1) / np.sqrt(n_splits)\n    \n    lower = mean_score - t_score * std_error\n    upper = mean_score + t_score * std_error\n    \n    return mean_score, lower, upper"
        ],
        correctIndex: 1,
        explanation: "La deuxième option est la plus appropriée pour estimer l'intervalle de confiance par bootstrap car elle utilise l'approche out-of-bag (OOB). Quand on échantillonne avec remise pour créer un ensemble bootstrap, environ 63.2% des observations originales sont incluses, laissant ~36.8% des données non utilisées (hors-sac). Ces données OOB fournissent un ensemble de test indépendant, ce qui est crucial pour une estimation non biaisée de la performance du modèle. En répétant ce processus de nombreuses fois, on obtient une distribution d'estimations de performance qui reflète la variabilité due à l'échantillonnage, permettant de calculer des intervalles de confiance robustes sans hypothèses paramétriques.",
        difficulty: "hard"
    }
];

// Exporter les questions
if (typeof module !== 'undefined') {
    module.exports = { questionsComplexes };
}